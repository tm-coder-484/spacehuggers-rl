"""
SpaceHuggers VecEnv — single Node.js process, N parallel game instances.

One pipe round-trip per batch step instead of N separate round-trips.
No SubprocVecEnv, no worker processes — Python + Node.js in the same process.

Expected throughput: 400-600+ sps at 12 envs (bottleneck shifts to PPO update math).
Memory: ~500 MB total for the Node process vs ~160 MB × N with SubprocVecEnv.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

N_ENEMIES      = 5
GRID_W, GRID_H = 13, 9
OBS_DIM        = 9 + N_ENEMIES * 3 + GRID_W * GRID_H   # 141  (v1)

# ── v2 observation (Tier-2: perception tuned to the actual combat model) ──────
# Bullets fire HORIZONTALLY only, so facing + vertical alignment + incoming-fire
# awareness are what matter.  Layout:
#   player(9) + progress(3) + facing(1)
#   + 10 enemies×3 (pos+health) + 10 enemies×2 (velocity) + 10 shootable flags
#   + 5 incoming bullets×4 (pos+velocity)
#   + grid(117)
N_ENEMIES_V2   = 10
N_BULLETS_V2   = 5
# v2 is a Dict obs: local CNN grid + coarse global map + a flat vector.
# grid channels: solid, ladder, enemy, hazard, line-of-fire, reachability(geodesic).
GRID_CH_V2     = 6
GLOBE_CH       = 3    # coarse global map channels: terrain, enemies-now, enemy-memory
GLOBE_SZ       = 8    # 8x8 egocentric cells
VEC_DIM_V2     = (9 + 3 + 1
                 + N_ENEMIES_V2 * 3 + N_ENEMIES_V2 * 2 + N_ENEMIES_V2
                 + N_BULLETS_V2 * 4
                 + 3                                                    # geo_dist + geo_dir x2
                 + N_ENEMIES_V2)                                        # 106 (+ aiming flags)

# Dig-progress reward (v2): a single-step geo_dist drop bigger than walking can
# explain means a path-blocking tile was just breached toward an enemy.
DIG_MIN_DROP = 0.06    # walking moves <=0.6 tile/step ~0.012 geo; dirt breach ~0.10, hard ~0.42
DIG_SCALE    = 1.4     # reward per unit geo_dist closed by a breach
DIG_STEP_CAP = 0.6     # max dig bonus from a single breach (avoids spikes on big shortcuts)
DIG_CAP_EP   = 20.0    # max total dig bonus per episode (anti-runaway-tunnelling)

_GAME_SERVER = os.path.join(os.path.dirname(__file__), 'game_server_workers.js')

# ── CPU affinity (P-core / E-core) detection ─────────────────────────────────
#
# On Intel hybrid CPUs (Alder/Raptor/Meteor Lake), P-cores are ~2× faster than
# E-cores for our workload.  Pinning the Python + Node processes to P-cores
# (a) keeps workers off the slow E-cores, and (b) more importantly, forces
# competing apps (Opera, etc.) onto E-cores so they can't steal P-core cycles
# mid-step.  Result: deterministic throughput instead of fighting the scheduler.
#
# Detection runs a CPU-bound microbench on each logical CPU, finds the biggest
# relative gap in sorted timings, and splits there.  Cached to disk keyed by
# CPU fingerprint so it only runs once per machine.

_LAYOUT_CACHE = Path(__file__).with_name('.cpu_layout.json')


def detect_cpu_layout(force: bool = False) -> dict:
    """Detect P-cores and E-cores. Cached. Returns {'p_cores':[...], 'e_cores':[...], 'all':[...]}."""
    import platform
    import psutil

    fingerprint = f'{platform.processor()}|{psutil.cpu_count()}'

    if not force and _LAYOUT_CACHE.exists():
        try:
            cached = json.loads(_LAYOUT_CACHE.read_text())
            if cached.get('fingerprint') == fingerprint:
                return cached
        except Exception:
            pass

    print('[cpu_layout] detecting P/E cores (one-time, ~5 s)...')
    self_proc = psutil.Process()
    original  = self_proc.cpu_affinity()

    # Warm-up to settle JIT / freq scaling
    self_proc.cpu_affinity([0])
    for _ in range(2):
        x = 0
        for i in range(1_000_000): x += i*i

    n = psutil.cpu_count()
    times = []
    for cpu in range(n):
        self_proc.cpu_affinity([cpu])
        runs = []
        for _ in range(3):  # min-of-3 to defeat scheduler noise
            t0 = time.perf_counter()
            x = 0
            for i in range(1_500_000): x += i*i
            runs.append(time.perf_counter() - t0)
        times.append(min(runs))

    self_proc.cpu_affinity(original)

    # Find the largest gap in sorted timings → splits P-cores from E-cores.
    # If the gap is small (<30%), assume a uniform CPU (no hybrid layout).
    indexed = sorted(enumerate(times), key=lambda x: x[1])
    gaps = [(indexed[i+1][1] / indexed[i][1], i) for i in range(len(indexed) - 1)]
    biggest_gap, split_idx = max(gaps) if gaps else (1.0, -1)

    if biggest_gap > 1.3:
        p_cores = sorted(c for c, _ in indexed[:split_idx + 1])
        e_cores = sorted(c for c, _ in indexed[split_idx + 1:])
    else:
        p_cores = list(range(n))
        e_cores = []

    layout = {
        'fingerprint': fingerprint,
        'p_cores':     p_cores,
        'e_cores':     e_cores,
        'all':         list(range(n)),
    }
    try:
        _LAYOUT_CACHE.write_text(json.dumps(layout, indent=2))
    except Exception:
        pass
    print(f'[cpu_layout] P-cores: {p_cores}  |  E-cores: {e_cores}')
    return layout


def resolve_pin(kind: str) -> list[int] | None:
    """Translate a --pin choice into a CPU list. 'none' returns None."""
    if kind in (None, 'none', ''):
        return None
    layout = detect_cpu_layout()
    if kind == 'p':
        cpus = layout['p_cores']
    elif kind == 'e':
        cpus = layout['e_cores'] or layout['all']  # fall back if no E-cores
    elif kind == 'all':
        cpus = layout['all']
    else:
        raise ValueError(f'unknown --pin value: {kind}')
    return cpus if cpus else None


def _to_obs(s: dict | None) -> np.ndarray:
    if s is None:
        return np.zeros(OBS_DIM, dtype=np.float32)
    obs = [
        s['px'], s['py'], s['vx'], s['vy'],
        s['health'], s['ground'], s['grenades'], s['dodge_ready'], s['on_fire'],
    ]
    for i in range(N_ENEMIES):
        obs.extend(s['enemies'][i] if i < len(s['enemies']) else [0.0, 0.0, 0.0])
    obs.extend(s.get('grid') or [0.0] * (GRID_W * GRID_H))
    return np.clip(np.array(obs, dtype=np.float32), -5.0, 5.0)


def _to_obs_v2(s: dict | None) -> dict:
    """Tier-2 observation: a multi-channel 2D grid (for a CNN) + a flat vector."""
    if s is None:
        return {
            'grid':   np.zeros((GRID_CH_V2, GRID_H, GRID_W), dtype=np.float32),
            'globe':  np.zeros((GLOBE_CH, GLOBE_SZ, GLOBE_SZ), dtype=np.float32),
            'vector': np.zeros(VEC_DIM_V2, dtype=np.float32),
        }

    # ── flat vector stream ────────────────────────────────────────────────────
    vec = [
        s['px'], s['py'], s['vx'], s['vy'],
        s['health'], s['ground'], s['grenades'], s['dodge_ready'], s['on_fire'],
        s.get('level', 1) / 10.0,
        s.get('enemies_remaining', 0) / 100.0,
        s.get('n_enemies', 0) / 10.0,
    ]
    facing = s.get('facing', 1)
    vec.append(facing)
    en = s.get('enemies') or []
    ev = s.get('enemy_vels') or []
    sh = s.get('enemy_shootable') or []
    for i in range(N_ENEMIES_V2):
        vec.extend(en[i] if i < len(en) else [0.0, 0.0, 0.0])
    for i in range(N_ENEMIES_V2):
        vec.extend(ev[i] if i < len(ev) else [0.0, 0.0])
    for i in range(N_ENEMIES_V2):
        vec.append(sh[i] if i < len(sh) else 0.0)
    bl = s.get('bullets') or []
    for i in range(N_BULLETS_V2):
        vec.extend(bl[i] if i < len(bl) else [0.0, 0.0, 0.0, 0.0])
    vec.append(s.get('geo_dist', 1.0))            # geodesic dist to nearest enemy
    gdir = s.get('geo_dir') or [0.0, 0.0]         # flow-field step direction
    vec.extend(gdir[:2] if len(gdir) >= 2 else [0.0, 0.0])
    am = s.get('enemy_aiming') or []              # which enemies are aiming at us
    for i in range(N_ENEMIES_V2):
        vec.append(am[i] if i < len(am) else 0.0)
    vector = np.clip(np.array(vec, dtype=np.float32), -5.0, 5.0)

    # ── multi-channel grid stream (CNN reads terrain + enemies + hazards) ──────
    terrain = np.array(s.get('grid_terrain') or [0.0] * (GRID_W * GRID_H),
                       dtype=np.float32).reshape(GRID_H, GRID_W)
    enemy   = np.array(s.get('grid_enemy') or [0.0] * (GRID_W * GRID_H),
                       dtype=np.float32).reshape(GRID_H, GRID_W)
    hazard  = np.array(s.get('grid_hazard') or [0.0] * (GRID_W * GRID_H),
                       dtype=np.float32).reshape(GRID_H, GRID_W)
    reach   = np.array(s.get('grid_reach') or [1.0] * (GRID_W * GRID_H),
                       dtype=np.float32).reshape(GRID_H, GRID_W)  # geodesic dist-to-enemy
    solid   = np.maximum(terrain, 0.0)            # walls / glass
    ladder  = (terrain < -0.5).astype(np.float32)  # climbable
    # line-of-fire: horizontal ray from player (window centre) in facing dir
    lof = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    cy, cx = GRID_H // 2, GRID_W // 2              # (4, 6)
    for k in range(1, cx + 1):
        c = cx + int(facing) * k
        if 0 <= c < GRID_W:
            lof[cy, c] = 1.0
    grid = np.stack([solid, ladder, enemy, hazard, lof, reach], axis=0)  # (6,9,13)

    # coarse global map: flat 192 from JS -> (3,8,8)
    gflat = s.get('globe') or [0.0] * (GLOBE_CH * GLOBE_SZ * GLOBE_SZ)
    globe = np.array(gflat, dtype=np.float32).reshape(GLOBE_CH, GLOBE_SZ, GLOBE_SZ)
    return {'grid': grid, 'globe': globe, 'vector': vector}


class NodeBatchVecEnv(VecEnv):
    """
    VecEnv backed by ONE Node.js process running N game instances.

    Compared to SubprocVecEnv + NodeEnv:
      - 1 pipe round-trip/step  (vs N concurrent round-trips)
      - No Python worker processes (saves ~250 MB/env of Python overhead)
      - Startup: one Node.js process initialises N vm contexts sequentially
        (~2-3 s for 12 instances vs ~3 s for 6 separate processes in parallel)
    """

    def __init__(self, game_path: str, n_envs: int = 6, action_repeat: int = 3,
                 pin_cpus: list[int] | None = None, max_steps: int = 30_000,
                 obs_version: int = 1):
        self.obs_version = obs_version
        if obs_version == 2:
            self._obs_fn = _to_obs_v2
            obs_space = gym.spaces.Dict({
                'grid':   gym.spaces.Box(-1.0, 1.0, (GRID_CH_V2, GRID_H, GRID_W), dtype=np.float32),
                'globe':  gym.spaces.Box(0.0, 1.0, (GLOBE_CH, GLOBE_SZ, GLOBE_SZ), dtype=np.float32),
                'vector': gym.spaces.Box(-5.0, 5.0, (VEC_DIM_V2,), dtype=np.float32),
            })
        else:
            self._obs_fn = _to_obs
            obs_space = gym.spaces.Box(low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32)
        act_space = gym.spaces.MultiDiscrete([3, 3, 2, 2, 2])
        super().__init__(n_envs, obs_space, act_space)

        self.game_path     = str(Path(game_path).resolve())
        self.action_repeat = max(1, action_repeat)
        self.pin_cpus      = pin_cpus    # list of logical CPU indices, or None
        self.max_steps     = max_steps   # truncate an episode after this many steps
        self._proc: subprocess.Popen | None = None
        self._pending_actions = None

        # Per-env episode tracking
        self._last_kills  = [0]    * n_envs
        self._last_lives  = [10]   * n_envs
        self._last_level  = [1]    * n_envs
        self._last_edist  = [None] * n_envs   # approach-reward distance
        self._step_counts = [0]    * n_envs
        self._ep_rewards  = [0.0]  * n_envs
        self._ep_lengths  = [0]    * n_envs
        self._visited     = [set() for _ in range(n_envs)]   # exploration (v2)
        self._last_geodist = [None] * n_envs                 # geodesic approach (v2)
        self._last_health  = [None] * n_envs                 # health-loss penalty (v2)
        self._steps_since_level = [0] * n_envs               # time-to-clear bonus (v2)
        self._last_grenades = [None] * n_envs                # grenade-throw reward (v2)
        self._dig_acc       = [0.0]  * n_envs                # dig-progress bonus accumulator (v2, capped/episode)
        self._last_cdist    = [None] * n_envs                # long-range coarse-field approach (v2, when fine geo saturated)

        # Optional profiling (set NODEBATCH_PROFILE=1).  Accumulates time spent
        # in the step_batch RPC (pipe + Node main thread + parallel workers)
        # vs. Python-side reward/obs/done post-processing.
        self._prof = os.environ.get('NODEBATCH_PROFILE') == '1'
        self._prof_rpc = 0.0
        self._prof_post = 0.0
        self._prof_reset = 0.0
        self._prof_n = 0

        self._launch()

    # ── process lifecycle ────────────────────────────────────────────────────

    def _launch(self):
        if self._proc is not None:
            try: self._proc.stdin.close()
            except Exception: pass
            try: self._proc.kill(); self._proc.wait(timeout=3)
            except Exception: pass

        env = dict(os.environ, GAME_PATH=self.game_path, N_GAMES=str(self.num_envs))
        if self.obs_version == 2:
            env['GEO'] = '1'   # enable geodesic flow-field BFS in the worker
        self._proc = subprocess.Popen(
            ['node', '--max-old-space-size=2048', '--v8-pool-size=0', _GAME_SERVER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, bufsize=0,
        )
        # Pin the Node process (and all its worker threads, since process
        # affinity covers all threads on Windows) to the requested CPUs.
        if self.pin_cpus:
            try:
                import psutil
                psutil.Process(self._proc.pid).cpu_affinity(self.pin_cpus)
                sys.stderr.write(f'[NodeBatchVecEnv] Node pinned to CPUs {self.pin_cpus}\n')
            except Exception as e:
                sys.stderr.write(f'[NodeBatchVecEnv] failed to pin Node ({e}) — continuing unpinned\n')
        threading.Thread(target=self._drain_stderr, daemon=True).start()

        # Wait for all N worker threads to initialise (startup can take 10-30 s
        # for 12 workers; they run on separate OS threads so some parallelism
        # applies, but each worker still reads game files from disk sequentially).
        deadline = time.time() + 120.0
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError('game_server_workers.js exited during startup')
            try:
                self._proc.stdin.write(b'{"type":"ping"}\n')
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
                if line and json.loads(line).get('type') == 'pong':
                    return
            except Exception:
                time.sleep(0.3)
        raise RuntimeError('game_server startup timeout (120 s)')

    def _drain_stderr(self):
        try:
            for raw in self._proc.stderr:
                sys.stderr.write('[node] ' + raw.decode(errors='replace').rstrip() + '\n')
        except Exception:
            pass

    # ── low-level RPC ────────────────────────────────────────────────────────

    def _rpc(self, msg: dict) -> dict:
        data = (json.dumps(msg) + '\n').encode()
        self._proc.stdin.write(data)
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise EOFError('game_server stdout closed unexpectedly')
        return json.loads(line.decode())

    # ── reward computation (identical weights to env_node.py) ────────────────

    def _reward(self, i: int, s: dict | None, action) -> float:
        kills     = s['kills']     if s else self._last_kills[i]
        lives     = s['lives']     if s else self._last_lives[i]
        level     = s['level']     if s else self._last_level[i]
        n_enemies = s['n_enemies'] if s else 0

        delta_kills = kills - self._last_kills[i]
        delta_lives = max(0, self._last_lives[i] - lives)
        delta_level = level - self._last_level[i]

        kill_r  = delta_kills * 10.0
        death_p = delta_lives * 5.0
        level_r = delta_level * 150.0

        ey = 0.0
        approach_r = 0.0
        if n_enemies > 0 and s and s.get('enemies'):
            ex, ey, _ = s['enemies'][0]
            dist_now = (ex*ex + ey*ey) ** 0.5
            if self._last_edist[i] is not None and delta_kills == 0:
                approach_r = max(min((self._last_edist[i] - dist_now) * 2.0, 0.05), -0.05)
            self._last_edist[i] = dist_now
        else:
            self._last_edist[i] = None

        on_fire   = s.get('on_fire', 0) if s else 0
        on_ground = bool(s.get('ground', 1)) if s else True
        grid      = s.get('grid', []) if s else []

        dodge_act = int(action[3])
        vert_act  = int(action[1])
        horiz_act = int(action[0])
        shoot_act = int(action[2])

        fire_r    = 0.1 if (on_fire and dodge_act) else 0.0
        on_ladder = len(grid) > 58 and grid[58] < -0.5
        ladder_r  = (0.015 if (n_enemies > 0 and ey > 0.05) else 0.003) \
                    if (on_ladder and vert_act != 0) else 0.0

        wall_left  = len(grid) > 57 and grid[57] > 0.4
        wall_right = len(grid) > 59 and grid[59] > 0.4
        pressing   = (wall_left and horiz_act == 1) or (wall_right and horiz_act == 2)
        wall_r     = 0.01 if (pressing and not on_ground and not on_ladder
                               and vert_act == 1 and shoot_act == 0) else 0.0

        self._last_kills[i] = kills
        self._last_lives[i] = lives
        self._last_level[i] = level

        # Survival bonus removed (was +0.002/step).  Over the ~28k-step episodes
        # it paid ~56 reward for merely staying alive, which fueled the passive
        # "turtle" optimum.  Death is still discouraged by the -5 life penalty
        # and the lost opportunity to earn kills/levels, so the agent still has
        # every reason to avoid dying — it's just no longer PAID to stall.
        total = kill_r - death_p + level_r + approach_r + fire_r + ladder_r + wall_r

        # ── v2-only shaping (needs the v2 obs fields) ────────────────────────
        if self.obs_version == 2 and s:
            # NOTE: a per-step "+0.02 while an enemy is shootable" firing bonus was
            # REMOVED — it was farmable (camp in position for +600/episode with 0
            # kills). Kills (+10) + geodesic approach already pull it into position.
            # exploration: reward visiting new areas (anti-camping). Quantise the
            # normalised position into a coarse grid; reward each new cell, capped.
            bx = int(min(max(s.get('px', 0.0), 0.0), 1.0) * 100)
            by = int(min(max(s.get('py', 0.0), 0.0), 1.0) * 100)
            cell = (bx, by)
            if cell not in self._visited[i]:
                self._visited[i].add(cell)
                if len(self._visited[i]) <= 300:
                    total += 0.02
            # geodesic approach: reward reducing distance-through-terrain to the
            # nearest enemy (terrain-aware replacement for Euclidean approach_r).
            gd = s.get('geo_dist', 1.0)
            if self._last_geodist[i] is not None:
                drop = self._last_geodist[i] - gd
                # normal approach (telescoping, bounded)
                total += max(min(drop * 3.0, 0.05), -0.05)
                # DIG-PROGRESS: a one-step geo_dist drop larger than walking can explain
                # means a path-blocking tile was just breached toward the nearest enemy.
                # Pays for the ~20-shot dig grind (otherwise unrewarded until the sparse
                # +150 clear). Tied to real path progress -> off-path digging gives no
                # geo drop -> no reward, so it's non-farmable; bounded + capped/episode.
                if drop > DIG_MIN_DROP and self._dig_acc[i] < DIG_CAP_EP:
                    b = min(drop * DIG_SCALE, DIG_STEP_CAP)
                    total += b
                    self._dig_acc[i] += b
            self._last_geodist[i] = gd

            # LONG-RANGE approach: the fine geodesic field only sees ~24 tiles; on
            # wide levels the last enemies sit beyond it (geo_dist pinned at 1.0).
            # The COARSE whole-level field then routes the long way around, and we
            # reward reducing the COARSE distance — which is detour-aware. (Plain
            # straight-line Euclidean would PUNISH the temporary "away" leg of a
            # detour, e.g. the right-50 part of right-50/up/left-50.) Telescoping,
            # bounded -> non-farmable.
            if gd >= 0.999:
                cdist = s.get('coarse_dist', 1.0)
                if self._last_cdist[i] is not None:
                    total += max(min((self._last_cdist[i] - cdist) * 3.0, 0.05), -0.05)
                self._last_cdist[i] = cdist
            else:
                self._last_cdist[i] = None

            # #26 health-loss penalty: dense signal to avoid damage (finer than
            # the sparse -5 death). health is per-life HP.
            hp = s.get('health', 0)
            if self._last_health[i] is not None and hp < self._last_health[i]:
                total -= (self._last_health[i] - hp) * 1.0
            self._last_health[i] = hp

            # #28 time-to-clear: bonus on level-up, larger the faster it cleared.
            self._steps_since_level[i] += 1
            if delta_level > 0:
                total += max(0.0, 40.0 * (1.0 - self._steps_since_level[i] / 8000.0))
                self._steps_since_level[i] = 0

            # #29 grenade-for-elevated reward: REMOVED. The flat +0.5 per throw was
            # being front-loaded — the agent dumped all 3 grenades at the start of
            # every life for a guaranteed +1.5, cratering the ground beneath itself
            # (grenades dig DOWNWARD, which bullets can't) and getting buried/killed.
            # Ammo-limiting didn't help (it just threw all 3 at once). Grenade use is
            # now driven only by the kill reward (+10): throw one when it helps kill.
            # (_last_grenades tracker left in place but unused — harmless.)

        return total

    # ── obs assembly (handles flat v1 array or Dict v2 obs) ──────────────────

    def _alloc_obs(self):
        if self.obs_version == 2:
            return {
                'grid':   np.zeros((self.num_envs, GRID_CH_V2, GRID_H, GRID_W), dtype=np.float32),
                'globe':  np.zeros((self.num_envs, GLOBE_CH, GLOBE_SZ, GLOBE_SZ), dtype=np.float32),
                'vector': np.zeros((self.num_envs, VEC_DIM_V2), dtype=np.float32),
            }
        return np.zeros((self.num_envs, OBS_DIM), dtype=np.float32)

    def _set_obs(self, obs, i, single):
        if self.obs_version == 2:
            obs['grid'][i]   = single['grid']
            obs['globe'][i]  = single['globe']
            obs['vector'][i] = single['vector']
        else:
            obs[i] = single

    # ── VecEnv interface ─────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        resp   = self._rpc({'type': 'reset_all'})
        states = resp['states']
        obs    = self._alloc_obs()
        for i, s in enumerate(states):
            self._set_obs(obs, i, self._obs_fn(s))
            self._last_kills[i]  = 0
            self._last_lives[i]  = s['lives'] if s else 10
            self._last_level[i]  = 1
            self._last_edist[i]  = None
            self._step_counts[i] = 0
            self._ep_rewards[i]  = 0.0
            self._ep_lengths[i]  = 0
            self._visited[i]     = set()
            self._last_geodist[i] = None
            self._last_health[i] = None
            self._steps_since_level[i] = 0
            self._last_grenades[i] = None
            self._dig_acc[i] = 0.0
            self._last_cdist[i] = None
        return obs

    def step_async(self, actions):
        self._pending_actions = actions

    def step_wait(self):
        _t0 = time.perf_counter() if self._prof else 0.0
        actions     = self._pending_actions
        action_list = [[int(x) for x in a] for a in actions]

        resp   = self._rpc({'type': 'step_batch', 'actions': action_list, 'n': self.action_repeat})
        states = resp['states']
        _t1 = time.perf_counter() if self._prof else 0.0

        obs     = self._alloc_obs()
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones   = np.zeros(self.num_envs, dtype=bool)
        infos   = [{} for _ in range(self.num_envs)]
        done_indices = []

        for i, (s, action) in enumerate(zip(states, actions)):
            single     = self._obs_fn(s)
            self._set_obs(obs, i, single)
            rewards[i] = self._reward(i, s, action)
            self._step_counts[i] += 1
            self._ep_rewards[i]  += float(rewards[i])
            self._ep_lengths[i]  += 1

            alive = s['alive'] if s else True
            lives = s['lives'] if s else self._last_lives[i]
            level = s['level'] if s else self._last_level[i]
            died      = (lives <= 0 and not alive)            # true terminal
            truncated = self._step_counts[i] >= self.max_steps  # timeout
            done      = died or truncated

            dones[i] = done
            infos[i] = {
                'kills': self._last_kills[i],
                'lives': lives,
                'level': level,
            }
            if done:
                # SB3 convention: 'episode' dict triggers episode logging
                infos[i]['episode'] = {
                    'r': self._ep_rewards[i],
                    'l': self._ep_lengths[i],
                    't': time.time(),
                }
                # Terminal obs stored so SB3 can compute correct value targets
                infos[i]['terminal_observation'] = (
                    {'grid': single['grid'].copy(), 'globe': single['globe'].copy(),
                     'vector': single['vector'].copy()}
                    if self.obs_version == 2 else obs[i].copy())
                # On TIMEOUT (not death) tell SB3 to BOOTSTRAP the value of the
                # final state instead of treating it as a hard terminal — the
                # episode didn't really "end", it was cut off.  Critical once the
                # cap is low enough that most episodes truncate; without this the
                # value targets are biased low for survivable states.
                if truncated and not died:
                    infos[i]['TimeLimit.truncated'] = True
                done_indices.append(i)

        _t2 = time.perf_counter() if self._prof else 0.0

        # Auto-reset done envs (rare — one extra round-trip only when needed)
        if done_indices:
            resp2   = self._rpc({'type': 'reset_batch', 'indices': done_indices})
            states2 = resp2['states']
            for j, i in enumerate(done_indices):
                s2     = states2[j]
                self._set_obs(obs, i, self._obs_fn(s2))
                self._last_kills[i]  = 0
                self._last_lives[i]  = s2['lives'] if s2 else 10
                self._last_level[i]  = 1
                self._last_edist[i]  = None
                self._step_counts[i] = 0
                self._ep_rewards[i]  = 0.0
                self._ep_lengths[i]  = 0
                self._visited[i]     = set()
                # reset v2 shaping trackers too (prevents a phantom geo_dist "drop"
                # across the episode boundary from firing a false dig bonus)
                self._last_geodist[i] = None
                self._last_health[i] = None
                self._steps_since_level[i] = 0
                self._last_grenades[i] = None
                self._dig_acc[i] = 0.0
                self._last_cdist[i] = None

        if self._prof:
            self._prof_rpc   += _t1 - _t0
            self._prof_post  += _t2 - _t1
            self._prof_reset += time.perf_counter() - _t2
            self._prof_n     += 1

        return obs, rewards, dones, infos

    def close(self):
        if self._proc is not None:
            try: self._proc.stdin.close()
            except Exception: pass
            try: self._proc.kill(); self._proc.wait(timeout=3)
            except Exception: pass
            self._proc = None

    # ── required VecEnv stubs (SB3 calls these for Monitor compatibility) ────

    def env_method(self, method_name, *method_args, indices=None):
        return [None] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
