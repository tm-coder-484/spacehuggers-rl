"""
SpaceHuggers gymnasium environment — Node.js headless backend.

Instead of launching a Chromium browser, this env starts game_server.js as a
subprocess and communicates via stdin/stdout newline-delimited JSON.

Expected speedup: 20-100× over Playwright (no 50 ms sleep, no browser overhead).
Memory: ~80 MB per env vs ~500 MB per Chromium instance.

Observations and action space are identical to env.py (drop-in replacement).
"""

import json
import os
import subprocess
import sys
import threading   # used for stderr drain only
import time
from pathlib import Path

import numpy as np
import gymnasium as gym

# ── observation / action constants (must match env.py) ───────────────────────
N_ENEMIES  = 5
GRID_W, GRID_H = 13, 9
OBS_DIM    = 9 + N_ENEMIES * 3 + GRID_W * GRID_H   # 9 + 15 + 117 = 141

_GAME_SERVER = os.path.join(os.path.dirname(__file__), "game_server.js")


class NodeEnv(gym.Env):
    """SpaceHuggers via Node.js subprocess — fast headless backend."""

    metadata = {"render_modes": ["none"], "render_fps": 60}

    def __init__(self,
                 game_path: str,
                 restart_every: int = 500,
                 step_timeout: float = 5.0):
        """
        game_path     : path to SpaceHuggers-main directory
        restart_every : restart the Node process every N episodes
                        (prevents memory growth; 0 = never restart)
        step_timeout  : seconds before a step() is considered hung
        """
        super().__init__()
        self.game_path     = str(Path(game_path).resolve())
        self.restart_every = restart_every
        self.step_timeout  = step_timeout

        self.action_space = gym.spaces.MultiDiscrete([3, 3, 2, 2, 2])
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self._proc: subprocess.Popen | None = None

        # episode state
        self._last_kills      = 0
        self._last_lives      = 6
        self._last_level      = 1
        self._last_enemy_dist = None
        self._step_n          = 0
        self._episode_n       = 0

    # ── process lifecycle ────────────────────────────────────────────────────

    def _launch(self):
        """Start (or restart) the Node.js game server subprocess."""
        self._kill()
        env = dict(os.environ, GAME_PATH=self.game_path)
        self._proc = subprocess.Popen(
            ["node", _GAME_SERVER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,   # unbuffered — we need real-time line responses
        )
        # Drain stderr in a background thread so it never blocks the main process
        def _drain_stderr(proc):
            try:
                for raw in proc.stderr:
                    sys.stderr.write("[node] " + raw.decode(errors='replace').rstrip() + "\n")
            except Exception:
                pass
        t = threading.Thread(target=_drain_stderr, args=(self._proc,), daemon=True)
        t.start()

        # Wait for server to be ready — use a raw ping loop that does NOT
        # trigger _rpc's auto-restart logic (which would loop infinitely during
        # the 2-3 s Node.js startup time on Windows).
        deadline = time.time() + 25.0
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("game_server.js exited immediately at startup")
            try:
                ping_bytes = (json.dumps({"type": "ping"}) + "\n").encode()
                self._proc.stdin.write(ping_bytes)
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
                if line and json.loads(line.decode()).get("type") == "pong":
                    return   # server ready
            except Exception:
                time.sleep(0.2)  # Node.js still loading — wait and retry
        raise RuntimeError("game_server.js did not respond to ping within 25 s")

    def _kill(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.kill()
                self._proc.wait(timeout=3)
            except Exception:
                pass
            self._proc = None

    def _ensure_started(self):
        if self._proc is None or self._proc.poll() is not None:
            self._launch()

    # ── low-level RPC ────────────────────────────────────────────────────────

    def _rpc(self, msg: dict, timeout: float | None = None) -> dict | None:
        """Send one JSON message, return parsed JSON response (blocking)."""
        # NOTE: We use plain blocking readline — Node responds in <1 ms, so
        # the only real risk is a crashed Node process.  We detect that via
        # empty readline (EOF) and restart.  Avoids per-call thread creation
        # which was adding ~10 ms on Windows per step.
        try:
            data = (json.dumps(msg) + "\n").encode()
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                raise EOFError("game_server stdout closed unexpectedly")
            return json.loads(line.decode())
        except Exception as e:
            sys.stderr.write(f"[NodeEnv] RPC error ({e}), restarting process\n")
            self._launch()
            return None

    # ── observation helpers ──────────────────────────────────────────────────

    def _to_obs(self, s: dict | None) -> np.ndarray:
        if s is None:
            return np.zeros(OBS_DIM, dtype=np.float32)
        obs = [
            s["px"], s["py"], s["vx"], s["vy"],
            s["health"], s["ground"],
            s["grenades"], s["dodge_ready"], s["on_fire"],
        ]
        for i in range(N_ENEMIES):
            if i < len(s["enemies"]):
                obs.extend(s["enemies"][i])
            else:
                obs.extend([0.0, 0.0, 0.0])
        obs.extend(s.get("grid") or [0.0] * (GRID_W * GRID_H))
        return np.clip(np.array(obs, dtype=np.float32), -5.0, 5.0)

    # ── gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_n += 1

        if self.restart_every and self._episode_n % self.restart_every == 0:
            print(f"  [NodeEnv] restarting Node process at episode {self._episode_n}")
            self._launch()
        else:
            self._ensure_started()

        resp = self._rpc({"type": "reset"})
        s = resp["state"] if resp else None

        self._last_kills      = 0
        self._last_lives      = s["lives"] if s else 10   # game starts with 10 lives
        self._last_level      = 1
        self._last_enemy_dist = None
        self._step_n          = 0

        return self._to_obs(s), {}

    def step(self, action):
        m, v, sh, dg, gr = (int(x) for x in action)
        resp = self._rpc({"type": "step", "action": [m, v, sh, dg, gr]})
        s    = resp["state"] if resp else None

        obs = self._to_obs(s)

        kills     = s["kills"]     if s else 0
        lives     = s["lives"]     if s else self._last_lives
        alive     = s["alive"]     if s else True
        level     = s["level"]     if s else self._last_level
        n_enemies = s["n_enemies"] if s else 0

        delta_kills = kills - self._last_kills
        delta_lives = max(0, self._last_lives - lives)
        delta_level = level - self._last_level

        # ── rewards (identical weights to env.py) ─────────────────────────────
        kill_r  = delta_kills * 10.0
        death_p = delta_lives * 5.0
        level_r = delta_level * 150.0

        ey = 0.0
        approach_r = 0.0
        if n_enemies > 0 and s and s.get("enemies"):
            ex, ey, _ = s["enemies"][0]
            dist_now = (ex*ex + ey*ey) ** 0.5
            if self._last_enemy_dist is not None and delta_kills == 0:
                d_delta    = self._last_enemy_dist - dist_now
                approach_r = max(min(d_delta * 2.0, 0.05), -0.05)
            self._last_enemy_dist = dist_now
        else:
            self._last_enemy_dist = None

        dodge_act = int(action[3])
        vert_act  = int(action[1])
        on_fire   = s.get("on_fire", 0) if s else 0
        on_ground = bool(s.get("ground", 1)) if s else True

        fire_r = 0.1 if (on_fire and dodge_act) else 0.0

        grid      = s.get("grid", []) if s else []
        on_ladder = len(grid) > 58 and grid[58] < -0.5
        enemy_above = n_enemies > 0 and ey > 0.05
        if on_ladder and vert_act != 0:
            ladder_r = 0.015 if enemy_above else 0.003
        else:
            ladder_r = 0.0

        air_dodge_r = 0.0

        wall_left  = len(grid) > 57 and grid[57] > 0.4
        wall_right = len(grid) > 59 and grid[59] > 0.4
        horiz_act  = int(action[0])
        shoot_act  = int(action[2])
        pressing_into_wall = (wall_left and horiz_act == 1) or \
                             (wall_right and horiz_act == 2)
        wall_climb_r = 0.01 if (pressing_into_wall and not on_ground
                                and not on_ladder and vert_act == 1
                                and shoot_act == 0) else 0.0

        survive = 0.002

        reward = (kill_r - death_p + level_r + approach_r
                  + fire_r + ladder_r + wall_climb_r + air_dodge_r + survive)

        self._last_kills  = kills
        self._last_lives  = lives
        self._last_level  = level
        self._step_n     += 1

        terminated = (lives <= 0 and not alive)
        # 30 000 steps (matches env.py) — at Node speeds this is much shorter
        # wall-clock time, but gives the same number of game decisions.
        truncated  = self._step_n > 30_000

        info = {"kills": kills, "lives": lives, "level": level}

        return obs, reward, terminated, truncated, info

    def close(self):
        self._kill()
