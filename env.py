"""
SpaceHuggers gymnasium environment — wraps the game via Playwright.

Reward design philosophy (v3 — clean, no penalty traps):
  Primary signals  : kills (+10), level completion (+150), death (-5)
  Dense shaping    : approach reward ±0.05/step to guide toward enemies
  Mechanic bonuses : fire-dodge (+0.1), ladder use (+0.003/0.015), wall-climb (+0.01)
  Survival         : +0.002/step

  NO action penalties — exploration must remain free.  Penalising actions
  (shoot, grenade, position) was causing entropy collapse: the agent learned
  "doing nothing is safest" and stopped exploring entirely.

Observations (141 floats):
  player_x, player_y, vx, vy, health, on_ground,
  grenade_count (0-3), dodge_ready (0/1), on_fire (0/1)  (9)
  nearest 5 enemies: rel_x, rel_y, health               (15)
  13x9 tile grid centered on player                     (117)
    -1.0 = ladder   0.0 = empty    0.3 = inert prop
     0.5 = glass   0.75 = explosive  1.0 = solid

Actions  MultiDiscrete([3, 3, 2, 2, 2]):
  [0] horizontal : 0=none  1=left        2=right
  [1] vertical   : 0=none  1=up/jump     2=down (ladder)
  [2] shoot      : 0=no    1=yes
  [3] dodge      : 0=no    1=yes   (X — recharges in ~2 s, puts out fire)
  [4] grenade    : 0=no    1=throw (C — refills to 3 on each respawn)

SpaceHuggers quick-ref:
  Enemy health : Recruit=1, Soldier=2, Captain=3, Specialist=4, DemoExpert=5
  Player lives : start 10, +3 per level completed
  Gun range    : ~10-15 tiles (get in close!)
  Roll in air  : gives upward speed boost (= double jump)
  Roll melee   : rolling into an enemy does damage + brief invulnerability
"""

import time
import numpy as np
import gymnasium as gym
from pathlib import Path
from playwright.sync_api import sync_playwright

_INJECT = """
(function() {
    // Prevent engine from wiping inputData when window loses focus
    document.hasFocus = () => true;

    window._setAIInput = function(left, right, up, down, shoot, dodge, grenade) {
        const set = (key, val) => {
            const wasDown = inputData[0][key]?.d || 0;
            inputData[0][key] = {
                d: val ? 1 : 0,
                p: val && !wasDown ? 1 : 0,
                r: !val && wasDown ? 1 : 0,
            };
        };
        set(37, left);      // left arrow
        set(39, right);     // right arrow
        set(38, up);        // up arrow  — jump + climb up ladders
        set(40, down);      // down arrow — climb down ladders
        set(90, shoot);     // Z = shoot
        set(88, dodge);     // X = dodge roll
        set(67, grenade);   // C = throw grenade
        isUsingGamepad = 0;
    };

    window._getGameState = function() {
        try {
            const p = players && players[0];
            if (!p) return null;

            const lx = levelSize ? levelSize.x : 200;
            const ly = levelSize ? levelSize.y : 100;

            // ── enemies ──────────────────────────────────────────────────────
            const enemies = [];
            for (const o of engineCollideObjects) {
                if (o.isCharacter && o.team === team_enemy && !o.isDead()) {
                    const dx = o.pos.x - p.pos.x;
                    const dy = o.pos.y - p.pos.y;
                    enemies.push({ x: dx/lx, y: dy/ly,
                                   health: o.health/5, dist: dx*dx+dy*dy });
                }
            }
            enemies.sort((a, b) => a.dist - b.dist);

            // ── 13x9 tile grid centred on player ─────────────────────────────
            const GW = 13, GH = 9;
            const HW = (GW-1)/2, HH = (GH-1)/2;
            const px = Math.round(p.pos.x);
            const py = Math.round(p.pos.y);
            const grid = new Array(GW * GH).fill(0);

            for (let dy = -HH; dy <= HH; dy++) {
                for (let dx = -HW; dx <= HW; dx++) {
                    const tile = getTileCollisionData(vec2(px+dx, py+dy));
                    let val = 0;
                    if      (tile === tileType_ladder) val = -1.0;
                    else if (tile === tileType_glass)  val =  0.5;
                    else if (tile > 0)                 val =  1.0;
                    grid[(dy+HH)*GW + (dx+HW)] = val;
                }
            }

            // Overlay props — explosive ones are tactically important
            for (const o of engineCollideObjects) {
                if (o.isGameObject && !o.isCharacter &&
                    !o.isWeapon && !o.isCheckpoint && !o.destroyed) {
                    const dx = Math.round(o.pos.x - px);
                    const dy = Math.round(o.pos.y - py);
                    if (Math.abs(dx) <= HW && Math.abs(dy) <= HH) {
                        grid[(dy+HH)*GW + (dx+HW)] =
                            o.explosionSize > 0 ? 0.75 : 0.3;
                    }
                }
            }

            const dodgeReady = !p.dodgeRechargeTimer.active() ? 1.0 : 0.0;
            const onFire     = p.burnTimer.active() ? 1.0 : 0.0;

            return {
                px:          p.pos.x / lx,
                py:          p.pos.y / ly,
                vx:          p.velocity.x / 0.2,
                vy:          p.velocity.y / 0.3,
                health:      p.health,
                ground:      p.groundTimer.active() ? 1.0 : 0.0,
                grenades:    (p.grenadeCount || 0) / 3.0,
                dodge_ready: dodgeReady,
                on_fire:     onFire,
                alive:       !p.isDead(),
                lives:       playerLives,
                kills:       totalKills,
                level:       level,
                warmup:      levelWarmup ? 1 : 0,
                enemies:     enemies.slice(0, 5).map(e => [e.x, e.y, e.health]),
                n_enemies:   enemies.length,
                grid:        grid,
            };
        } catch (e) {
            return null;
        }
    };

    window._resetGame = () => resetGame();

    console.log('[AI] helpers injected');
})();
"""

N_ENEMIES  = 5
GRID_W, GRID_H = 13, 9
OBS_DIM    = 9 + N_ENEMIES * 3 + GRID_W * GRID_H   # 9 + 15 + 117 = 141


class SpaceHuggersEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 60}

    def __init__(self,
                 game_path: str,
                 headless: bool = True,
                 frame_ms: int = 50,
                 action_repeat: int = 1,
                 restart_every: int = 200):
        """
        game_path      : path to SpaceHuggers-main/
        headless       : run browser without a window
        frame_ms       : ms to hold each action (50 ms ≈ 3 game frames at 60fps)
        action_repeat  : repeat each action this many sub-frames (1 = no repeat)
        restart_every  : restart browser every N episodes (prevents memory leaks)
        """
        super().__init__()
        self.game_path      = Path(game_path).resolve()
        self.headless       = headless
        self.frame_ms       = frame_ms
        self.restart_every  = restart_every

        # [horizontal, vertical, shoot, dodge, grenade]
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 2, 2, 2])
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self.action_repeat  = action_repeat
        self._pw = self._browser = self._page = None

        # Episode state
        self._last_kills      = 0
        self._last_lives      = 10   # game starts with 10 lives
        self._last_level      = 1
        self._last_enemy_dist = None  # for approach reward
        self._step_n          = 0
        self._episode_n       = 0

    # ── browser lifecycle ────────────────────────────────────────────────────

    def _launch(self):
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
        if self._pw:
            try:
                self._pw.stop()
            except Exception:
                pass

        self._pw      = sync_playwright().start()
        args = ["--mute-audio"]
        if self.headless:
            args += ["--use-gl=swiftshader"]
        self._browser = self._pw.chromium.launch(headless=self.headless, args=args)
        self._page    = self._browser.new_page(viewport={"width": 800, "height": 600})
        self._page.goto(self.game_path.as_uri() + "/index.html")
        time.sleep(2.5)
        self._page.evaluate(_INJECT)
        time.sleep(0.2)

    def _ensure_started(self):
        if self._page is None:
            self._launch()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _raw(self):
        try:
            return self._page.evaluate("window._getGameState()")
        except Exception:
            return None

    def _to_obs(self, s) -> np.ndarray:
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

    def _send(self, action):
        m, v, sh, dg, gr = (int(x) for x in action)
        left  = 1 if m == 1 else 0
        right = 1 if m == 2 else 0
        up    = 1 if v == 1 else 0
        down  = 1 if v == 2 else 0
        self._page.evaluate(
            f"window._setAIInput({left},{right},{up},{down},{sh},{dg},{gr})"
        )

    # ── gym API ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_n += 1

        if self.restart_every and self._episode_n % self.restart_every == 0:
            print(f"  [env] restarting browser at episode {self._episode_n}")
            self._launch()
        else:
            self._ensure_started()

        self._page.evaluate("window._resetGame()")
        time.sleep(2.8)   # level warmup is 2 s; extra buffer for safety

        self._last_kills      = 0
        self._last_lives      = 10
        self._last_level      = 1
        self._last_enemy_dist = None
        self._step_n          = 0

        return self._to_obs(self._raw()), {}

    def step(self, action):
        # ── Action repeat ─────────────────────────────────────────────────────
        # Hold the same action for `action_repeat` sub-frames (default 4 × 50 ms
        # = 200 ms per decision).  Benefits:
        #   • Actions look deliberate instead of randomly flickering every 50 ms
        #   • Longer commitment window improves temporal credit assignment
        #   • Same 25-min wall-clock budget (7 500 steps × 200 ms)
        total_reward = 0.0
        obs          = np.zeros(OBS_DIM, dtype=np.float32)
        info         = {}
        terminated   = False
        truncated    = False

        for _ in range(self.action_repeat):
            self._send(action)
            time.sleep(self.frame_ms / 1000.0)

            s   = self._raw()
            obs = self._to_obs(s)

            kills     = s["kills"]     if s else 0
            lives     = s["lives"]     if s else self._last_lives
            alive     = s["alive"]     if s else True
            level     = s["level"]     if s else self._last_level
            n_enemies = s["n_enemies"] if s else 0

            delta_kills = kills - self._last_kills
            delta_lives = max(0, self._last_lives - lives)   # lives only count down
            delta_level = level - self._last_level

            # ── Primary rewards ───────────────────────────────────────────────
            kill_r  = delta_kills * 10.0    # +10 per enemy killed
            death_p = delta_lives * 5.0     # -5 per life lost
            level_r = delta_level * 150.0   # +150 per level completed

            # ── Dense navigation: approach reward ─────────────────────────────
            # Clipped to ±0.05/step so it shapes without overwhelming kill reward.
            # Skipped on kill steps (nearest enemy changes → dist delta invalid).
            ey = 0.0   # nearest enemy Y — initialised here so ladder block reads it
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

            # ── Mechanic bonuses ──────────────────────────────────────────────
            dodge_act = int(action[3])
            vert_act  = int(action[1])
            on_fire   = s.get("on_fire", 0) if s else 0
            on_ground = bool(s.get("ground", 1)) if s else True

            fire_r = 0.1 if (on_fire and dodge_act) else 0.0

            grid        = s.get("grid", []) if s else []
            on_ladder   = len(grid) > 58 and grid[58] < -0.5
            enemy_above = n_enemies > 0 and ey > 0.05
            if on_ladder and vert_act != 0:
                ladder_r = 0.015 if enemy_above else 0.003
            else:
                ladder_r = 0.0

            # air_dodge_r was causing jump+dodge spam worth ~+90/episode — removed.
            air_dodge_r = 0.0

            wall_left   = len(grid) > 57 and grid[57] > 0.4
            wall_right  = len(grid) > 59 and grid[59] > 0.4
            horiz_act   = int(action[0])
            shoot_act   = int(action[2])
            pressing_into_wall = (wall_left  and horiz_act == 1) or \
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
            total_reward     += reward

            terminated = (lives <= 0 and not alive)
            # 30 000 steps × 50 ms = 25 min max — enough to complete several levels
            truncated  = self._step_n > 30_000

            info = {"kills": kills, "lives": lives, "level": level}

            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info

    def close(self):
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._pw:
            try:
                self._pw.stop()
            except Exception:
                pass
            self._pw = None
        self._page = None
