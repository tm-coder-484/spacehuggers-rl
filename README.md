# SpaceHuggers PPO RL Agent — Complete Project Handoff

> **Purpose of this document:** This is a *self-contained* engineering bible for an
> ongoing reinforcement-learning project. It is written so that a fresh AI agent (or
> human) can pick the work up **cold** — with zero prior conversation context — and
> continue productively. Read it top to bottom once; it is long on purpose. Every
> non-obvious design decision is recorded here with its *why*, because most of them
> were learned the hard way over millions of training steps.

---

## Table of Contents

1. [TL;DR](#1-tldr)
2. [Current Status & The Big Picture](#2-current-status--the-big-picture)
3. [Quickstart](#3-quickstart)
4. [The Game: SpaceHuggers (mechanics that matter for RL)](#4-the-game-spacehuggers)
5. [System Architecture](#5-system-architecture)
6. [The Two Tiers: v1 (flat MLP) vs v2 (Dict + CNN)](#6-the-two-tiers)
7. [Observation Space (every field)](#7-observation-space)
8. [Action Space](#8-action-space)
9. [Reward Shaping (every term + history)](#9-reward-shaping)
10. [The Navigation System (the crown jewel)](#10-the-navigation-system)
11. [Training Internals (PPO, network, callbacks)](#11-training-internals)
12. [The Backends & the FILES Patch System](#12-the-backends--the-files-patch-system)
13. [CPU Pinning & Performance](#13-cpu-pinning--performance)
14. [Watching & Introspection Tools](#14-watching--introspection-tools)
15. [File-by-File Reference](#15-file-by-file-reference)
16. [The Debugging Journey (decisions log)](#16-the-debugging-journey)
17. [Known Issues & Prioritized Next Steps](#17-known-issues--prioritized-next-steps)
18. [Operational Gotchas](#18-operational-gotchas)
19. [Constants & Tuning Knobs Cheat Sheet](#19-constants--tuning-knobs-cheat-sheet)
20. [Glossary](#20-glossary)

---

## 1. TL;DR

We are training a **PPO** agent (Stable-Baselines3) to play **SpaceHuggers**, a 2D
run-and-gun platformer built on the **LittleJS** engine. The agent learns from a
**headless Node.js** simulation of the game (no browser needed for training), driven
through a JSON stdin/stdout batch protocol. Observations are a **Dict** of a
multi-channel CNN "image" + a coarse global map + a flat feature vector. The reward
is shaped to encourage killing enemies, clearing levels, navigating toward enemies
(including *digging* through destructible terrain), and surviving.

The central, hard-won asset of this project is the **navigation system**: a
two-tier flow-field (fine local geodesic BFS + coarse whole-level pathfinder) that is
**dig-aware** (knows the gun can tunnel horizontally through terrain) and
**jump-aware** (knows the agent can only jump ~8 tiles vertically). This was built
incrementally to break a long-standing plateau where the agent could fight well but
could not *finish levels*.

**As of this writing the agent has just broken into level 3** after being stuck at
level 2 for 3M+ steps. The frontier is *completing* level 3+ and the remaining
known navigation gaps (see §17).

---

## 2. Current Status & The Big Picture

### Where things stand
- **~7.29M training steps** in the current (`v2`) model — `game_models_v2/ppo_sh_latest.zip`.
- **Max level reached: 3.** Record single episode: **724.7 reward / 42 kills / level 3.**
- **`ep_rew_mean` ≈ 260–266** and climbing.
- Combat is strong (25–58 kills/episode). The historical bottleneck was never
  *fighting* — it was *navigation to finish levels*.

### The arc of the project
1. **v1** — a flat 141-float observation + plain MLP policy. Got to ~17.5M steps,
   decent fighter, but a coarse observation. (Kept frozen in `game_models/`.)
2. **v2 (current)** — a richer **Dict** observation with a CNN over a local tile
   grid + a coarse global map + a vector, with a custom `SpatialExtractor` feature
   extractor. This is the active line of work.
3. **The level-completion wall.** For millions of steps the agent plateaued at
   level 1–2. It would rack up 25–35 kills but never clear a level. Investigation
   (and the user literally playing the game) revealed the real mechanics:
   - Level 1 has **~37 enemies** (not the ~15 the code constants naïvely implied).
   - Many enemies are **buried in destructible terrain** that must be shot through
     ("dug") to reach.
   - On wide levels the **last few stragglers sit far away**, beyond the local
     navigation horizon.
   - Some enemies sit **above the agent** where it cannot shoot (the gun is
     horizontal-only) and cannot reach without a real route.
4. **The navigation overhaul** (see §10) systematically fixed each of these:
   dig-aware pathfinding, a long-range coarse pathfinder, a jump-height model, plus
   reward changes (long-range approach, dig-progress, and removing a grenade-spam
   exploit). Shortly after these landed, **level 3 fell.**

### What "good" looks like next
- `max_level` reaching **4+** and, eventually, *completing* level 3 routinely.
- `ep_rew_mean` continuing to climb without collapse.
- The agent visibly **digging toward buried enemies** and **routing around
  obstacles** (observable in the watch tool, §14).

---

## 3. Quickstart

### 3.1 Environment setup
Python deps (CPU torch is intentional — this trains on CPU via the Node backend):
```
pip install -r requirements.txt
# requirements.txt:
#   torch --index-url https://download.pytorch.org/whl/cpu
#   stable-baselines3[extra]
#   gymnasium
#   playwright          (only needed for the browser-based watch tool)
#   tensorboard
```
For the **watch** tool (browser rendering) you also need Chromium:
```
playwright install chromium
playwright install-deps chromium   # Linux only
```
Node.js (>= 16, for `worker_threads`) must be on PATH for the training backend.

The game (**SpaceHuggers-main/**) is **included in this repo** (self-contained). If
for some reason it is missing, it is the public game by Frank Force:
`git clone https://github.com/KilledByAPixel/SpaceHuggers.git SpaceHuggers-main`.
**Do NOT upgrade the game blindly** — the backends patch *specific source lines*
(see §12); a newer game version may break those string-match patches.

### 3.2 The game path (`GAME_PATH`)
All entry points resolve the game directory via the `GAME_PATH` environment variable,
falling back to a hardcoded Windows path and then to `./SpaceHuggers-main` next to the
script. **In the cloud, set `GAME_PATH` to the repo's `SpaceHuggers-main` or rely on
the local fallback.** Example:
```
export GAME_PATH="$(pwd)/SpaceHuggers-main"
```

### 3.3 Train
The canonical launcher is `train.bat` (Windows). On Linux/cloud, call the Python
directly (the .bat just wraps it):
```
# Windows (presets):
train --p 3 --v2          # 12 envs, no CPU pin, Tier-2 model
train --p 1 --v2          # 6 envs, E-core pin (use while actively using the PC)

# Direct (any OS) — equivalent to --p 3 --v2:
python -u train_game.py --forever --backend node-workers --envs 12 --repeat 3 --pin none --v2
```
Key flags (see `train_game.py` argparse): `--total N | --forever`, `--envs N`,
`--backend node-workers`, `--repeat K` (action repeat), `--pin {none,p,e,all}`,
`--v2` (Tier-2 Dict obs + CNN), `--cap N` (episode truncation length),
`--ent FLOAT`, `--gamma FLOAT`, `--v` (live temp/clock monitor window, Windows).

Training **auto-resumes** from `game_models_v2/ppo_sh_latest.zip` if present (warm
start). It prints a `Resuming: ... done <N>` line confirming the step count loaded.

**Capturing logs to a file (read by other tools / agents):** use plain `cmd` with
`>` (NOT PowerShell, which writes UTF-16 and corrupts the file — see §18), and set
`PYTHONUNBUFFERED=1` so lines flush live:
```
set "PYTHONUNBUFFERED=1"
train --p 3 --v2 > train_log.txt 2>&1
```

### 3.4 Watch the agent play (with brain introspection)
```
python watch_game.py --v2                 # render the game + live neuron readout
python watch_game.py --v2 --no-probe      # render only
python watch_game.py --v2 --probe-every 1 # neuron panel every step
python watch_game.py                      # v1 model, no probe
```
While watching v2, **press `L` in the game window** to dump the exact current state
(ASCII terrain + reachability maps, geo direction, neuron activations, action probs)
to `watch_dumps.txt` — invaluable for diagnosing "why is it stuck here" cases.

### 3.5 TensorBoard
```
tensorboard --logdir game_logs_v2
```

---

## 4. The Game: SpaceHuggers

SpaceHuggers is a 2D platform shooter (Frank Force, js13k 2021, on the **LittleJS**
engine). For RL purposes, the mechanics that *matter* are below. **These mechanics
are why the navigation system is shaped the way it is — read carefully.**

### 4.1 Coordinate system
- **+Y is UP** (LittleJS world coords). Enemy "above" → positive relative y.
- Tiles are integer grid cells. `getTileCollisionData(vec2(x,y))` returns the tile
  type at a cell.

### 4.2 Tile types (collision data values)
| Const | Value | Notes |
|---|---|---|
| `tileType_ladder` | -1 | climbable, passable |
| `tileType_empty` | 0 | passable air |
| `tileType_solid` | 1 | wall |
| `tileType_dirt` | 2 | destructible fill |
| `tileType_base` | 3 | structure wall |
| `tileType_pipeH` | 4 | pipe |
| `tileType_pipeV` | 5 | pipe |
| `tileType_glass` | 6 | window (1 shot to break; passable-ish) |
| `tileType_baseBack` | 7 | background (non-colliding) |
| `tileType_window` | 8 | background |

### 4.3 The gun fires **HORIZONTALLY ONLY** (critical!)
- Bullets travel left or right, **never up or down**. Facing = sign of the last
  horizontal move input (`player.mirror`).
- Range ~8 tiles, small spread.
- **You cannot shoot the ceiling or floor.** This single fact drives the entire
  "digging is horizontal-only" and "you need a real route to reach enemies above"
  design.

### 4.4 Digging (destructible terrain)
Bullets destroy tiles probabilistically on hit — `appObjects.js` `collideWithTile`:
```js
const destroyTileChance = data == tileType_glass ? 1     // glass: 100% (1 shot)
                        : data == tileType_dirt  ? .2     // dirt:  20% (~5 shots)
                                                 : .05;   // base/pipe/solid: 5% (~20 shots)
```
So the player can **tunnel horizontally** through terrain to reach buried enemies.
Glass breaks instantly; dirt is cheap (~5 shots, and it cascades); hard walls are
expensive (~20 shots ≈ 2.5 s of sustained fire at `fireRate=8/s`). Grenades also
destroy terrain in a radius (and can breach vertically, unlike bullets).

### 4.5 Movement & jumping
- `maxCharacterSpeed = 0.2` tiles/frame (~12 tiles/s at 60fps).
- **Jump height ≈ 8 tiles** with a well-timed double-jump (air-dodge gives an upward
  boost). This was measured empirically by the human player. The agent **cannot
  float up infinitely** — a lone platform >8 tiles above with no ladder/wall is
  unreachable by jumping. The pathfinder models this (see §10.4).
- The agent can climb **ladders** (free vertical) and **walls** (`climbingWall` /
  wall-jump — significant vertical along a wall).
- Falling/descent through air is free (gravity).

### 4.6 Levels, enemies, lives
- `appLevel.js` `nextLevel()`: `playerLives += 4; levelEnemyCount = 15 + min(level*30, 300); ++level`.
  **But the *effective* enemy count is higher** than the constant suggests
  (procedural `buildBase()` placement). Empirically **level 1 ≈ 37 enemies.** Each
  subsequent level has more, on a bigger map.
- Win condition: when `enemiesCount` (live enemies) hits 0 and `levelEndTimer` isn't
  set, the timer starts; after ~3s, `nextLevel()`. So **you must kill every enemy to
  advance** — including the buried/far/elevated stragglers. This is why navigation,
  not combat, was the wall.
- Player starts with 10 lives, +4 per level. Grenades: **3 per life**, refill on
  respawn. Enemy health: Recruit 1 … DemoExpert 5.
- `totalKills` (in `appCharacters.js`) counts **enemy deaths only**.

### 4.7 Enemy behavior
- Enemies also shoot **horizontally**, dodge (i-frames), and are placed both in the
  open and **buried inside destructible bases** (you must dig to them).

---

## 5. System Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │  train_game.py  (PPO / Stable-Baselines3)      │
                 │   - SpatialExtractor (CNN+CNN+MLP)             │
                 │   - callbacks: save-best, save-best-level,     │
                 │     rolling checkpoints, entropy schedule      │
                 └───────────────┬──────────────────────────────┘
                                 │ Dict obs / actions (VecEnv API)
                 ┌───────────────▼──────────────────────────────┐
                 │  env_node_batch.py  (NodeBatchVecEnv)          │
                 │   - builds Dict obs (_to_obs_v2)               │
                 │   - computes ALL reward shaping (_reward)      │
                 │   - spawns ONE Node process, N game instances  │
                 └───────────────┬──────────────────────────────┘
                                 │ JSON over stdin/stdout (batched)
                 ┌───────────────▼──────────────────────────────┐
                 │  game_server_workers.js  (Node worker_threads) │
                 │   - loads & patches the LittleJS game          │
                 │   - steps N game instances headlessly          │
                 │   - emits rich game state (geo field, globe,   │
                 │     grid channels, enemies, bullets, ...)      │
                 └───────────────┬──────────────────────────────┘
                                 │ requires
                 ┌───────────────▼──────────────────────────────┐
                 │  SpaceHuggers-main/  (the LittleJS game)       │
                 └──────────────────────────────────────────────┘
```

**Separate "watch" path** (for human viewing + introspection):
```
watch_game.py  ──>  env.py (SpaceHuggersEnv, Playwright/Chromium)
                      - injects its OWN _getGameState JS into the page
                      - renders the real game in a browser window
                      - produces the SAME v2 Dict obs as training
```

Two things to internalize:
1. **Training uses the headless Node backend** (`game_server_workers.js`). It is
   fast (no rendering) and runs N parallel game instances via `worker_threads`.
2. **Watching uses the browser backend** (`env.py`'s injected JS). It renders the
   real game so a human can see it. **The two backends must be kept in sync** — both
   compute the geodesic field, coarse field, globe, etc. When you change navigation
   logic, change it in **both** `game_server_workers.js` and `env.py`'s `_INJECT`
   string. (There is also an older `game_server.js` single-instance node backend
   used by the v1 `env_node.py`; it lacks the v2 fields and is largely legacy.)

---

## 6. The Two Tiers

### v1 (legacy, frozen)
- Observation: flat **141-float** Box. (9 player scalars + 5 enemies×3 + 13×9 tile
  grid.)
- Policy: plain `MlpPolicy`, `net_arch=[256,256]`.
- Model dir: `game_models/`. Env: `env.py` with `obs_version=1` (browser) or
  `env_node.py` (single-instance node). Kept for reference / comparison; ~17.5M
  steps. **Not the active line.**

### v2 (current, active)
- Observation: **Dict** `{grid: (6,9,13), globe: (3,8,8), vector: (106)}`.
- Policy: `MultiInputPolicy` with custom `SpatialExtractor`, `net_arch=dict(pi=[512,768,512], vf=[512,768,512])`.
- Model dir: `game_models_v2/`. Logs: `game_logs_v2/`.
- Env: `NodeBatchVecEnv` (`env_node_batch.py`, `obs_version=2`) for training;
  `SpaceHuggersEnv` (`env.py`, `obs_version=2`) for watching.
- Enabled with the `--v2` flag.

The two tiers live in **separate model/log namespaces** so they never collide.

---

## 7. Observation Space

The v2 Dict obs is built by `env_node_batch.py::_to_obs_v2(state)` from the JSON
state the Node backend emits. Constants (top of `env_node_batch.py`):
`GRID_W, GRID_H = 13, 9`, `GRID_CH_V2 = 6`, `GLOBE_CH = 3`, `GLOBE_SZ = 8`,
`N_ENEMIES_V2 = 10`, `N_BULLETS_V2 = 5`, `VEC_DIM_V2 = 106`.

### 7.1 `grid` — (6, 9, 13) local CNN image, egocentric (±6 x, ±4 y around player)
Channels (in order), each a 9×13 plane:
0. **solid / dig-cost terrain.** `max(terrain, 0)`. The terrain channel is encoded
   with **dig cost**: glass `0.25`, dirt `0.5`, hard wall `1.0`, ladder `-1`, empty
   `0`. So this channel doubles as "how expensive is it to dig here."
1. **ladder** — 1 where `terrain < -0.5`.
2. **enemy occupancy** — per-cell enemy presence (scaled by health).
3. **hazard** — enemy bullets + explosive props + props.
4. **line-of-fire** — a horizontal ray from the player center in the facing
   direction (where the gun can hit).
5. **reachability** — the fine geodesic flow-field distance-from-player (see §10).

### 7.2 `globe` — (3, 8, 8) coarse egocentric global map (8 tiles/cell, ±32 tiles)
0. **terrain** (1 if the sampled cell is solid).
1. **enemies-now** (current enemy occupancy, coarse).
2. **enemy memory** — a decaying heatmap (`global._gmem`, decays ×0.97/step, resets
   per level) of where enemies have been seen. Lets the agent "remember" offscreen
   enemies.

### 7.3 `vector` — 106 floats (clipped to ±5)
Layout (consumed in `_to_obs_v2`):
- `[0:4]` player px, py, vx, vy
- `[4:9]` health, on_ground, grenades(0–1), dodge_ready, on_fire
- `[9:12]` level/10, enemies_remaining/100, n_enemies/10
- `[12]` facing (−1/+1)
- `[13:43]` 10 enemies × (rel_x, rel_y, health)  *(sorted nearest-first)*
- `[43:63]` 10 enemies × (vel_x, vel_y)
- `[63:73]` 10 enemies × shootable flag
- `[73:93]` 5 enemy bullets × (rel_x, rel_y, vel_x, vel_y)
- `[93]` **geo_dist** — normalized navigation distance to nearest enemy (fine
  geodesic if reachable, else coarse; 1.0 if truly unreachable)
- `[94:96]` **geo_dir** — flow-field first-step direction (sign x, sign y)
- `[96:106]` 10 enemies × "aiming at us" flag

**Important for warm-starting:** changing the *shape* of any of these breaks a warm
start. Several upgrades were deliberately kept **shape-preserving** (e.g.,
`coarse_dist` rides in the JSON state but is used only by the reward, NOT added to the
vector) precisely so the model could warm-start through the change. When you must
change shape, expect to retrain (or do a fresh run).

---

## 8. Action Space

`MultiDiscrete([3, 3, 2, 2, 2])` — 5 independent components:
| Idx | Component | Values |
|---|---|---|
| 0 | horizontal | 0 none, 1 left, 2 right |
| 1 | vertical | 0 none, 1 up/jump, 2 down (ladder) |
| 2 | shoot | 0 no, 1 yes |
| 3 | dodge | 0 no, 1 yes (roll; i-frames; puts out fire) |
| 4 | grenade | 0 no, 1 throw |

`action_repeat` (`--repeat`, default 3) holds each chosen action for K game frames,
which improves temporal credit assignment and throughput.

**Emergent tactic observed:** the agent learned to rapidly alternate left/right with
shoot held, spraying bullets in *both* horizontal directions ("face-left-shoot,
face-right-shoot"). This is an optimal answer to horizontal-only aiming for exposed
enemies — but it is *counterproductive for digging*, which needs sustained fire in
one direction. The dig-progress reward (§9) is meant to teach it to commit when
breaching a wall.

---

## 9. Reward Shaping

**All reward logic lives in `env_node_batch.py::_reward(i, state, action)`** (v2
block gated by `obs_version == 2`). The browser `env.py` has its own simpler v1-style
reward used only when watching (reward doesn't matter for watching).

### 9.1 Primary signals
- **Kill:** `+10 × Δkills` (weapon-agnostic — a grenade kill and a bullet kill both
  give +10).
- **Death:** `−5 × Δlives_lost`.
- **Level complete:** `+150 × Δlevel`.
- **Time-to-clear bonus:** on level-up, `+40 × (1 − steps_since_level/8000)`
  (faster clear → bigger bonus).
- **Health-loss penalty:** `−1 × HP lost` (dense damage-avoidance signal).

### 9.2 Navigation shaping (the important, subtle part)
- **Geodesic approach (fine):** telescoping reward on `geo_dist` (the fine
  flow-field distance to nearest enemy), `max(min(Δ·3, 0.05), −0.05)` — i.e., bounded
  ±0.05/step for getting closer/farther *through terrain*.
- **Dig-progress bonus:** if `geo_dist` drops in one step by more than walking could
  explain (`> DIG_MIN_DROP = 0.06`; a dirt breach ≈ 0.10, a hard breach ≈ 0.42), a
  path-blocking tile was just breached → reward `min(drop·DIG_SCALE, DIG_STEP_CAP)`
  (`DIG_SCALE=1.4`, `DIG_STEP_CAP=0.6`), capped at `DIG_CAP_EP=20`/episode. This pays
  for the otherwise-unrewarded ~20-shot grind of digging through a wall toward a
  buried enemy. **Non-farmable:** tied to actual geodesic progress; digging a
  pointless tunnel away from enemies yields no `geo_dist` drop → no reward.
- **Long-range approach (coarse):** when the fine field can't reach the nearest enemy
  (`geo_dist ≥ 0.999`, enemy beyond the 24-tile local horizon), reward reducing the
  **coarse** distance `coarse_dist` (telescoping, bounded ±0.05). This is
  **detour-aware** — it pays the agent to follow the macro-route the long way around
  (e.g., the "right-50/up/left-50" leg) instead of punishing the temporary
  "away-from-enemy" movement that a naïve straight-line (Euclidean) reward would.
- **Exploration:** `+0.02` per new coarse position cell visited, capped at 300 cells
  (anti-camping).

### 9.3 Reward terms that were REMOVED (and why) — do not re-add naïvely
- **Per-step "firing-position" bonus** (`+0.02` while an enemy is shootable):
  **REMOVED.** It was camp-farmable — the agent learned to sit in a safe spot racking
  reward with 0 kills (~+600/episode of pure farm).
- **Per-step "grenade button" bonus:** **REMOVED.** Rewarded pressing the button, not
  throwing.
- **Grenade-for-elevated bonus** (`+0.5` per actual throw at a non-horizontally
  aligned enemy): **REMOVED.** Even gated on a real throw it got **front-loaded** —
  the agent dumped all 3 grenades at spawn every life for a guaranteed +1.5,
  cratering the ground beneath itself (grenades dig *downward*, which bullets can't)
  and burying/killing itself. **Lesson: reward *outcomes* (kills, path progress), not
  *actions* (throwing).** Grenade use is now driven purely by the kill reward and the
  dig-progress reward (a grenade that opens a path toward an enemy is rewarded via
  the weapon-agnostic dig bonus).

### 9.4 Trackers (reset on episode end, in `reset()` and the `step_wait` done-block)
`_last_kills, _last_lives, _last_level, _last_geodist, _last_health,
_steps_since_level, _last_grenades (unused now), _dig_acc, _last_cdist, _visited`.
**Note:** a latent bug was fixed where the mid-episode done-block didn't reset the v2
trackers (`_last_geodist`, etc.), which would fire a phantom reward across the episode
boundary. Keep these resets in sync if you add a tracker.

---

## 10. The Navigation System (the crown jewel)

This is the most important and most carefully-built subsystem. It exists because
**combat was never the bottleneck — reaching every enemy to finish a level was.**
The agent's "where do I go" signal is `geo_dir` (a flow-field step direction) and its
"how far / making progress" signal is `geo_dist` / `coarse_dist`.

There are **three layers**, used in priority order:
**fine geodesic field → coarse global field → straight-line fallback.**

### 10.1 Fine geodesic field — `_bfsFrom(sx, sy, cx, cy, R)` (R=24)
A **weighted shortest-path** (Dial's bucket algorithm) over a bounded
`(2R+1)×(2R+1)` window centered on the player. It produces:
- `geo_dist` — normalized distance to the nearest enemy (`min(1, cost/GEO_NORM)`,
  `GEO_NORM=48`).
- `geo_dir` — the first-step direction toward the nearest enemy, via steepest-descent
  backtracking.
- the **reachability** grid channel (`min(1, cost/REACH_NORM)`, `REACH_NORM=16`).

**Edge cost is DIRECTIONAL and dig-aware** (`cost(fx,fy,tx,ty)`):
- empty / ladder: cost **1** (walk/climb/fall).
- a **pure vertical move into a destructible tile** (`tx === fx`): **blocked** — you
  can't dig up/down with the horizontal gun.
- glass: **1** (shoot through a window horizontally).
- dirt: **5** (~5 shots to breach, horizontally).
- base/pipe/solid: **20** (~20 shots).

These costs are *real traversal time* (walk ≈ 1 tile, dirt ≈ 5 tiles, hard wall ≈ 20
tiles of equivalent effort), grounded in `fireRate=8/s` and walk speed. **Walk cost
stays 1** (identical to the pre-dig binary BFS) so warm-starts transferred cleanly —
only buried enemies became a genuinely new signal. `MAXC=96` bounds exploration.

### 10.2 Why the field is the way it is (history)
Originally the BFS was binary (walls impassable), so buried enemies read as
*unreachable* → no `geo_dir`, no reward gradient → the agent never learned to dig.
Making destructible terrain **passable-with-cost** turned `geo_dist` into a
"dig-distance," and the telescoping geodesic reward then automatically pays for
digging toward buried enemies. The cost magnitudes were tuned to keep `geo_dist` in
the same range as before so the model could warm-start.

### 10.3 Coarse global field — `_coarseField(px, py, ex, ey)` + `_buildCoarseMap()`
The fine field only sees **24 tiles**. On wide levels the last stragglers sit 30–60+
tiles away → `geo_dist` saturates at 1.0 and the agent has *no direction*. The coarse
field fixes long-range routing:
- **Coarse map:** the whole level downsampled into **4-tile cells**. Each cell costs
  **1** if it has any open space, **4** if dirt-only (diggable), **blocked** if all
  solid rock. So **rock masses become walls the route goes around.** Built **once per
  level** (terrain is static apart from digging) and **cached** (`global._coarseMap`,
  keyed on `level`; the cache is invalidated on `resetGame` because a new level-1
  layout reuses `level==1`).
- **Coarse BFS:** each frame (only when the fine field fails), a weighted BFS over the
  ~1250-cell coarse grid → a macro direction + `coarse_dist` (`min(1, cost/120)`).
- **Integration:** when the fine field can't reach the nearest enemy, `geo_dir` uses
  the **coarse macro-direction**; the **long-range reward** telescopes on
  `coarse_dist` (detour-aware). Only if the coarse field *also* finds no path do we
  fall back to a **straight-line** sign toward the enemy.

### 10.4 Jump-height model — `_canAscendTo(x, y)` + the up-move gate
The gun can't shoot up, and the agent can't float up forever (jump ≈ **8 tiles**).
Without modeling this, the fine field promised impossible "go straight up" routes into
ceilings / at lone high platforms, and the agent got stuck mashing up. The gate, in
the fine BFS neighbor loop:
> An **up-move into open air** (`ny > y` and target is `tileType_empty`) is allowed
> **only if** `_canAscendTo` is true — i.e., a **solid/ladder launch surface lies
> within `_JUMP_REACH = 8` tiles below**, OR a **climbable wall is adjacent**.

This makes lone platforms >8 tiles above (with no ladder/wall) correctly
**unreachable**, so the agent stops wasting time and the coarse field routes it to a
real way up. Validated empirically: from ground, open-air cells up to height 7 read
ascendable, height 8+ blocked — matching the measured jump. **The coarse field is
still vertically optimistic** (see §17) — jump-height is modeled only in the fine
field.

### 10.5 Keeping the two backends in sync
`_bfsFrom`, `_canAscendTo`, `_coarseField`, `_buildCoarseMap`, the geo block, the
globe block, and the `coarse_dist`/`geo_dir` outputs exist in **both**
`game_server_workers.js` (training) and the `_INJECT` JS string in `env.py`
(watching). **Any nav change must be mirrored in both** or training and watching
diverge. The constants (`_JUMP_REACH`, `GEO_NORM`, `REACH_NORM`, `MAXC`,
`_COARSE_CELL`, coarse `NORM`/`MAXC`) appear in both.

---

## 11. Training Internals

### 11.1 The feature extractor — `SpatialExtractor` (in `train_game.py`)
```
grid  (6,9,13) → Conv2d(6,32,3,pad1) → ReLU → Conv2d(32,64,3,pad1) → ReLU
              → Flatten → Linear(→128) → ReLU                      = 128
globe (3,8,8)  → Conv2d(3,16,3,pad1) → ReLU → Conv2d(16,32,3,pad1) → ReLU
              → Flatten → Linear(→64) → ReLU                       = 64
vector (106)   → Linear(→256) → ReLU                               = 256
concat → 448-dim feature vector
```
Then `MultiInputPolicy` builds separate policy/value MLPs `[512, 768, 512]` (Tanh
activations by default), and an `action_net: Linear(512 → 12)` producing the
MultiDiscrete logits (3+3+2+2+2). The custom extractor must be importable when
loading a saved model — `watch_game.py` injects it into `__main__` for unpickling.

### 11.2 PPO config (`train_game.py`)
- `ent_coef = 0.015` (entropy bonus; there is also an `EntropyScheduleCallback` that
  can ramp it).
- `gamma = 0.997`.
- `NET_ARCH_V2 = [512, 768, 512]`.
- LR schedule `linear_schedule` (≈constant for `--forever`).
- `--cap N` sets episode truncation (the env sets `infos[i]['TimeLimit.truncated']`
  on timeout-not-death so value bootstrapping is correct).

### 11.3 Callbacks
- **SaveBestCallback** — saves `ppo_sh_best.zip` on a new best `ep_rew_mean`.
- **Save-best-by-level** — saves `ppo_sh_bestlevel.zip` on a new best mean level.
- **"NEW MAX LEVEL"** print when `max_level` increases.
- **RollingCheckpointCallback** — keeps the last ~5 `ppo_sh_<steps>_steps.zip`.
- **`ppo_sh_latest.zip`** — the warm-start point; saved periodically and on clean
  exit (Ctrl-C). After an *unclean* stop (power loss), the latest periodic save is
  the resume point — verify integrity with `zipfile.ZipFile(f).testzip()`.

### 11.4 Log line formats (for parsing reports)
```
  steps    7,123,456  |  ep_reward  266.80  (best 266.80)  |  max_level 3  |  43 sps  |  2.5 h
  ep  step   7,000,000  |  reward   724.7  |  kills  42  |  level 3
  ★ NEW MAX LEVEL: 3 ★
  ★ NEW BEST ep_rew_mean 266.8 (was 266.6) → saved ppo_sh_best.zip
```

---

## 12. The Backends & the FILES Patch System

The Node backends **load the game's JS source and patch specific lines** before
executing it in a VM/worker context. In `game_server_workers.js` there is a `FILES`
array of `[filename, [[findString, replaceString], ...]]`. Notable patches:
- `appLevel.js`: `const warmUpTime = 2` → `0` (skip the level intro delay).
- `engine/engineWebGL.js`: `const glEnable = 1` → `0` (no GL).
- `engine/engineAudio.js`: `const soundEnable = 1` → `0` (silent).
- a render-skip wrapper around `appRender` gated on `_HEADLESS_NO_RENDER`.

**Because patches are exact string matches, a different game version can silently
fail to patch** (a CRLF/whitespace mismatch once made a render-skip patch a no-op —
harmless in that case, but be aware). This is the main reason the game is pinned/
vendored into the repo rather than cloned fresh.

The game state the backend emits (`_getGameState`) is the single source of truth for
the observation. It computes: player scalars, nearest-10 enemies (+ velocity,
shootable, aiming), nearest-5 enemy bullets, the 6 local grid channels, the 3 globe
channels, the fine geodesic field (`geo_dist`/`geo_dir`/reachability), and
`coarse_dist`. `env_node_batch.py` turns that JSON into the Dict obs and computes
reward.

The browser backend (`env.py`) injects an equivalent `_getGameState` (and the nav
helpers) as a JS string (`_INJECT`) into the Playwright page, plus `_setAIInput`
(maps the action to key events) and a `keydown` listener for the **`L` dump key**.

---

## 13. CPU Pinning & Performance

Target hardware is an Intel hybrid CPU (i5-1240P: P-cores logical 0–7, E-cores 8–15),
power/thermal limited. `--pin {none,p,e,all}` pins the Python + Node processes:
- `--pin p` → P-cores (fastest, but contends with foreground use).
- `--pin e` → E-cores (slower ~40–45 sps, but keeps the machine responsive — use
  while actively on the PC).
- `--pin none` → let the OS schedule (preset 3; ~68 sps with 12 envs).
Throughput is ~40–70 steps/s depending on env count and pinning. The geodesic +
coarse BFS run inside the Node workers; they are cheap relative to the game sim.

**Headless particle reaper (~3× sps).** With rendering skipped
(`_HEADLESS_NO_RENDER`), the game's only particle-expiry path
(`Particle.render`) never runs, so particles became immortal — `engineObjects`
grew unboundedly (~4/frame) and per-frame sim cost degraded *linearly within an
episode*. `_reapParticles()` in `game_server_workers.js` (called each `_step`)
replicates just the expiry (preserving `destroyCallback`), restoring parity with
the rendered game. Headless-only (gated on `_HEADLESS_NO_RENDER`); the browser
watch path renders normally and doesn't need it.

**Further headless throughput opts.** On top of the reaper: (1) **particle
stillbirth** — `ParticleEmitter.emitParticle` still *spawns* each particle (so the
shared `rand()` stream is identical — critical for sim parity) but marks it
`destroyed` immediately, so it never runs a single `update()` (supersedes the
reaper for emitter particles; reaper kept as a catch-all). (2) **`_tcGet`** — a
direct `tileCollision[y*W+x]` read replacing `getTileCollisionData(vec2(...))` in
the BFS hot paths (`cost`, `_canAscendTo`, `_buildCoarseMap`, the jump-gate),
dropping a `Vector2` allocation per tile read. (3) **Reused typed-array BFS
buffers** — a per-`R` `Int16Array` dist grid + `Int8Array` ascend-memo, reused
across frames instead of reallocating, with the jump-reach check memoized per cell.
(4) **JSON-string IPC** — workers `JSON.stringify` the state once and the main
thread concatenates the strings, avoiding a structured-clone of the big nested
state object across the worker boundary. All preserve sim/obs behaviour exactly
(validated: shapes, all-finite obs, and BFS geo gradients unchanged). These were
the biggest throughput wins — apply them before judging sps.

---

## 14. Watching & Introspection Tools

### `watch_game.py --v2`
Renders the real game (browser) **and** prints a live "brain" panel each step:
- **INPUTS** — selected named vector features (health, geo_dist, geo_dir, nearest
  enemy, facing, grenades, …).
- **TOP-10 HIDDEN NEURONS** — the first policy hidden layer (after the first Linear),
  each neuron's pre-activation ("in") and post-activation ("out"), top-10 by |output|.
  Implemented with forward hooks (`NeuronProbe`).
- **OUTPUTS** — *all* action-head neurons (the 12 logits), grouped by the 5
  MultiDiscrete components, with logit + softmax probability, chosen action marked.

### The `L`-key stuck-state dump
Press **`L`** in the game window to append a full diagnostic snapshot to
`watch_dumps.txt`:
- header: `geo_dist`, `geo_dir` (with arrow), facing, health, grenades, n_enemies.
- nearest enemies (normalized + approx tiles).
- an auto **CASE hint** (e.g., "geo_dir points straight at the enemy vertically —
  impossible climb?").
- an **ASCII TERRAIN map** (`@`=you, `E`=enemy, `#`=wall, `%`=dirt, `"`=glass,
  `H`=ladder, `*`=hazard) with up at top.
- an **ASCII REACHABILITY map** (digits 0=here … 9=far, `.`=can't reach).
- the full brain panel at that instant.

This tool was *essential* for diagnosing the navigation gaps — e.g., it revealed the
"enemy 16 tiles up behind a ceiling, reachable only via a long detour" case that
motivated the coarse pathfinder, and the "enemies 30–60 tiles away, geo_dir = (0,0)"
case that motivated the long-range reward. **Use it.** When the agent is stuck,
press `L` and read the maps.

### Visualization scripts (offline)
- `viz_weights.py` → weight histograms/heatmaps/conv-filter images in `weight_viz/`.
- `viz_neurons.py` → interactive `neuron_viz.html` + ONNX export (`v1_policy.onnx`,
  `v2_policy.onnx`).
- `viz_web.py` → force-directed `neuron_web.html`.
- `visualize_weights.py` (older v1 variant).

---

## 15. File-by-File Reference

| Path | Role |
|---|---|
| `train_game.py` | PPO training entry point; `SpatialExtractor`; callbacks; argparse. |
| `env_node_batch.py` | **v2** VecEnv (`NodeBatchVecEnv`); builds Dict obs (`_to_obs_v2`); **all reward shaping** (`_reward`); spawns the Node batch backend. |
| `game_server_workers.js` | **v2** headless backend; worker_threads; loads+patches the game; emits rich state incl. fine geo field, coarse field, globe, grid channels. |
| `env.py` | Browser env (`SpaceHuggersEnv`, Playwright). Supports v1 (flat) and v2 (Dict) obs. Injects `_INJECT` JS (its own `_getGameState` + nav helpers + `L`-dump). Used for **watching**. |
| `watch_game.py` | Watch the agent; `NeuronProbe` brain panel; `L`-key dump handling; v1/v2. |
| `env_node.py` | **v1** single-instance Node env (legacy). |
| `game_server.js` | **v1** single-instance Node backend (legacy; lacks v2 nav fields). |
| `train.bat` | Windows launcher with presets `--p 1..4` and pass-through flags. Uses `python -u`. |
| `bench_node.py`, `check_fps.py` | Throughput/FPS benchmarking. |
| `viz_weights.py`, `viz_neurons.py`, `viz_web.py`, `visualize_weights.py` | Weight/neuron visualizers. |
| `temp_monitor.ps1` | Windows temp/clock monitor window (the `--v` flag). |
| `setup.sh` | One-shot environment setup (deps, Chromium, clone game, mkdir). |
| `requirements.txt` | Python deps. |
| `docs/tier2-design.md` | Earlier design note for the Tier-2 obs. |
| `SpaceHuggers-main/` | **The game** (vendored into the repo; LittleJS). |
| `game_models_v2/` | **Active** v2 checkpoints: `ppo_sh_latest`, `ppo_sh_best`, `ppo_sh_bestlevel`, rolling `ppo_sh_<steps>_steps`. |
| `game_models/` | v1 checkpoints (frozen). |
| `game_logs_v2/`, `game_logs/` | TensorBoard logs. |
| `game_models_v2_farm_*`, `*_pre_perception_*`, `*_old_*`, `*_backup_*`, `back1234/` | **Archived experiments** (failed reward-farm runs, pre-navigation-overhaul snapshots, backups). ~390M total. **Not needed to continue the work** — safe to delete to slim the repo. |
| `watch_dumps.txt` | Output of the `L`-key stuck-state dumps. |
| `train_log.txt` | Redirected training console log (see §18 for the encoding gotcha). |

---

## 16. The Debugging Journey (decisions log)

A compressed history so you don't re-learn these the hard way:

1. **Reward farming (1288 reward, 0 kills).** A per-step firing-position bonus and a
   grenade-button bonus let the agent camp-farm. **Removed both.** Lesson: never
   reward a *per-step state* the agent can sit in.
2. **The level-2 wall.** 3M+ steps stuck. Combat was fine (25–35 kills); the agent
   just couldn't *finish* levels. Root causes uncovered (partly by the human playing
   the game): ~37 enemies/level, many **buried** in destructible terrain, others
   **far** or **above**.
3. **Dig-aware nav.** Made the geodesic BFS treat destructible terrain as
   passable-with-cost (horizontal-only). Buried enemies became reachable; the
   geodesic reward started paying for digging. Costs tuned to preserve warm-start.
4. **Dig-progress reward.** The ~20-shot grind to breach a hard wall was unrewarded
   until the (sparse) +150 clear, so the agent avoided it. Added a bounded,
   non-farmable per-breach bonus (detected via large single-step `geo_dist` drops).
5. **Far-enemy problem.** On wide levels the last enemies sat beyond the 24-tile
   horizon → `geo_dir=(0,0)`, no pull. First fix: a straight-line (Euclidean)
   long-range approach reward + fallback `geo_dir`. This **broke the level-1 wall**
   (the agent started clearing the scattered stragglers).
6. **Grenade-spam self-burial.** The grenade-for-elevated reward got front-loaded;
   the agent nuked the ground at spawn and buried itself. **Removed it** (outcomes,
   not actions). Reward jumped to a new peak right after.
7. **The long-detour / vertical-block case.** The `L`-dump revealed an enemy 16 tiles
   up behind a ceiling, reachable only via a ~100-tile detour — and the *Euclidean*
   fallback was actively pointing the agent *into the ceiling*. Built the **coarse
   global pathfinder** (routes around rock masses) and switched the long-range reward
   from Euclidean to **coarse-distance** (detour-aware).
8. **Jump-height blind spot.** The field assumed infinite vertical air movement.
   Added the **jump-height model** (`_canAscendTo`, `_JUMP_REACH=8`), calibrated to
   the human-measured ~8-tile double-jump.
9. **Level 3 fell** shortly after 6–8 landed, from a warm start at ~7.19M.

Recurring meta-lessons:
- **Reward outcomes, not actions.** Every farm/exploit came from rewarding a
  controllable action or sittable state.
- **Keep changes shape-preserving** to warm-start through them.
- **Mirror nav changes in both backends.**
- **Use the `L`-dump** to see the *actual* stuck geometry before theorizing.

---

## 17. Known Issues & Prioritized Next Steps

1. **Coarse field is vertically optimistic (thin ceilings).** The coarse map marks a
   4-tile cell "open" if *any* tile is passable, so a **thin (1-tile) horizontal
   ceiling** inside an otherwise-open cell isn't seen as a barrier — the coarse route
   may still try to go up through it. Thick rock masses *are* handled. Fix idea:
   encode per-cell **vertical passability** (or jump-height) into the coarse map.
2. **Target the nearest *reachable* enemy, not the nearest absolute.** If the
   nearest enemy is genuinely unreachable (a lone platform with no route), the agent
   fixates on it (geo_dir/coarse/straight-line all point at it) and ignores reachable
   enemies. Fix idea: among enemies, pick the nearest one whose coarse cell is
   reachable; target that. This pairs naturally with the jump-height model.
3. **Completing level 2/3 (endurance).** Reaching level 3 ≠ clearing it. Bigger
   levels = more enemies + more lives needed. Watch whether survival/health
   management improves; consider whether the death penalty / health penalty balance
   is right for long episodes.
4. **The face-left/right spray vs digging tension.** The agent's bidirectional spray
   is great for exposed enemies but bad for digging (halves shots-per-direction).
   Watch whether the dig-progress reward teaches it to commit fire at walls; if not,
   consider a small extra signal for sustained directional fire *into a path wall*.
5. **Coarse map staleness from digging.** The coarse map is cached per level and not
   rebuilt when the agent digs new openings. Conservative (won't route through real
   walls) but may miss dug shortcuts. Probably fine; revisit only if it matters.

When you investigate any of these, **first reproduce it in `watch_game.py --v2` and
press `L`** to capture the real geometry.

---

## 18. Operational Gotchas

- **Log encoding (important).** Redirect training output with **plain `cmd`** and
  `>` — NOT PowerShell. Windows PowerShell 5.1's `>` / `Tee-Object` write **UTF-16 LE
  (BOM)**, which renders as garbage CJK characters and corrupts the file (and trips
  text filters). Use `set "PYTHONUNBUFFERED=1"` so lines flush live. To watch a
  cmd-redirected file live, use a *second* window:
  `powershell -Command "Get-Content train_log.txt -Wait -Tail 30"`. **Never open the
  log in an editor (Notepad++) and "convert encoding" while it's being written** —
  that bakes corruption and fights the writer.
- **Power loss / unclean stop.** Training does NOT auto-restart on boot. After a hard
  stop, just relaunch `train ... --v2`; it warm-starts from `ppo_sh_latest.zip`.
  **Verify checkpoint integrity first** if you're worried:
  `python -c "import zipfile,glob; [print(f, zipfile.ZipFile(f).testzip()) for f in glob.glob('game_models_v2/*.zip')]"`
  (`None` = OK). All checkpoints survived a recent power loss intact.
- **Warm-start vs fresh.** Reward/cost magnitude changes were deliberately tuned to
  preserve warm-start. After a change, expect a brief dip-then-climb as the policy
  adapts; judge by `max_level`/kills, not raw reward (the reward scale may shift).
- **Mirror nav edits** in `game_server_workers.js` AND `env.py`'s `_INJECT`.
- **Don't blanket-kill `node`/`python`** while diagnosing — Claude/MCP and the game
  workers are all node/python; filter by command line (`*train_game.py*`,
  `*game_server_workers*`).

---

## 19. Constants & Tuning Knobs Cheat Sheet

**`env_node_batch.py`:**
- `GRID_W,GRID_H=13,9`, `GRID_CH_V2=6`, `GLOBE_CH=3`, `GLOBE_SZ=8`,
  `N_ENEMIES_V2=10`, `N_BULLETS_V2=5`, `VEC_DIM_V2=106`.
- Dig reward: `DIG_MIN_DROP=0.06`, `DIG_SCALE=1.4`, `DIG_STEP_CAP=0.6`, `DIG_CAP_EP=20`.
- Reward magnitudes: kill `+10`, death `−5`, level `+150`, time-to-clear `+40·(1−s/8000)`,
  health `−1/HP`, geodesic `±0.05`, coarse approach `±0.05`, exploration `+0.02`
  (cap 300 cells).

**`game_server_workers.js` & `env.py` `_INJECT` (keep in sync):**
- Fine field: `R=24`, `GEO_NORM=48`, `REACH_NORM=16`, `MAXC=96`.
  Edge costs: walk/glass `1`, dirt `5`, hard `20`, pure-vertical-dig `blocked`.
- Jump model: `_JUMP_REACH=8`.
- Coarse field: `_COARSE_CELL=4`, coarse `NORM=120`, coarse `MAXC=240`.
  Cell cost: open `1`, dirt-only `4`, all-solid `blocked`.

**`train_game.py`:**
- `NET_ARCH_V2=[512,768,512]`, `ENT_COEF=0.015`, `GAMMA=0.997`,
  extractor outputs `cnn=128, globe=64, vec=256` → `448`.
- CLI: `--total/--forever --envs N --backend node-workers --repeat K --pin {none,p,e,all} --v2 --cap N --ent F --gamma F`.

**Game facts:** +Y up; gun horizontal-only; jump ≈ 8 tiles; level 1 ≈ 37 enemies;
3 grenades/life; dig chances glass 100% / dirt 20% / hard 5%; `fireRate=8/s`;
`maxCharacterSpeed=0.2`.

---

## 20. Glossary

- **geo_dist / geo_dir** — fine geodesic flow-field distance/direction to the nearest
  enemy (dig-aware, jump-aware). The agent's primary navigation signal.
- **coarse_dist** — whole-level coarse pathfinder distance (detour-aware); used for
  the long-range reward and `geo_dir` when the fine field can't reach.
- **dig-aware** — the pathfinder treats destructible terrain as passable-with-cost,
  horizontally only (mirroring the horizontal-only gun).
- **jump-aware** — the fine field forbids ascending open air more than ~8 tiles above
  a launch surface (no infinite float).
- **globe** — the coarse 8×8 egocentric global map obs channels (terrain, enemies-now,
  decaying memory).
- **the `L`-dump** — press L while watching to capture the exact stuck geometry.
- **warm-start** — resuming PPO from `ppo_sh_latest.zip`; most changes were kept
  shape-preserving to allow it.
- **farm / exploit** — a reward the agent can accumulate without doing the intended
  task (e.g., camping in a firing position). Several were found and removed; avoid
  rewarding actions/sittable states.
- **v1 / v2 / Tier-2** — the flat-MLP (legacy) vs Dict+CNN (active) observation/model
  generations; v2 == Tier-2 == `--v2`.

---

### Final note to the next agent

The hard part of this project is **navigation reward/perception**, not RL
hyperparameters. When the agent "can't do X," the question is almost always *"does the
flow-field/reward make X reachable and rewarded?"* — go look with `watch_game.py
--v2` + `L`, read the terrain/reachability maps, and fix the *signal*, not the
optimizer. Keep changes shape-preserving, mirror them across both backends, and
reward outcomes rather than actions. Good luck — the agent just reached level 3; help
it go further.
