# SpaceHuggers RL — Tier 2 Design: Perception & Capacity Overhaul

*Draft — 2026-06-07. Status: awaiting decisions (see "Open Decisions").*

## 1. Goal

Break the **level-3 ceiling**. Tier-1 (gamma↑, entropy↑, survival-reward removed,
episode cap) tunes the *existing* agent but can't fix what it fundamentally
**cannot perceive**. Tier-2 rebuilds the observation and network so the agent can
actually reason about the situation it's in.

## 2. Root cause (why it's stuck)

| Limitation | Today | Consequence |
|---|---|---|
| Enemy awareness | nearest **5** of up to **75+** | fights blind in a swarm |
| Spatial sight | **13×9** local grid (±6×±4 tiles) of a 400-wide level | no map, no flanking awareness |
| Temporal info | **single frame** (one-frame velocity only) | can't track enemy/projectile motion |
| Level progress | **not in observation** | doesn't know it's one kill from winning |
| Weapon/ammo | **not in observation** | can't manage resources |
| Network | MLP `[256,256]` on a *flattened* grid | relearns 2D structure from scratch |

**Key insight:** raw network width/depth (your suggestion) only helps if the
*inputs carry the information*. More neurons on top of 5-enemy tunnel vision just
memorizes noise. So Tier-2 leads with **observation**, then adds capacity.

## 3. Non-goals / constraints

- **Speed is not a goal.** We're at the 28W CPU power wall; Tier-2 will be
  *slower per step* and we accept that for better learning.
- **No weight transfer.** Changing obs shape + architecture means a **fresh run
  from scratch**. The current model (and your 10.5M backup) are incompatible.
- **Do not clobber v1 models.** Tier-2 saves to a **new namespace**
  (`game_models_v2/`, `game_logs_v2/`) so v1 stays intact.
- Backend stays **node-workers** (`game_server_workers.js` + `env_node_batch.py`).

## 4. Proposed design

### 4.1 Observation → `Dict` space

**`vector` stream (per frame, ~76 floats):**
- player (9): px, py, vx, vy, health, ground, grenades, dodge_ready, on_fire
- level progress (4): level (norm), enemies_remaining (norm), fraction_cleared, warmup
- weapon/ammo (3): weapon_type, ammo (norm), reloading  *(if exposed by game)*
- nearest **10** enemies × 6 (60): dx, dy, health, vx, vy, aiming_at_player

**`grid` stream (multi-channel 2D "image", 13×9 — optionally 15×11):**
- ch0 terrain (walls / ladders / glass) — existing
- ch1 enemy occupancy in the local window
- ch2 hazards (grenades / explosions / projectiles)

**`globe` stream (optional, 8×8 = 64):** coarse enemy-density map over the *whole*
level, so the agent perceives the swarm beyond its local window. Solves "5 of 75"
better than just lengthening the enemy list.

**Frame stacking `k=3`** (`VecFrameStack`): gives true velocity/trajectory of
everything for cheap. Vector → ×3; grid → channels ×3.

### 4.2 Network → `MultiInputPolicy` + custom extractor

```
grid  (3*k ch × 13×9) → Conv(32,3x3)→ReLU → Conv(64,3x3)→ReLU → flatten → Linear(128)
vector (76*k)         → Linear(256)→ReLU
globe (64*k)          → Linear(64)→ReLU            [if used]
                         concat (≈448) → shared [512, 512] → π head + V head
```

- **Wider/deeper (your ask):** shared `[512,512]` (up from `[256,256]`), plus the
  CNN adds genuine spatial depth. Option to push to `[768,768]` or add a 3rd
  shared layer — see the compute trade-off below.
- CNN on a 13×9 grid is *cheap* (tiny image), so most added cost is the wider MLP.

### 4.3 Reward

Carry over Tier-1 (kills +10, level +150, death −5, gamma 0.997, ent 0.015, no
survival, tunable cap). Now that `enemies_remaining` is observable, add a small
**level-progress shaping** term so closing out a level has a dense gradient, not
just the sparse +150.

### 4.4 Normalization

Add **`VecNormalize`** (observation normalization; clip rewards optional). The new
heterogeneous features (counts, positions, velocities) live on very different
scales — normalization is near-mandatory for the MLP to train well. Must
save/load the running stats alongside the model.

## 5. Compute reality check

Current: `256×256` MLP, ~78 sps (6 env) / ~120 sps (12 env). Game-sim cost is
unchanged (same frames), so the slowdown is only in the **policy forward + PPO
update**:
- 4-frame stack + `512×512` + small CNN → estimate **~60-80% of current sps**
  (heavier periodic updates; per-step forward still sub-ms on CPU).
- Going to `[768,768]`/3 layers → steeper drop. **Recommend starting at `512×512`.**

Because it's a **fresh run on CPU**, reaching v1-level competence will take real
wall-clock (days). Worth it only if committing to a long run.

## 6. Files to change (build sequence)

1. **`game_server_workers.js` / `game_server.js`** — `_getGameState()` emits the
   richer state: nearest 10 enemies (+velocity, +aiming), `levelEnemyCount`,
   weapon/ammo, enemy-occupancy + hazard channels, optional 8×8 density map.
2. **`env_node_batch.py`** (+`env_node.py`, `env.py` for parity) — `_to_obs`
   builds the `Dict` obs; `observation_space` → `gym.spaces.Dict`; level-progress
   reward term.
3. **`train_game.py`** — custom `BaseFeaturesExtractor` (CNN+MLP), `MultiInputPolicy`,
   `VecFrameStack`, `VecNormalize`; v2 model/log dirs; net_arch config.
4. **Smoke test** — verify obs shapes, one rollout, one update, no NaNs.
5. **Launch fresh v2 run** (new namespace, doesn't touch v1).

## 7. Open Decisions (need your call before build)

- **A. Temporal:** Frame stacking *(recommended — cheap, CPU-friendly)* vs
  `RecurrentPPO` LSTM *(more powerful, notably slower on CPU, more complex)*.
- **B. Spatial:** `Dict` + CNN-on-grid *(recommended — real spatial reasoning)*
  vs a simpler **single bigger flat MLP** *(your instinct — less code, but a dense
  net is poor at 2D structure)*. Could do the flat-MLP first as a cheaper test.
- **C. Net size:** `512×512` *(recommended start)* vs larger `768×768`/3-layer
  *(more capacity, lower sps)*.
- **D. Global awareness:** include the 8×8 density map *(recommended)* vs just the
  nearest-10 enemy list.
- **E. Scope:** full overhaul in one go, vs an incremental first step (e.g. just
  **frame-stack + more enemies + bigger flat MLP**, no CNN/Dict) to test the
  hypothesis cheaply before the full rebuild.

## 8. Recommendation

Two viable paths:
- **Minimal test (fast to build, ~½ the gain):** frame-stack(3) + nearest-10
  enemies + level-progress in a flat vector + bigger flat MLP `[512,512]`. No
  Dict/CNN. Confirms whether perception is really the wall before committing to
  the full rebuild.
- **Full Tier-2 (recommended if you're serious about pushing past level 5):** the
  Dict + CNN + density-map + frame-stack design above.

I'd start with the **minimal test** — if frame-stacking + more enemies alone lifts
it past level 3, you've validated the diagnosis cheaply and can then add the CNN
for the next ceiling. If it doesn't move, the full spatial overhaul is justified.
