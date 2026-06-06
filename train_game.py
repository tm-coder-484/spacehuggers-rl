"""
Train PPO on SpaceHuggers.  Fully resumable across any number of sessions.

Usage:
    python train_game.py                      # train / resume (4M step target)
    python train_game.py --forever            # run until Ctrl+C, no step limit
    python train_game.py --total 10000000     # custom target
    python train_game.py --envs 6             # parallel envs (default 6)
    python train_game.py --headless false     # watch one env (playwright only)
    python train_game.py --backend playwright # use Chromium browser backend

Backends:
    node       (default) — Node.js headless, ~20-100x faster, ~80 MB/env
    playwright           — Chromium browser,  ~50 ms/step,   ~500 MB/env

Speed estimates (i5-1240p, 6 envs, node backend):
    ~500+ sps  →  same 4-day run ≈ 200M+ steps

Saves every 5000 steps.  Keeps last 5 checkpoints.
Ctrl+C saves immediately — fully resumable.

TensorBoard:  tensorboard --logdir game_logs
"""

import argparse
import glob
import json
import logging
import os
import time
import warnings

# Suppress Windows asyncio / Playwright cleanup noise on Ctrl+C
warnings.filterwarnings("ignore", category=ResourceWarning)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import SpaceHuggersEnv
from env_node import NodeEnv

# ── config ───────────────────────────────────────────────────────────────────
GAME_PATH  = os.environ.get(
    "GAME_PATH",
    r"D:\tmaco0\Onedrive - Department of Education\Documents\Downloads\sd card\rk-games\games\SpaceHuggers-main"
    if os.name == "nt" else
    os.path.join(os.path.dirname(__file__), "SpaceHuggers-main")
)
MODEL_DIR  = "game_models"
LOG_DIR    = "game_logs"
META_FILE  = os.path.join(MODEL_DIR, "meta.json")
KEEP_CKPTS = 5
SAVE_FREQ  = 5_000      # steps between saves — less churn than 1500
FOREVER    = 10**12     # effectively infinite step target


# ── rolling checkpoint cleanup ────────────────────────────────────────────────
class RollingCheckpointCallback(CheckpointCallback):
    def __init__(self, keep: int = KEEP_CKPTS, **kwargs):
        super().__init__(**kwargs)
        self._keep = keep

    def _on_step(self) -> bool:
        result = super()._on_step()
        pattern = os.path.join(self.save_path, f"{self.name_prefix}_*.zip")
        files   = sorted(glob.glob(pattern))
        for old in files[: -self._keep]:
            try:
                os.remove(old)
            except OSError:
                pass
        return result


# ── progress callback ─────────────────────────────────────────────────────────
class ProgressCallback(BaseCallback):
    def __init__(self, print_every: int = 5_000, steps_already: int = 0,
                 total_target: int = FOREVER, max_level_seen: int = 1):
        super().__init__()
        self._print_every   = print_every
        self._steps_already = steps_already
        self._total_target  = total_target
        self._last_print    = 0
        self._start         = time.time()
        self._best_reward   = float("-inf")
        self._max_level     = max_level_seen
        self._initial_ts    = None   # num_timesteps at start of this learn() call

    def _on_step(self) -> bool:
        if self._initial_ts is None:
            self._initial_ts = self.num_timesteps

        n = self.num_timesteps - self._initial_ts   # steps this session only

        total_so_far = self._steps_already + n
        for info in self.locals.get("infos", []):
            lvl = info.get("level", 1)
            if lvl > self._max_level:
                self._max_level = lvl
                print(f"\n  ★ NEW MAX LEVEL: {self._max_level} ★\n")
            ep = info.get("episode")
            if ep:
                kills = info.get("kills", "?")
                level = info.get("level", "?")
                print(f"  ep  step {total_so_far:>11,}  |  reward {ep['r']:>7.1f}"
                      f"  |  kills {kills:>3}  |  level {level}")

        if n - self._last_print >= self._print_every:
            elapsed = time.time() - self._start
            total   = self._steps_already + n
            buf     = self.model.ep_info_buffer
            if buf:
                mean_r = sum(e["r"] for e in buf) / len(buf)
                self._best_reward = max(self._best_reward, mean_r)
            else:
                mean_r = float("nan")
            sps = n / elapsed if elapsed > 0 else 0
            if self._total_target < FOREVER:
                pct  = 100 * total / self._total_target
                eta  = (self._total_target - total) / sps / 3600 if sps > 0 else float("inf")
                tail = f"  ({pct:.1f}%  ETA {eta:.1f} h)"
            else:
                tail = ""
            print(
                f"  steps {total:>12,}{tail}"
                f"  |  ep_reward {mean_r:>7.2f}  (best {self._best_reward:.2f})"
                f"  |  max_level {self._max_level}"
                f"  |  {sps:>5.0f} sps  |  {elapsed/3600:.2f} h"
            )
            self._last_print = n
        return True


# ── checkpoint helpers ────────────────────────────────────────────────────────
def _load_meta() -> dict:
    if os.path.exists(META_FILE):
        with open(META_FILE) as f:
            return json.load(f)
    return {"total_steps_done": 0, "max_level_seen": 1}


def _save_meta(total: int, max_level: int):
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(META_FILE, "w") as f:
        json.dump({"total_steps_done": total, "max_level_seen": max_level}, f, indent=2)


def _latest_checkpoint() -> str | None:
    explicit = os.path.join(MODEL_DIR, "ppo_sh_latest.zip")
    if os.path.exists(explicit):
        return explicit[:-4]
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "ppo_sh_[0-9]*.zip")))
    return files[-1][:-4] if files else None


# ── env factory ──────────────────────────────────────────────────────────────
def make_env(headless: bool, backend: str = "node", action_repeat: int = 3):
    def _init():
        if backend == "node":
            return Monitor(NodeEnv(GAME_PATH, action_repeat=action_repeat))
        return Monitor(SpaceHuggersEnv(GAME_PATH, headless=headless,
                                       action_repeat=action_repeat))
    return _init


# ── main ─────────────────────────────────────────────────────────────────────
def main(total_target: int, n_envs: int, headless: bool,
         backend: str = "node", action_repeat: int = 3):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    meta           = _load_meta()
    steps_done     = meta["total_steps_done"]
    max_level_seen = meta.get("max_level_seen", 1)
    remaining      = total_target - steps_done

    if remaining <= 0 and total_target < FOREVER:
        print(f"Already at {steps_done:,} / {total_target:,} steps.")
        print("Use --forever or increase --total to keep training.")
        return

    # Build environments
    print(f"Backend: {backend}  |  {n_envs} env(s)  |  action_repeat={action_repeat}")
    if n_envs == 1:
        if backend == "node":
            env = Monitor(NodeEnv(GAME_PATH, action_repeat=action_repeat))
        else:
            env = Monitor(SpaceHuggersEnv(GAME_PATH, headless=headless,
                                          action_repeat=action_repeat))
    else:
        if backend == "node":
            factories = [make_env(headless=True, backend="node",
                                  action_repeat=action_repeat) for _ in range(n_envs)]
        else:
            factories = [make_env(headless=(i > 0 or headless), backend="playwright",
                                  action_repeat=action_repeat) for i in range(n_envs)]
        env = SubprocVecEnv(factories)

    ckpt = _latest_checkpoint()
    if ckpt:
        model = PPO.load(ckpt, env=env, tensorboard_log=LOG_DIR)
        # Trust the checkpoint's own num_timesteps — it's the ground truth.
        # meta.json can drift; override it.
        steps_done = model.num_timesteps
        # Force-override ent_coef: the saved model stores the old value (0.02).
        # At 10M+ steps we want 0.005 — commit rather than explore randomly.
        model.ent_coef = 0.005
        label = "forever" if total_target >= FOREVER else f"{total_target:,}"
        print(f"Resuming: {ckpt}.zip  |  done {steps_done:,}  |  target {label}  |  ent_coef=0.005")
    else:
        sps_est = n_envs * 15
        h_est   = remaining / sps_est / 3600 if total_target < FOREVER else None
        eta_str = f"~{h_est:.0f} h" if h_est else "until Ctrl+C"
        print(f"Fresh start — {n_envs} env(s), ~{sps_est} sps, {eta_str}")
        model = PPO(
            "MlpPolicy", env,
            # ── core ──────────────────────────────────────────────────────────
            learning_rate = 2.5e-4,     # slightly lower than default → stable
            n_steps       = 512,        # steps per env before each update
            batch_size    = 256,        # minibatch size for gradient updates
            n_epochs      = 10,         # passes over the rollout buffer
            # ── discounting ────────────────────────────────────────────────────
            gamma         = 0.99,       # future reward discount
            gae_lambda    = 0.95,       # GAE bias/variance trade-off
            # ── exploration ────────────────────────────────────────────────────
            ent_coef      = 0.005,      # low entropy — policy should commit at 10M+ steps
            # ── stability ──────────────────────────────────────────────────────
            clip_range    = 0.2,        # PPO clip parameter
            vf_coef       = 0.5,        # value function loss coefficient
            max_grad_norm = 0.5,        # gradient clipping
            # ── logging ────────────────────────────────────────────────────────
            verbose          = 0,
            tensorboard_log  = LOG_DIR,
            # ── network ────────────────────────────────────────────────────────
            # 256×256 MLP handles the 141-dim observation well
            policy_kwargs = dict(net_arch=[256, 256]),
        )

    ts_before_learn = model.num_timesteps   # snapshot before learn() advances it

    prog_cb = ProgressCallback(
        print_every    = 5_000,
        steps_already  = steps_done,
        total_target   = total_target,
        max_level_seen = max_level_seen,
    )

    callbacks = [
        prog_cb,
        RollingCheckpointCallback(
            keep        = KEEP_CKPTS,
            save_freq   = SAVE_FREQ,
            save_path   = MODEL_DIR,
            name_prefix = "ppo_sh",
            verbose     = 0,
        ),
    ]

    try:
        model.learn(
            total_timesteps    = remaining,
            callback           = callbacks,
            reset_num_timesteps = False,
            tb_log_name        = "ppo_sh",
        )
        if total_target < FOREVER:
            print("\nTarget reached!")
    except KeyboardInterrupt:
        print("\nInterrupted — saving...")

    final = os.path.join(MODEL_DIR, "ppo_sh_latest")
    model.save(final)

    # Only count steps gathered THIS session to avoid compounding double-count
    session_steps = model.num_timesteps - ts_before_learn
    new_total     = steps_done + session_steps
    _save_meta(new_total, prog_cb._max_level)

    # SubprocVecEnv workers may already be dead after Ctrl+C
    try:
        env.close()
    except Exception:
        pass

    print(f"Saved {final}.zip")
    print(f"Total lifetime steps: {new_total:,}")
    if total_target < FOREVER:
        print(f"Resume: python train_game.py --total {total_target} --envs {n_envs}")
    else:
        print(f"Resume: python train_game.py --forever --envs {n_envs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--total",    type=int, default=4_000_000)
    ap.add_argument("--forever",  action="store_true")
    ap.add_argument("--envs",     type=int, default=6)
    ap.add_argument("--headless", type=str, default="true")
    ap.add_argument("--backend",  type=str, default="node",
                    choices=["node", "playwright"],
                    help="node (fast, default) or playwright (browser)")
    ap.add_argument("--repeat",   type=int, default=3,
                    help="game frames per step (default 3 = ~50ms game time). "
                         "Higher = fewer pipe round-trips, more game throughput. "
                         "Match training: node default=3, playwright default=1.")
    args = ap.parse_args()

    target = FOREVER if args.forever else args.total
    main(target, args.envs, args.headless.lower() != "false",
         args.backend, args.repeat)
