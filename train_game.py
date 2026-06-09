"""
Train PPO on SpaceHuggers.  Fully resumable across any number of sessions.

Usage:
    python train_game.py                         # train / resume (4M step target)
    python train_game.py --forever               # run until Ctrl+C, no step limit
    python train_game.py --total 10000000        # custom target
    python train_game.py --envs 6                # parallel envs (default 6)
    python train_game.py --headless false        # watch one env (playwright only)
    python train_game.py --backend playwright    # use Chromium browser backend
    python train_game.py --backend node-workers  # worker_threads backend (fastest)

Backends:
    node-workers (fastest) — worker_threads, true parallelism, ~300-350 sps
    node-batch             — single Node process, N vm contexts, ~150 sps
    node                   — SubprocVecEnv + NodeEnv, ~80 sps
    playwright             — Chromium browser,  ~50 ms/step,   ~500 MB/env

Speed estimates (i5-1240p, 12 envs, node-workers backend):
    ~300-350 sps  →  same 4-day run ≈ 100M+ steps

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

import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env import SpaceHuggersEnv
from env_node import NodeEnv
from env_node_batch import NodeBatchVecEnv, resolve_pin

# ── config ───────────────────────────────────────────────────────────────────
GAME_PATH  = os.environ.get(
    "GAME_PATH",
    r"D:\tmaco0\Onedrive - Department of Education\Documents\Downloads\sd card\rk-games\games\SpaceHuggers-main"
    if os.name == "nt" else
    os.path.join(os.path.dirname(__file__), "SpaceHuggers-main")
)
MODEL_DIR  = "game_models"
LOG_DIR    = "game_logs"
# Tier-2 (--v2) uses a SEPARATE namespace so it never touches the v1 models.
# v2 has a different observation (210-dim) and network, so weights are
# incompatible — it must train fresh in its own dir.
MODEL_DIR_V2 = "game_models_v2"
LOG_DIR_V2   = "game_logs_v2"
KEEP_CKPTS = 5
SAVE_FREQ  = 5_000      # steps between saves — less churn than 1500
FOREVER    = 10**12     # effectively infinite step target

# Tier-2 network: wider/deeper MLP on the richer 210-dim observation.
NET_ARCH_V2 = [512, 768, 512]

# Entropy coefficient — raised from 0.005 to re-inject exploration after the
# policy collapsed into a passive/turtle local optimum (entropy_loss had decayed
# to ~-2.0).  Higher = more exploration.  Override per-run with --ent.
ENT_COEF   = 0.015

# Discount factor — raised from 0.99 to 0.997 so the sparse +150 level-completion
# reward actually propagates back.  At 0.99 the effective horizon is ~100 steps
# (~5 s of game time); clearing a level takes far longer, so the agent only
# optimised the dense +10 kills it could "see" and ignored finishing the level.
# 0.997 → ~333-step horizon.  Resume-safe (GAE param, not a network weight).
# If value loss / explained_variance destabilises, dial back toward 0.995.
GAMMA      = 0.997


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


# ── save-best callback ────────────────────────────────────────────────────────
class SaveBestCallback(BaseCallback):
    """
    Saves model to ppo_sh_best.zip whenever the rolling mean episode reward
    (model.ep_info_buffer) reaches a new high.  The best value persists across
    sessions in best.json, so resuming never overwrites a good model with a
    worse one.  This protects peaks that the rolling checkpoints (last 5 only)
    would otherwise silently lose.
    """
    def __init__(self, save_path: str, best_file: str,
                 check_freq: int = 5_000, min_episodes: int = 20,
                 margin: float = 0.0):
        super().__init__()
        self._save_path    = save_path
        self._best_file    = best_file
        self._check_freq   = check_freq
        self._min_episodes = min_episodes
        self._margin       = margin
        self._last_check   = 0
        # Load persisted best so a resume doesn't reset the bar to -inf.
        self._best = float("-inf")
        if os.path.exists(best_file):
            try:
                with open(best_file) as f:
                    self._best = json.load(f).get("best_ep_rew_mean", float("-inf"))
            except Exception:
                pass

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_check < self._check_freq:
            return True
        self._last_check = self.num_timesteps

        buf = self.model.ep_info_buffer
        if not buf or len(buf) < self._min_episodes:
            return True

        mean_r = sum(e["r"] for e in buf) / len(buf)
        if mean_r > self._best + self._margin:
            prev = self._best
            self._best = mean_r
            best_path = os.path.join(self._save_path, "ppo_sh_best")
            self.model.save(best_path)
            try:
                os.makedirs(self._save_path, exist_ok=True)
                with open(self._best_file, "w") as f:
                    json.dump({"best_ep_rew_mean": mean_r,
                               "step": int(self.num_timesteps)}, f, indent=2)
            except Exception:
                pass
            prev_str = "none" if prev == float("-inf") else f"{prev:.1f}"
            print(f"  ★ NEW BEST ep_rew_mean {mean_r:.1f} (was {prev_str}) "
                  f"→ saved ppo_sh_best.zip")
        return True


# ── Tier-2 CNN feature extractor (Dict obs: grid image + flat vector) ─────────
class SpatialExtractor(BaseFeaturesExtractor):
    """CNN over the local grid + CNN over the coarse global map + MLP over the
    vector, concatenated.  The 'un-flattened' network: convolutions read 2D
    layout with locality + translation invariance an MLP can't get."""
    def __init__(self, observation_space, cnn_out: int = 128,
                 globe_out: int = 64, vec_out: int = 256):
        super().__init__(observation_space, features_dim=cnn_out + globe_out + vec_out)
        n_ch = observation_space["grid"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flat = self.cnn(th.zeros(1, *observation_space["grid"].shape)).shape[1]
        self.cnn_head = nn.Sequential(nn.Linear(n_flat, cnn_out), nn.ReLU())
        # coarse global map branch
        g_ch = observation_space["globe"].shape[0]
        self.gcnn = nn.Sequential(
            nn.Conv2d(g_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            g_flat = self.gcnn(th.zeros(1, *observation_space["globe"].shape)).shape[1]
        self.gcnn_head = nn.Sequential(nn.Linear(g_flat, globe_out), nn.ReLU())
        self.vec_head = nn.Sequential(
            nn.Linear(observation_space["vector"].shape[0], vec_out), nn.ReLU())

    def forward(self, obs):
        g  = self.cnn_head(self.cnn(obs["grid"]))
        gl = self.gcnn_head(self.gcnn(obs["globe"]))
        v  = self.vec_head(obs["vector"])
        return th.cat([g, gl, v], dim=1)


def linear_schedule(initial: float):
    """LR schedule: decays linearly with remaining progress (≈constant for --forever)."""
    return lambda progress_remaining: initial * progress_remaining


# ── entropy-coefficient schedule (explore early, commit later) ────────────────
class EntropyScheduleCallback(BaseCallback):
    def __init__(self, start: float, end: float, horizon: int):
        super().__init__()
        self._start, self._end, self._horizon = start, end, horizon

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / max(1, self._horizon))
        self.model.ent_coef = self._start + (self._end - self._start) * frac
        return True


# ── save-best-by-LEVEL (honest metric — reward is gameable) ───────────────────
class SaveBestByLevelCallback(BaseCallback):
    def __init__(self, save_path: str, check_freq: int = 5_000, window: int = 50):
        super().__init__()
        self._save_path  = save_path
        self._check_freq = check_freq
        self._window     = window
        self._levels: list[int] = []
        self._best = float("-inf")
        self._last_check = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if info.get("episode") is not None:
                self._levels.append(info.get("level", 1))
                self._levels = self._levels[-self._window:]
        if self.num_timesteps - self._last_check < self._check_freq:
            return True
        self._last_check = self.num_timesteps
        if len(self._levels) >= 10:
            mean_lvl = sum(self._levels) / len(self._levels)
            if mean_lvl > self._best:
                self._best = mean_lvl
                self.model.save(os.path.join(self._save_path, "ppo_sh_bestlevel"))
                print(f"  ★ NEW BEST mean level {mean_lvl:.2f} → saved ppo_sh_bestlevel.zip")
        return True


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
def _load_meta(meta_file: str) -> dict:
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            return json.load(f)
    return {"total_steps_done": 0, "max_level_seen": 1}


def _save_meta(total: int, max_level: int, meta_file: str):
    os.makedirs(os.path.dirname(meta_file), exist_ok=True)
    with open(meta_file, "w") as f:
        json.dump({"total_steps_done": total, "max_level_seen": max_level}, f, indent=2)


def _latest_checkpoint(model_dir: str) -> str | None:
    explicit = os.path.join(model_dir, "ppo_sh_latest.zip")
    if os.path.exists(explicit):
        return explicit[:-4]
    files = sorted(glob.glob(os.path.join(model_dir, "ppo_sh_[0-9]*.zip")))
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
         backend: str = "node", action_repeat: int = 3, pin: str = "none",
         monitor: bool = False, ent_coef: float = ENT_COEF, gamma: float = GAMMA,
         cap: int = 30_000, v2: bool = False):
    # Tier-2 (--v2): richer 210-dim observation + bigger net, in a SEPARATE
    # namespace so it never touches v1.  The v2 obs is only built by
    # NodeBatchVecEnv, so v2 requires the batch/workers backend.
    if v2 and backend not in ("node-batch", "node-workers"):
        print(f"ERROR: --v2 requires --backend node-workers (or node-batch); "
              f"got '{backend}'. The v2 observation is only built by NodeBatchVecEnv.")
        return
    model_dir   = MODEL_DIR_V2 if v2 else MODEL_DIR
    log_dir     = LOG_DIR_V2   if v2 else LOG_DIR
    meta_file   = os.path.join(model_dir, "meta.json")
    best_file   = os.path.join(model_dir, "best.json")
    obs_version = 2 if v2 else 1
    net_arch    = dict(pi=NET_ARCH_V2, vf=NET_ARCH_V2) if v2 else [256, 256]
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    if v2:
        print(f"=== TIER-2 (--v2) === models→{model_dir}/  "
              f"obs=Dict(grid 6x9x13 + globe 3x8x8 + vec 106)  "
              f"2xCNN+MLP→{NET_ARCH_V2}  (separate namespace, trains fresh)")

    # ── optional live temp/clock monitor (--v) ────────────────────────────────
    # Launches temp_monitor.ps1 in its own console window (no admin needed).
    # Auto-closed on exit via atexit.  Off by default for every preset.
    if monitor and os.name == "nt":
        try:
            import atexit
            import subprocess as _sp
            _mon_path = os.path.join(os.path.dirname(__file__), "temp_monitor.ps1")
            _mon_proc = _sp.Popen(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-File", _mon_path],
                creationflags=_sp.CREATE_NEW_CONSOLE,
            )
            atexit.register(lambda: _mon_proc.terminate() if _mon_proc.poll() is None else None)
            print("Live monitor: ON (--v) — temp + clock throttle in separate window")
        except Exception as e:
            print(f"WARN: could not launch --v monitor ({e})")
    elif monitor:
        print("NOTE: --v monitor is Windows-only; ignoring on this platform.")

    # ── CPU affinity (P-core / E-core pinning) ────────────────────────────────
    # Pin BOTH the Python process and the Node process to the same set of
    # logical CPUs.  On hybrid CPUs (e.g. i5-1240p) this keeps competing apps
    # (Opera, etc.) off our cores and stops Windows migrating workers onto slow
    # E-cores.  --pin p = P-cores only, --pin e = E-cores only, --pin none = off.
    pin_cpus = resolve_pin(pin)
    if pin_cpus:
        try:
            import psutil
            psutil.Process().cpu_affinity(pin_cpus)
            print(f"Pinned Python process to CPUs: {pin_cpus}  (--pin {pin})")
        except Exception as e:
            print(f"WARN: failed to pin Python process ({e}) — continuing unpinned")
            pin_cpus = None
        if backend not in ("node-batch", "node-workers"):
            print(f"NOTE: --pin only pins the Node child for backends 'node-batch' "
                  f"and 'node-workers'.  Backend '{backend}' will inherit Python's "
                  f"affinity but spawned subprocesses are not explicitly pinned.")

    meta           = _load_meta(meta_file)
    steps_done     = meta["total_steps_done"]
    max_level_seen = meta.get("max_level_seen", 1)
    remaining      = total_target - steps_done

    if remaining <= 0 and total_target < FOREVER:
        print(f"Already at {steps_done:,} / {total_target:,} steps.")
        print("Use --forever or increase --total to keep training.")
        return

    # Build environments
    print(f"Backend: {backend}  |  {n_envs} env(s)  |  action_repeat={action_repeat}")
    if backend in ("node-batch", "node-workers"):
        # Single Node process, N game instances via MessageChannel worker threads
        # (node-workers) or vm.createContext (node-batch, legacy).
        # Both use the same NodeBatchVecEnv Python class and batch protocol;
        # the JS server is selected by _GAME_SERVER in env_node_batch.py.
        env = NodeBatchVecEnv(GAME_PATH, n_envs=n_envs,
                              action_repeat=action_repeat, pin_cpus=pin_cpus,
                              max_steps=cap, obs_version=obs_version)
    elif n_envs == 1:
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

    ckpt = _latest_checkpoint(model_dir)
    if ckpt:
        model = PPO.load(ckpt, env=env, tensorboard_log=log_dir)
        # Trust the checkpoint's own num_timesteps — it's the ground truth.
        # meta.json can drift; override it.
        steps_done = model.num_timesteps
        # Force-override ent_coef + gamma on resume (the saved model stores the
        # old values).  ent_coef raised to re-inject exploration; gamma raised so
        # the +150 level reward propagates.  Both are resume-safe (no weight
        # change): ent_coef is a loss term, gamma is used in GAE/returns.
        model.ent_coef = ent_coef
        model.gamma    = gamma
        label = "forever" if total_target >= FOREVER else f"{total_target:,}"
        print(f"Resuming: {ckpt}.zip  |  done {steps_done:,}  |  target {label}  "
              f"|  ent_coef={ent_coef}  |  gamma={gamma}")
    else:
        sps_est = n_envs * 15
        h_est   = remaining / sps_est / 3600 if total_target < FOREVER else None
        eta_str = f"~{h_est:.0f} h" if h_est else "until Ctrl+C"
        print(f"Fresh start — {n_envs} env(s), ~{sps_est} sps, {eta_str}")
        policy  = "MultiInputPolicy" if v2 else "MlpPolicy"
        lr      = linear_schedule(2.5e-4) if v2 else 2.5e-4
        pkwargs = dict(net_arch=net_arch)
        if v2:
            pkwargs["features_extractor_class"] = SpatialExtractor   # CNN+MLP
        model = PPO(
            policy, env,
            # ── core ──────────────────────────────────────────────────────────
            learning_rate = lr,         # v2: linear decay; v1: flat 2.5e-4
            n_steps       = 512,        # steps per env before each update
            batch_size    = 256,        # minibatch size for gradient updates
            n_epochs      = 10,         # passes over the rollout buffer
            # ── discounting ────────────────────────────────────────────────────
            gamma         = gamma,      # future reward discount (see GAMMA / --gamma)
            gae_lambda    = 0.95,       # GAE bias/variance trade-off
            # ── exploration ────────────────────────────────────────────────────
            ent_coef      = ent_coef,   # exploration pressure (see ENT_COEF / --ent)
            # ── stability ──────────────────────────────────────────────────────
            clip_range    = 0.2,        # PPO clip parameter
            vf_coef       = 0.5,        # value function loss coefficient
            max_grad_norm = 0.5,        # gradient clipping
            # ── logging ────────────────────────────────────────────────────────
            verbose          = 0,
            tensorboard_log  = log_dir,
            # ── network ────────────────────────────────────────────────────────
            # v1: 256×256 MLP on 141-dim obs.
            # v2: CNN(grid)+MLP(vector) feature extractor -> [512,768,512] trunk.
            policy_kwargs = pkwargs,
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
            save_path   = model_dir,
            name_prefix = "ppo_sh",
            verbose     = 0,
        ),
        SaveBestCallback(
            save_path  = model_dir,
            best_file  = best_file,
            check_freq = SAVE_FREQ,
            min_episodes = 20,
        ),
    ]
    if v2:
        # honest metric (level, not gameable reward) + entropy decay
        callbacks.append(SaveBestByLevelCallback(save_path=model_dir, check_freq=SAVE_FREQ))
        callbacks.append(EntropyScheduleCallback(start=0.02, end=0.008, horizon=5_000_000))

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

    final = os.path.join(model_dir, "ppo_sh_latest")
    model.save(final)

    # Only count steps gathered THIS session to avoid compounding double-count
    session_steps = model.num_timesteps - ts_before_learn
    new_total     = steps_done + session_steps
    _save_meta(new_total, prog_cb._max_level, meta_file)

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
                    choices=["node", "node-batch", "node-workers", "playwright"],
                    help="node-workers (fastest: worker_threads, true parallelism, ~300-350 sps) | "
                         "node-batch (1 Node process, N vm contexts, ~150 sps) | "
                         "node (SubprocVecEnv, ~80 sps) | playwright (browser)")
    ap.add_argument("--repeat",   type=int, default=3,
                    help="game frames per step (default 3 = ~50ms game time). "
                         "Higher = fewer pipe round-trips, more game throughput. "
                         "Match training: node default=3, playwright default=1.")
    ap.add_argument("--pin",      type=str, default="none",
                    choices=["none", "p", "e", "all"],
                    help="Pin Python+Node to specific cores on hybrid CPUs. "
                         "p = P-cores only (fastest, recommended when other apps "
                         "are running — forces them onto E-cores). "
                         "e = E-cores only (frees P-cores for foreground apps; "
                         "training will be ~40%% slower but won't fight Opera). "
                         "all = all cores (explicit no-op). "
                         "none = no pinning (default, lets Windows schedule). "
                         "First use detects P/E layout via a microbench (~5s, "
                         "cached). Best with --backend node-workers or node-batch.")
    ap.add_argument("--v",        dest="monitor", action="store_true",
                    help="Live CPU temp + clock-throttle monitor in a separate "
                         "window (Windows, no admin). Off by default. Watch the "
                         "'vs base' %% — below 100%% means the CPU is throttling.")
    ap.add_argument("--ent",      type=float, default=ENT_COEF,
                    help=f"Entropy coefficient (exploration pressure). "
                         f"Default {ENT_COEF}. Raise to explore more, lower to "
                         f"commit. Overrides the saved model's value on resume.")
    ap.add_argument("--gamma",    type=float, default=GAMMA,
                    help=f"Discount factor. Default {GAMMA}. Higher = longer "
                         f"reward horizon (helps the +150 level reward propagate). "
                         f"Dial toward 0.995 if value loss destabilises. "
                         f"Overrides the saved model's value on resume.")
    ap.add_argument("--cap",      type=int, default=30_000,
                    help="Max steps before an episode is truncated (timeout). "
                         "Default 30000 (~very long). LOWER it (e.g. 8000-12000) "
                         "to break the turtle/dawdle behaviour: forces decisive "
                         "play, multiplies feedback rate, and lets gamma's horizon "
                         "cover a real fraction of the episode. Resume-safe.")
    ap.add_argument("--v2",       action="store_true",
                    help="Tier-2 mode: Dict obs (multi-channel CNN grid + vector) "
                         "with a CNN+MLP extractor feeding a [512,768,512] trunk, "
                         "plus firing-position & exploration rewards, LR/entropy "
                         "schedules, save-best-by-level. Trains FRESH in "
                         "game_models_v2/, never touches v1. Needs node-workers.")
    args = ap.parse_args()

    target = FOREVER if args.forever else args.total
    main(target, args.envs, args.headless.lower() != "false",
         args.backend, args.repeat, args.pin, args.monitor, args.ent, args.gamma,
         args.cap, args.v2)
