"""
Watch the trained agent play SpaceHuggers.

Usage:
    python watch_game.py                         # load best/latest checkpoint
    python watch_game.py --model game_models/ppo_sh_50000
    python watch_game.py --episodes 10
"""

import argparse
import glob
import os

from stable_baselines3 import PPO

from env import SpaceHuggersEnv

GAME_PATH = os.environ.get(
    "GAME_PATH",
    r"D:\tmaco0\Onedrive - Department of Education\Documents\Downloads\sd card\rk-games\games\SpaceHuggers-main"
    if os.name == "nt" else
    os.path.join(os.path.dirname(__file__), "SpaceHuggers-main")
)


def find_model(override: str | None) -> str:
    if override:
        path = override if not override.endswith(".zip") else override[:-4]
        if os.path.exists(path + ".zip"):
            return path
        raise FileNotFoundError(f"Model not found: {override}")

    # prefer latest save, fall back to any checkpoint
    candidates = [
        os.path.join("game_models", "ppo_sh_latest"),
    ]
    checkpoints = sorted(glob.glob("game_models/ppo_sh_*.zip"))
    if checkpoints:
        candidates.append(checkpoints[-1][:-4])

    for c in candidates:
        if os.path.exists(c + ".zip"):
            return c

    raise FileNotFoundError(
        "No model found. Train first:  python train_game.py"
    )


def main(model_path: str | None, episodes: int, deterministic: bool):
    path = find_model(model_path)
    print(f"Loading: {path}.zip")
    model = PPO.load(path)

    # frame_ms=50 matches training — agent was never trained at 16ms cadence
    env   = SpaceHuggersEnv(GAME_PATH, headless=False, frame_ms=50)
    total_kills = 0

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()

        # Click the canvas so the browser registers a user gesture.
        # Without this, headed Chromium sometimes silently drops the JS-injected
        # inputs until a real interaction occurs, causing the agent to stand still.
        try:
            env._page.click("canvas", timeout=1000)
        except Exception:
            pass  # canvas may not be focusable — carry on regardless

        ep_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        total_kills += info["kills"]
        print(
            f"  Episode {ep:>3}  |  reward {ep_reward:>8.2f}  |"
            f"  kills {info['kills']}  |  level {info['level']}"
        )

    env.close()
    print(f"\nTotal kills over {episodes} episodes: {total_kills}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    default=None, help="Path to model (no .zip)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--det",      action="store_true",
                    help="Use deterministic actions (always picks argmax). "
                         "Default: stochastic sampling — less likely to get stuck.")
    args = ap.parse_args()
    main(args.model, args.episodes, deterministic=args.det)
