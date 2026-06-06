"""
Quick check: launch one headless browser and measure the game's actual FPS
by comparing game-time elapsed vs real-time elapsed over 3 seconds.

Should be close to 60 fps if running at full speed.

Usage:  python check_fps.py
"""
import time
import os
from env import SpaceHuggersEnv

GAME_PATH = os.environ.get(
    "GAME_PATH",
    r"D:\tmaco0\Onedrive - Department of Education\Documents\Downloads\sd card\rk-games\games\SpaceHuggers-main"
    if os.name == "nt" else
    os.path.join(os.path.dirname(__file__), "SpaceHuggers-main")
)

env = SpaceHuggersEnv(GAME_PATH, headless=True, frame_ms=50)
env.reset()

# Read game time via JS — LittleJS exposes `time` (game seconds elapsed)
get_time = "typeof time !== 'undefined' ? time : (typeof engineTimer !== 'undefined' ? engineTimer.elapsed() : -1)"

t0_real = time.time()
t0_game = env._page.evaluate(get_time)

print("Measuring game speed for 3 seconds...")
time.sleep(3.0)

t1_real = time.time()
t1_game = env._page.evaluate(get_time)
env.close()

real_elapsed = t1_real - t0_real
game_elapsed = t1_game - t0_game

if t0_game < 0 or game_elapsed <= 0:
    print("Could not read game time — check JS variable names for this LittleJS build.")
else:
    fps_effective = game_elapsed * 60 / real_elapsed  # game runs at 60 fps target
    ratio = game_elapsed / real_elapsed
    print(f"  real time  : {real_elapsed:.2f}s")
    print(f"  game time  : {game_elapsed:.2f}s")
    print(f"  speed ratio: {ratio:.2f}x  (1.0 = real-time, <1 = too slow)")
    print(f"  effective fps: ~{fps_effective:.0f} / 60 target")
    if ratio >= 0.9:
        print("  ✓ Running at full speed")
    elif ratio >= 0.5:
        print("  ⚠ Somewhat slow — training quality may be affected")
    else:
        print("  ✗ Too slow — training will be unreliable")
