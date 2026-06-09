"""
Standalone throughput benchmark for the node-workers backend.

Measures PURE env-stepping speed with random actions — NO PPO, NO policy
network, NO gradient updates.  This isolates the backend (pipe + Node main
thread + parallel workers + Python post-processing) from the training loop, so
the sps number here is the *ceiling* the backend can deliver.

It also breaks each step into:
    rpc   — step_batch round-trip (pipe write → Node steps N workers → read)
    post  — Python reward/obs/done computation (GIL-bound, serial over N envs)
    reset — auto-reset round-trips for episodes that ended

Usage:
    python bench_node.py                  # 12 envs, 500 steps, repeat 3
    python bench_node.py 6                 # 6 envs
    python bench_node.py 16 800 3          # 16 envs, 800 steps, repeat 3
"""

import os
import sys
import time

import numpy as np

os.environ['NODEBATCH_PROFILE'] = '1'

from env_node_batch import NodeBatchVecEnv

GAME_PATH = os.environ.get(
    'GAME_PATH',
    r'D:\tmaco0\Onedrive - Department of Education\Documents\Downloads\sd card\rk-games\games\SpaceHuggers-main'
)


def bench(n_envs: int, n_steps: int, repeat: int):
    print(f'\n=== {n_envs} envs | {n_steps} steps | repeat={repeat} ===')
    t_launch = time.perf_counter()
    env = NodeBatchVecEnv(GAME_PATH, n_envs=n_envs, action_repeat=repeat)
    print(f'  startup: {time.perf_counter() - t_launch:5.1f} s')

    rng = np.random.default_rng(0)
    # MultiDiscrete([3,3,2,2,2]) — sample valid random actions
    highs = np.array([3, 3, 2, 2, 2])

    env.reset()

    # warm up (let JIT / GC settle) — not counted
    for _ in range(20):
        a = rng.integers(0, highs, size=(n_envs, 5))
        env.step_async(a)
        env.step_wait()

    env._prof_rpc = env._prof_post = env._prof_reset = 0.0
    env._prof_n = 0

    t0 = time.perf_counter()
    for _ in range(n_steps):
        a = rng.integers(0, highs, size=(n_envs, 5))
        env.step_async(a)
        env.step_wait()
    elapsed = time.perf_counter() - t0

    env_steps = n_steps * n_envs
    sps = env_steps / elapsed
    per_batch_ms = elapsed / n_steps * 1000

    n = max(1, env._prof_n)
    rpc_ms   = env._prof_rpc   / n * 1000
    post_ms  = env._prof_post  / n * 1000
    reset_ms = env._prof_reset / n * 1000

    print(f'  throughput : {sps:7.0f} sps   ({per_batch_ms:5.1f} ms/batch)')
    print(f'  breakdown  : rpc {rpc_ms:5.1f} ms  |  post {post_ms:5.1f} ms  |  reset {reset_ms:5.1f} ms')
    print(f'  rpc share  : {100*rpc_ms/per_batch_ms:4.0f}%   (Node/pipe vs Python)')

    env.close()
    return sps


if __name__ == '__main__':
    n_envs  = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    repeat  = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    bench(n_envs, n_steps, repeat)
