"""
Watch the trained agent play SpaceHuggers (renders the real game in a browser).

Usage:
    python watch_game.py                         # v1 model (game_models/)
    python watch_game.py --v2                     # v2 Tier-2 model (game_models_v2/)
    python watch_game.py --v2 --probe-every 4     # v2 + live neuron readout every 4 steps
    python watch_game.py --model game_models/ppo_sh_50000
    python watch_game.py --episodes 10 --det

--v2 loads the Dict-obs CNN model and renders with the matching v2 observation
(the browser game server now emits the full v2 perception: dig-aware geo field,
globe, dig-cost terrain). With --v2 it also prints, each step, a live "brain"
panel: some named inputs, the top-10 first-policy-layer neurons (value in -> out),
and ALL action outputs (logit + probability, chosen marked).
"""

import argparse
import glob
import os
import sys

import numpy as np
from stable_baselines3 import PPO

from env import SpaceHuggersEnv

GAME_PATH = os.environ.get(
    "GAME_PATH",
    r"D:\tmaco0\Onedrive - Department of Education\Documents\Downloads\sd card\rk-games\games\SpaceHuggers-main"
    if os.name == "nt" else
    os.path.join(os.path.dirname(__file__), "SpaceHuggers-main")
)


def find_model(override: str | None, v2: bool) -> str:
    if override:
        path = override if not override.endswith(".zip") else override[:-4]
        if os.path.exists(path + ".zip"):
            return path
        raise FileNotFoundError(f"Model not found: {override}")

    mdir = "game_models_v2" if v2 else "game_models"
    candidates = [os.path.join(mdir, "ppo_sh_best"),
                  os.path.join(mdir, "ppo_sh_latest")]
    checkpoints = sorted(glob.glob(os.path.join(mdir, "ppo_sh_*.zip")))
    if checkpoints:
        candidates.append(checkpoints[-1][:-4])
    for c in candidates:
        if os.path.exists(c + ".zip"):
            return c
    raise FileNotFoundError(f"No model found in {mdir}/. Train first.")


# ── neuron readout ────────────────────────────────────────────────────────────
# Named, human-readable slices of the 106-dim v2 vector (the "some inputs" view).
VEC_INPUTS = [
    ("health",    4),  ("grenades",  6),  ("dodge_rdy", 7),  ("on_fire",  8),
    ("level",     9),  ("enem_left", 10), ("n_enemies", 11), ("facing",   12),
    ("near_dx",   13), ("near_dy",   14), ("near_hp",   15),
    ("geo_dist",  93), ("geo_dir_x", 94), ("geo_dir_y", 95),
]
# Action head layout: MultiDiscrete([3,3,2,2,2]) -> 12 logits, in this order.
ACTION_GROUPS = [
    ("horizontal", ["none", "left", "right"]),
    ("vertical",   ["none", "up", "down"]),
    ("shoot",      ["no", "yes"]),
    ("dodge",      ["no", "yes"]),
    ("grenade",    ["no", "throw"]),
]


class NeuronProbe:
    """Forward-hooks the first policy hidden layer and the action head, so we can
    read each neuron's pre-activation ('in') and post-activation ('out'), plus the
    full action-logit vector, on every prediction."""

    def __init__(self, model: PPO):
        self.model = model
        self._pre = self._post = self._logits = None

        pnet = model.policy.mlp_extractor.policy_net   # Sequential(Linear, act, ...)
        # first activation = index of the first non-Linear module after Linear[0]
        act_idx = 1
        for i in range(1, len(pnet)):
            if not hasattr(pnet[i], "weight"):
                act_idx = i
                break
        self.layer_name = f"policy_net[{act_idx}]  ({pnet[0].out_features} neurons)"
        pnet[act_idx].register_forward_hook(self._hidden_hook)
        model.policy.action_net.register_forward_hook(self._action_hook)

    def _hidden_hook(self, mod, inp, out):
        self._pre  = inp[0].detach().cpu().numpy().ravel()   # going IN  (pre-activation)
        self._post = out.detach().cpu().numpy().ravel()      # coming OUT (post-activation)

    def _action_hook(self, mod, inp, out):
        self._logits = out.detach().cpu().numpy().ravel()

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def render(self, obs: dict, action, step: int) -> str:
        vec = np.asarray(obs["vector"]).ravel()
        L = ["", "=" * 66, f" STEP {step:<6}   brain readout   (layer: {self.layer_name})",
             "-" * 66]

        # INPUTS (some, named)
        L.append(" INPUTS (selected):")
        row = ""
        for j, (name, idx) in enumerate(VEC_INPUTS):
            row += f"  {name:>9}={vec[idx]:+6.3f}"
            if j % 3 == 2:
                L.append(row); row = ""
        if row:
            L.append(row)

        # HIDDEN top-10 neurons by |output|
        L.append("-" * 66)
        L.append(" TOP-10 HIDDEN NEURONS        in (pre)    ->   out (post)")
        if self._post is not None:
            order = np.argsort(-np.abs(self._post))[:10]
            for r in order:
                pre = self._pre[r] if self._pre is not None else float("nan")
                L.append(f"   #{int(r):<4}                  {pre:+8.4f}   ->  {self._post[r]:+8.4f}")

        # OUTPUTS (all action neurons)
        L.append("-" * 66)
        L.append(" OUTPUTS (all action neurons):     logit     prob")
        act = [int(a) for a in np.asarray(action).ravel()]
        off = 0
        for g, (name, labels) in enumerate(ACTION_GROUPS):
            n = len(labels)
            logits = (self._logits[off:off + n] if self._logits is not None
                      else np.zeros(n))
            probs = self._softmax(logits)
            chosen = act[g] if g < len(act) else -1
            L.append(f"  {name}:")
            for k, lab in enumerate(labels):
                mark = "  <= CHOSEN" if k == chosen else ""
                L.append(f"      {lab:>6}   {logits[k]:+8.4f}   {probs[k]:5.2f}{mark}")
            off += n
        L.append("=" * 66)
        return "\n".join(L)


def _ascii_maps(obs) -> str:
    """Two egocentric 13x9 maps of what the agent sees: terrain/enemies, and the
    geodesic reachability field. Rendered with UP at the top."""
    grid = np.asarray(obs["grid"])                       # (6,9,13)
    solid, ladder, enemy, hazard, _lof, reach = grid
    H, W = solid.shape
    pr, pc = H // 2, W // 2                               # player at centre (4,6)

    def sym(r, c):
        if r == pr and c == pc: return "@"
        if enemy[r, c] > 0:     return "E"
        if ladder[r, c] > 0:    return "H"
        s = solid[r, c]
        if s >= 0.9:            return "#"                # hard wall (~20 shots)
        if s >= 0.4:            return "%"                # dirt (~5 shots)
        if s >= 0.1:            return '"'                # glass (1 shot)
        if hazard[r, c] > 0:    return "*"
        return "."

    out = ['  TERRAIN/ENEMIES  (@=you E=enemy #=wall %=dirt "=glass H=ladder *=hazard)  [top=up]']
    for r in range(H - 1, -1, -1):
        out.append("    " + "".join(sym(r, c) for c in range(W)))
    out.append("  REACHABILITY  (0=here .. 9=far   .=can't reach   @=you)  [top=up]")
    for r in range(H - 1, -1, -1):
        row = ""
        for c in range(W):
            if r == pr and c == pc:
                row += "@"
            else:
                v = reach[r, c]
                row += "." if v >= 0.999 else str(int(min(9, v * 9)))
        out.append("    " + row)
    return "\n".join(out)


_ARROWS = {(0, 0): "(none)", (0, 1): "UP", (0, -1): "DOWN", (1, 0): "RIGHT",
           (-1, 0): "LEFT", (1, 1): "up-right", (-1, 1): "up-left",
           (1, -1): "down-right", (-1, -1): "down-left"}


def render_dump(obs, action, probe, step) -> str:
    """Full diagnostic snapshot of a (stuck) state."""
    vec = np.asarray(obs["vector"]).ravel()
    gdx, gdy = float(vec[94]), float(vec[95])
    arrow = _ARROWS.get((int(np.sign(gdx)), int(np.sign(gdy))), "?")
    ndy = float(vec[14])   # nearest enemy y (normalised; >0 / <0 = which side vertically)

    # crude case hint (matches the case 1/2 framing)
    if abs(vec[93]) >= 0.999:
        hint = "enemy effectively UNREACHABLE (geo_dist=1) -> field found no path at all"
    elif abs(gdx) < 0.5 and gdy != 0 and np.sign(gdy) == np.sign(ndy):
        hint = ("CASE 1? geo_dir points straight at the enemy vertically - if it can't "
                "climb, the field is promising an impossible jump")
    elif gdx != 0:
        hint = ("CASE 2? geo_dir points SIDEWAYS (a detour route) - field is right, "
                "agent may just not be following it yet")
    else:
        hint = "ambiguous"

    L = ["", "#" * 72, f" STUCK-STATE DUMP   step={step}", "#" * 72,
         f"  geo_dist={vec[93]:.3f}   geo_dir=({gdx:+.0f},{gdy:+.0f}) -> {arrow}"
         f"   facing={vec[12]:+.0f}",
         f"  health={vec[4]:+.2f}  grenades={vec[6]:+.2f}  n_enemies={vec[11]*10:.0f}"
         f"  enemies_left={vec[10]*100:.0f}",
         "  nearest enemies (rel x,y normalised | approx tiles dx,dy):"]
    for i in range(3):
        ex, ey = float(vec[13 + i * 3]), float(vec[14 + i * 3])
        if ex == 0 and ey == 0:
            continue
        L.append(f"     #{i}:  x={ex:+.3f} y={ey:+.3f}   (~dx={ex*200:+5.0f}, dy={ey*100:+5.0f} tiles)")
    L.append(f"  HINT: {hint}")
    L.append("-" * 72)
    L.append(_ascii_maps(obs))
    if probe is not None:
        L.append(probe.render(obs, action, step))
    L.append("#" * 72)
    return "\n".join(L)


def main(model_path, episodes, deterministic, v2, probe_every, probe):
    # v2 needs the custom feature-extractor class importable for unpickling.
    if v2:
        import train_game
        from train_game import SpatialExtractor
        setattr(sys.modules["__main__"], "SpatialExtractor", SpatialExtractor)

    path = find_model(model_path, v2)
    print(f"Loading: {path}.zip  ({'v2 Tier-2' if v2 else 'v1'})")
    model = PPO.load(path)

    # Probe exists for all v2 runs (used by both the live panel and L-key dumps);
    # only the live *printing* is gated by --no-probe / --probe-every.
    probe_obj = NeuronProbe(model) if v2 else None
    DUMP_FILE = "watch_dumps.txt"

    # frame_ms=50 matches training cadence
    env = SpaceHuggersEnv(GAME_PATH, headless=False, frame_ms=50,
                          obs_version=2 if v2 else 1)
    total_kills = 0
    last_dump = 0
    if v2:
        print(f"[watch] press 'L' in the game window to dump the current state -> {DUMP_FILE}")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        try:
            env._page.click("canvas", timeout=1000)
        except Exception:
            pass

        ep_reward = 0.0
        done = False
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            if probe_obj is not None and probe and step % probe_every == 0:
                print(probe_obj.render(obs, action, step))

            # manual L-key state dump (poll the in-page flag)
            if v2:
                try:
                    req = int(env._page.evaluate("window._dumpReq||0"))
                except Exception:
                    req = last_dump
                if req > last_dump:
                    last_dump = req
                    dump = render_dump(obs, action, probe_obj, step)
                    with open(DUMP_FILE, "a", encoding="utf-8") as fh:
                        fh.write(dump + "\n")
                    print(f"  >> dumped state #{req} (ep {ep}, step {step}) -> {DUMP_FILE}")

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1
            done = terminated or truncated

        total_kills += info["kills"]
        print(f"  Episode {ep:>3}  |  reward {ep_reward:>8.2f}  |"
              f"  kills {info['kills']}  |  level {info['level']}")

    env.close()
    print(f"\nTotal kills over {episodes} episodes: {total_kills}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    default=None, help="Path to model (no .zip)")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--v2",       action="store_true",
                    help="Load the Tier-2 Dict-obs CNN model (game_models_v2/)")
    ap.add_argument("--det",      action="store_true",
                    help="Deterministic actions (argmax). Default: stochastic sampling.")
    ap.add_argument("--probe-every", type=int, default=4,
                    help="(v2) print the neuron readout every N steps (default 4)")
    ap.add_argument("--no-probe", action="store_true",
                    help="(v2) disable the neuron readout panel")
    args = ap.parse_args()
    main(args.model, args.episodes, deterministic=args.det, v2=args.v2,
         probe_every=max(1, args.probe_every), probe=not args.no_probe)
