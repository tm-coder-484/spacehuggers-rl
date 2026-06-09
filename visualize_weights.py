"""
Visualise PPO model weights while training.

Usage:
    python visualize_weights.py                 # show once (latest checkpoint)
    python visualize_weights.py --watch 60      # auto-refresh every 60 s
    python visualize_weights.py --save          # save PNG to weights_vis/ folder
    python visualize_weights.py --circuit       # circuit diagram style
    python visualize_weights.py --model path    # specific checkpoint
"""

import argparse
import glob
import os
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from stable_baselines3 import PPO

# ── observation labels (33 display nodes) ────────────────────────────────────

PLAYER_NAMES = [
    "player_x", "player_y", "vel_x", "vel_y",
    "health", "on_ground", "grenades", "dodge_ready", "on_fire",
]
ENEMY_NAMES = [
    f"enemy{i}_{f}" for i in range(5) for f in ("x", "y", "hp")
]
GRID_ROW_NAMES = [f"grid_row{r}" for r in range(9)]

# 9 player + 15 enemy + 9 grid-rows = 33 display inputs
ALL_INPUT_NAMES = PLAYER_NAMES + ENEMY_NAMES + GRID_ROW_NAMES

ACTION_NAMES = [
    "Horiz: None",   "Horiz: ← Left",  "Horiz: Right →",
    "Vert: None",    "Vert: Jump ↑",   "Vert: Down ↓",
    "Shoot: No",     "Shoot: FIRE",
    "Dodge: No",     "Dodge: Roll",
    "Grenade: No",   "Grenade: Throw",
]

MODEL_DIR = "game_models"


# ── helpers ───────────────────────────────────────────────────────────────────

def find_model(override=None):
    if override:
        p = override if not override.endswith(".zip") else override[:-4]
        if os.path.exists(p + ".zip"):
            return p
        raise FileNotFoundError(f"Not found: {override}")
    for candidate in [
        os.path.join(MODEL_DIR, "ppo_sh_latest"),
        *sorted(glob.glob(os.path.join(MODEL_DIR, "ppo_sh_[0-9]*.zip")))
    ]:
        path = candidate.replace(".zip", "")
        if os.path.exists(path + ".zip"):
            return path
    raise FileNotFoundError("No checkpoint found. Run train_game.py first.")


def load_weights(model_path):
    """Extract actor weights from checkpoint, aggregating tile grid by row."""
    model = PPO.load(model_path)
    p = {n: v.detach().numpy() for n, v in model.policy.named_parameters()}

    W1_raw = p.get("mlp_extractor.policy_net.0.weight")  # [256, 141]
    W2     = p.get("mlp_extractor.policy_net.2.weight")  # [256, 256]
    W_act  = p.get("action_net.weight")                  # [12,  256]

    if W1_raw is None:
        keys = list(p.keys())
        raise RuntimeError(
            f"Unexpected parameter names. Got: {keys[:8]}…\n"
            "Model architecture may have changed."
        )

    # Aggregate 117 tile values → 9 row means  (9 rows × 13 cols = 117)
    W1_player    = W1_raw[:, :9]                                    # [256, 9]
    W1_enemy     = W1_raw[:, 9:24]                                  # [256, 15]
    W1_grid_rows = W1_raw[:, 24:].reshape(256, 9, 13).mean(axis=2) # [256, 9]

    W1_display = np.concatenate([W1_player, W1_enemy, W1_grid_rows], axis=1)  # [256, 33]

    return W1_display, W2, W_act


# ── individual plots ──────────────────────────────────────────────────────────

def plot_feature_importance(W1, ax):
    """Bar chart: which of the 33 input groups drive hidden layer 1 the most."""
    importance = np.abs(W1).sum(axis=0)  # [33]
    order = np.argsort(importance)[::-1]

    bar_colors = []
    for i in order:
        if i < 9:    bar_colors.append("#2ecc71")   # player
        elif i < 24: bar_colors.append("#3498db")   # enemy
        else:        bar_colors.append("#e74c3c")   # tile grid

    ax.barh(range(33), importance[order], color=bar_colors, edgecolor="none")
    ax.set_yticks(range(33))
    ax.set_yticklabels([ALL_INPUT_NAMES[i] for i in order], fontsize=7)
    ax.set_xlabel("Σ |weight| into hidden layer 1")
    ax.set_title("Feature Importance", fontweight="bold")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="Player state"),
        mpatches.Patch(color="#3498db", label="Enemy info"),
        mpatches.Patch(color="#e74c3c", label="Tile grid rows"),
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="lower right")


def plot_action_heatmap(W_act, ax):
    """Heatmap: which hidden neurons (x-axis) drive which actions (y-axis)."""
    vmax = np.abs(W_act).max()
    im = ax.imshow(W_act, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(12))
    ax.set_yticklabels(ACTION_NAMES, fontsize=7)
    ax.set_xlabel("Hidden neuron index (0–255)")
    ax.set_title("Action head weights  (blue = +, red = −)", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)


def plot_weight_distributions(W1, W2, W_act, ax):
    """Histogram of |weight| per layer — shows how 'active' each layer is."""
    all_max = max(np.abs(W1).max(), np.abs(W2).max(), np.abs(W_act).max())
    bins = np.linspace(0, all_max, 60)

    ax.hist(np.abs(W1.flatten()),   bins=bins, alpha=0.6, color="#2ecc71",
            label=f"Layer 1   μ={np.abs(W1).mean():.4f}")
    ax.hist(np.abs(W2.flatten()),   bins=bins, alpha=0.6, color="#3498db",
            label=f"Layer 2   μ={np.abs(W2).mean():.4f}")
    ax.hist(np.abs(W_act.flatten()), bins=bins, alpha=0.6, color="#e74c3c",
            label=f"Action head  μ={np.abs(W_act).mean():.4f}")
    ax.set_xlabel("|weight|")
    ax.set_ylabel("Count")
    ax.set_title("Weight magnitude distributions", fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def plot_circuit(W1, W_act, ax, top_n=20):
    """
    Circuit diagram:  33 input groups  →  top_n hidden neurons  →  12 actions
    Blue lines = positive weights, red = negative.  Alpha ∝ magnitude.
    """
    n_in  = len(ALL_INPUT_NAMES)   # 33
    n_out = len(ACTION_NAMES)      # 12

    # Select top hidden neurons by total |output weight| to actions
    neuron_importance = np.abs(W_act).sum(axis=0)  # [256]
    top_idx = np.argsort(neuron_importance)[-top_n:][::-1]

    # Y coordinates (evenly spread in [0, 1])
    max_nodes = max(n_in, top_n, n_out)
    def ys(n): return [i / max(n - 1, 1) for i in range(n)]

    in_y  = ys(n_in)
    hid_y = ys(top_n)
    out_y = ys(n_out)

    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    ax.set_title(f"Circuit: inputs → top {top_n} hidden neurons → actions",
                 fontweight="bold")

    # W1_sub[h, i] = weight from input i to hidden neuron h  (selected)
    W1_sub  = W1[top_idx, :]           # [top_n, 33]
    W_sub   = W_act[:, top_idx]        # [12, top_n]

    # -- input → hidden connections --
    # Use per-group thresholds so weak tile-grid connections aren't drowned
    # out by the much stronger enemy/player weights.
    for h in range(top_n):
        for i in range(n_in):
            w = W1_sub[h, i]
            # group-local threshold: player (0-8), enemy (9-23), grid (24-32)
            if   i < 9:  group = W1_sub[:, :9]
            elif i < 24: group = W1_sub[:, 9:24]
            else:        group = W1_sub[:, 24:]
            thresh = np.percentile(np.abs(group), 75)
            if abs(w) < thresh:
                continue
            a = min(abs(w) / (thresh * 2.0), 0.8)
            ax.plot([0, 1], [in_y[i], hid_y[h]],
                    color=("#4a90d9" if w > 0 else "#e74c3c"),
                    alpha=a, linewidth=0.6, zorder=1)

    # -- hidden → action connections --
    thresh2 = np.percentile(np.abs(W_sub), 65)
    for o in range(n_out):
        for h in range(top_n):
            w = W_sub[o, h]
            if abs(w) < thresh2:
                continue
            a = min(abs(w) / (thresh2 * 2.5), 0.85)
            ax.plot([1, 2], [hid_y[h], out_y[o]],
                    color=("#4a90d9" if w > 0 else "#e74c3c"),
                    alpha=a, linewidth=0.7, zorder=1)

    # -- input nodes --
    for i, (name, y) in enumerate(zip(ALL_INPUT_NAMES, in_y)):
        color = "#2ecc71" if i < 9 else "#3498db" if i < 24 else "#e74c3c"
        ax.plot(0, y, "o", color=color, markersize=5, zorder=3)
        ax.text(-0.03, y, name, ha="right", va="center",
                fontsize=5.5, color="#222222")

    # -- hidden nodes --
    for h, y in enumerate(hid_y):
        ax.plot(1, y, "o", color="#888888", markersize=4, zorder=3)

    # -- action nodes --
    for o, (name, y) in enumerate(zip(ACTION_NAMES, out_y)):
        ax.plot(2, y, "o", color="#27ae60", markersize=7, zorder=3)
        ax.text(2.03, y, name, ha="left", va="center",
                fontsize=6, color="#222222", fontweight="bold")

    # Column labels
    for x, label in [(0, "Inputs (33)"),
                     (1, f"Hidden\n(top {top_n})"),
                     (2, "Actions (12)")]:
        ax.text(x, -0.04, label, ha="center", va="top",
                fontsize=8, fontweight="bold", color="#333333")


# ── main figure builder ───────────────────────────────────────────────────────

def build_figure(model_path, circuit=False):
    W1, W2, W_act = load_weights(model_path)
    label = os.path.basename(model_path)

    if circuit:
        fig, axes = plt.subplots(1, 2, figsize=(22, 13))
        plot_circuit(W1, W_act, axes[0])
        plot_feature_importance(W1, axes[1])
    else:
        fig = plt.figure(figsize=(18, 10))
        gs  = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.32)
        ax_imp  = fig.add_subplot(gs[:, 0])
        ax_act  = fig.add_subplot(gs[0, 1])
        ax_hist = fig.add_subplot(gs[1, 1])

        plot_feature_importance(W1, ax_imp)
        plot_action_heatmap(W_act, ax_act)
        plot_weight_distributions(W1, W2, W_act, ax_hist)

    fig.suptitle(
        f"SpaceHuggers PPO — weight visualisation  |  {label}",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.patch.set_facecolor("#f8f8f8")
    return fig


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default=None,
                    help="Path to checkpoint without .zip extension")
    ap.add_argument("--watch",   type=int, default=0, metavar="SECS",
                    help="Auto-refresh every N seconds (0 = run once)")
    ap.add_argument("--save",    action="store_true",
                    help="Save PNG to weights_vis/ instead of opening a window")
    ap.add_argument("--circuit", action="store_true",
                    help="Circuit diagram style (like spacehuggers_decoded_circuit.png)")
    args = ap.parse_args()

    os.makedirs("weights_vis", exist_ok=True)

    def run_once():
        path = find_model(args.model)
        fig  = build_figure(path, circuit=args.circuit)
        if args.save:
            out = f"weights_vis/weights_{int(time.time())}.png"
            fig.savefig(out, dpi=130, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"Saved → {out}")
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()

    if args.watch:
        print(f"Watch mode — refreshing every {args.watch}s  (Ctrl+C to stop)")
        matplotlib.use("Agg")   # headless for watch+save
        while True:
            try:
                path = find_model(args.model)
                fig  = build_figure(path, circuit=args.circuit)
                out  = f"weights_vis/weights_{int(time.time())}.png"
                fig.savefig(out, dpi=130, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                print(f"[{time.strftime('%H:%M:%S')}] Saved → {out}")
                plt.close(fig)
                time.sleep(args.watch)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(args.watch)
    else:
        run_once()


if __name__ == "__main__":
    main()
