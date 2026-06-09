"""
Fully visualise the weights of the trained SpaceHuggers PPO models.

Loads the policy state_dict straight out of the SB3 .zip (no policy
reconstruction / no env needed), then renders, for each model:
  - a per-layer stats table (shape, params, mean, std, min/max, % near-zero)
  - per-layer weight histograms
  - heatmaps of every 2D (Linear) weight matrix
  - conv-filter visualisations (v2 CNN only)
plus a side-by-side overview comparing the two models.
"""
import zipfile, io, os, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "weight_viz"
os.makedirs(OUT, exist_ok=True)


def load_state_dict(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read("policy.pth")
    buf = io.BytesIO(raw)
    try:
        return torch.load(buf, map_location="cpu")
    except Exception:
        buf.seek(0)
        return torch.load(buf, map_location="cpu", weights_only=False)


def to_np(sd):
    return {k: v.detach().cpu().numpy() for k, v in sd.items()
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point}


# ── locate models ─────────────────────────────────────────────────────────────
v2_glob = sorted(glob.glob("game_models_v2_pre_perception_*/ppo_sh_bestlevel.zip"))
MODELS = {}
if os.path.exists("game_models/ppo_sh_latest.zip"):
    MODELS["v1 (17.5M, flat MLP)"] = "game_models/ppo_sh_latest.zip"
if v2_glob:
    MODELS["v2 (CNN, ~420k)"] = v2_glob[-1]

nets = {name: to_np(load_state_dict(p)) for name, p in MODELS.items()}

# ── per-layer stats table (text) ───────────────────────────────────────────────
print("=" * 100)
for name, w in nets.items():
    total = sum(a.size for a in w.values())
    print(f"\n### {name}  —  {total:,} params across {len(w)} tensors")
    print(f"{'layer':<46}{'shape':<20}{'mean':>9}{'std':>9}{'min':>8}{'max':>8}{'%~0':>7}")
    for k, a in w.items():
        nz = 100.0 * np.mean(np.abs(a) < 1e-2)
        print(f"{k:<46}{str(tuple(a.shape)):<20}{a.mean():>9.4f}{a.std():>9.4f}"
              f"{a.min():>8.3f}{a.max():>8.3f}{nz:>6.0f}%")
print("=" * 100)


# ── helper: grid of histograms ─────────────────────────────────────────────────
def fig_histograms(name, w, fname):
    items = list(w.items())
    n = len(items); cols = 4; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.3))
    axes = np.atleast_1d(axes).ravel()
    for ax, (k, a) in zip(axes, items):
        ax.hist(a.ravel(), bins=80, color="#3b82c4", log=True)
        ax.set_title(k.replace("features_extractor.", "fe."), fontsize=6)
        ax.tick_params(labelsize=5)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.suptitle(f"{name} — weight distributions (log y)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(fname, dpi=110); plt.close(fig)


# ── helper: heatmaps of 2D weight matrices ─────────────────────────────────────
def fig_heatmaps(name, w, fname):
    items = [(k, a) for k, a in w.items() if a.ndim == 2]
    if not items:
        return None
    n = len(items); cols = 3; rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.0))
    axes = np.atleast_1d(axes).ravel()
    for ax, (k, a) in zip(axes, items):
        lim = np.percentile(np.abs(a), 99) or 1e-6
        im = ax.imshow(a, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"{k.replace('features_extractor.','fe.')}\n{a.shape}", fontsize=6)
        ax.tick_params(labelsize=5)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.suptitle(f"{name} — weight-matrix heatmaps (RdBu, ±99th pct)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(fname, dpi=110); plt.close(fig)
    return fname


# ── helper: conv-filter visualisation (4D tensors) ─────────────────────────────
def fig_convs(name, w, fname):
    convs = [(k, a) for k, a in w.items() if a.ndim == 4]
    if not convs:
        return None
    fig, axes = plt.subplots(len(convs), 1, figsize=(10, 3.0 * len(convs)))
    axes = np.atleast_1d(axes).ravel()
    for ax, (k, a) in zip(axes, convs):
        O, I, H, W = a.shape           # flatten to (O, I*H*W) heatmap
        mat = a.reshape(O, I * H * W)
        lim = np.percentile(np.abs(mat), 99) or 1e-6
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"{k.replace('features_extractor.','fe.')}  conv {a.shape} "
                     f"-> ({O}, {I*H*W})", fontsize=8)
        ax.set_xlabel("in_ch x kH x kW", fontsize=7); ax.set_ylabel("out filter", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.02)
    fig.suptitle(f"{name} — conv filters", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(fname, dpi=110); plt.close(fig)
    return fname


# ── overview: side-by-side comparison ──────────────────────────────────────────
def fig_overview(nets, fname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    # all-weights distribution overlay
    for name, w in nets.items():
        allw = np.concatenate([a.ravel() for a in w.values()])
        axes[0].hist(allw, bins=120, histtype="step", log=True, label=name, linewidth=1.3)
    axes[0].set_title("All weights — distribution (log y)"); axes[0].legend(fontsize=8)
    axes[0].set_xlabel("weight value")
    # per-layer std
    for name, w in nets.items():
        stds = [a.std() for a in w.values()]
        axes[1].plot(range(len(stds)), stds, marker="o", ms=3, label=name)
    axes[1].set_title("Per-layer weight std (depth ->)"); axes[1].legend(fontsize=8)
    axes[1].set_xlabel("layer index"); axes[1].set_ylabel("std")
    fig.tight_layout(); fig.savefig(fname, dpi=120); plt.close(fig)
    return fname


made = []
made.append(fig_overview(nets, f"{OUT}/00_overview.png"))
for name, w in nets.items():
    tag = "v1" if name.startswith("v1") else "v2"
    fig_histograms(name, w, f"{OUT}/{tag}_1_histograms.png"); made.append(f"{OUT}/{tag}_1_histograms.png")
    h = fig_heatmaps(name, w, f"{OUT}/{tag}_2_heatmaps.png")
    if h: made.append(h)
    c = fig_convs(name, w, f"{OUT}/{tag}_3_convfilters.png")
    if c: made.append(c)

print("\nWROTE:")
for m in made:
    print(" ", m)
