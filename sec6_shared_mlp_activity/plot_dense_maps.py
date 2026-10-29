"""
plot_dense_maps.py  (local)
---------------------------
Dense views of FULL z-scored MLP activation maps exported by export_dense_maps.py.

(A) fingerprint: each layer's 14,336 neurons folded into a 112x128 tile (same neuron order
    for every language). Rows = languages + a control row (a different fact in EN).
    The value under each tile is its cosine with the EN tile of the same layer.
(B) hexbin: EN z vs L z over all neurons of all layers, same fact vs control.

Usage: python plot_dense_maps.py <dense_maps.npz> [--fact STEM] [--control STEM] [--layers 17,21,25,29] [--out DIR]
       python plot_dense_maps.py <dense_maps.npz> --list     (cosines per candidate, to pick a typical fact)
"""
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("npz")
p.add_argument("--fact"); p.add_argument("--control")
p.add_argument("--layers", default="17,21,25,29")
p.add_argument("--langs", default="en,es,ru,zh,ja")
p.add_argument("--out", default=".")
p.add_argument("--list", action="store_true")
a = p.parse_args()

Z = np.load(a.npz, allow_pickle=False)
LAYERS = [int(x) for x in Z["layers"]]
STEMS = [str(s) for s in Z["stems"]]
LANGS = a.langs.split(",")
get = lambda s, l: Z[f"z__{s}__{l}"].astype(np.float32)          # [n_layers, d]
cos = lambda x, y: float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))

def mean_cos(s, t=None):
    """mean over layers and non-EN langs of cos(EN of s, L of t or s)."""
    t = t or s
    return np.mean([cos(get(s, "en")[i], get(t, l)[i]) for l in LANGS[1:] for i in range(len(LAYERS))])

if a.list:
    rows = sorted(((mean_cos(s), s) for s in STEMS), reverse=True)
    for c, s in rows: print(f"{c:.3f}  {s}")
    raise SystemExit

fact = a.fact or STEMS[0]
ctrl = a.control or next(s for s in STEMS if s != fact)
show = [LAYERS.index(int(x)) for x in a.layers.split(",")]
D = get(fact, "en").shape[1]; H = 112; W = D // H
names = {"en": "English", "es": "Spanish", "ru": "Russian", "zh": "Chinese", "ja": "Japanese"}

# ---- (A) fingerprint grid
rows = [(names.get(l, l), get(fact, l)) for l in LANGS] + [("control (EN,\nother fact)", get(ctrl, "en"))]
en = get(fact, "en")
fig, axes = plt.subplots(len(rows), len(show), figsize=(3.3, 0.62 * len(rows) + 0.35), squeeze=False)
for r, (lab, m) in enumerate(rows):
    for c, li in enumerate(show):
        ax = axes[r][c]
        ax.imshow(m[li].reshape(H, W), cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_linewidth(0.3)
        if r > 0:
            ax.text(0.97, 0.06, f"{cos(en[li], m[li]):.2f}", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=5, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
        if r == 0: ax.set_title(f"layer {LAYERS[li]}", fontsize=6, pad=2)
        if c == 0: ax.set_ylabel(lab, fontsize=5.5, rotation=0, ha="right", va="center", labelpad=2)
fig.subplots_adjust(left=0.2, right=0.995, top=0.94, bottom=0.01, wspace=0.04, hspace=0.06)
fa = os.path.join(a.out, f"dense_fingerprint__{fact}.png"); fig.savefig(fa, dpi=300); plt.close(fig)

# ---- (B) hexbin, all neurons of all layers
others = LANGS[1:]
fig, axes = plt.subplots(1, len(others) + 1, figsize=(6.8, 1.55), sharex=True, sharey=True)
x = en.ravel()
for ax, (lab, y) in zip(axes, [(names.get(l, l), get(fact, l).ravel()) for l in others] + [("control", get(ctrl, "en").ravel())]):
    ax.hexbin(np.clip(x, -6, 6), np.clip(y, -6, 6), gridsize=45, bins="log", cmap="Blues", mincnt=1, linewidths=0)
    ax.set_title(f"{lab}  (cos {cos(x, y):.2f})", fontsize=6, pad=2)
    ax.tick_params(labelsize=5, length=1.5); ax.set_aspect("equal")
axes[0].set_ylabel("z in language $L$", fontsize=6)
fig.text(0.5, 0.01, "z in English (same fact)", ha="center", fontsize=6)
fig.subplots_adjust(left=0.06, right=0.995, top=0.88, bottom=0.18, wspace=0.08)
fb = os.path.join(a.out, f"dense_hexbin__{fact}.png"); fig.savefig(fb, dpi=300); plt.close(fig)
print("fact", fact, json.loads(str(Z[f"known__{fact}"])), "| control", ctrl, "\nsaved", fa, "\nsaved", fb)
