"""Variant of fig:factmap_strict: hexbin of EN z (fact f) vs L z, with two cross-lingual controls:
a same-relation fact and a different-relation fact, both in a control language (default es).
Same data and z-scores as plot_dense_maps.py (dense_export npz).

Usage: python plot_dense_hexbin_controls.py <dense_maps.npz> --fact 6427__capital [--ctrl_lang es] [--out DIR]
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("npz"); p.add_argument("--fact", required=True)
p.add_argument("--langs", default="es,ru,zh,ja"); p.add_argument("--ctrl_lang", default="es")
p.add_argument("--out", default=".")
p.add_argument("--c_same", default=None, help="force the same-relation control fact")
p.add_argument("--no_diff", action="store_true", help="omit the other-relation panel")
a = p.parse_args()
Z = np.load(a.npz, allow_pickle=True)
STEMS = [str(s) for s in Z["stems"]]
get = lambda s, l: Z[f"z__{s}__{l}"].astype(np.float32)
cos = lambda x, y: float(x.ravel() @ y.ravel() / (np.linalg.norm(x) * np.linalg.norm(y)))
rel = lambda s: s.split("__", 1)[1]
names = {"es": "Spanish", "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "en": "English"}

f, cl = a.fact, a.ctrl_lang
en = get(f, "en")
cands = [s for s in STEMS if s != f and f"z__{s}__{cl}" in Z.files]
same = sorted(((cos(en, get(s, cl)), s) for s in cands if rel(s) == rel(f)))
diff = sorted(((cos(en, get(s, cl)), s) for s in cands if rel(s) != rel(f)))
print("same-relation candidates (cos, stem):", [(round(c, 3), s) for c, s in same])
print("diff-relation candidates: median", round(float(np.median([c for c, _ in diff])), 3))
c_same = a.c_same or same[len(same) // 2][1]   # median same-relation fact unless forced
c_diff = diff[len(diff) // 2][1]           # median different-relation fact

panels = [(names[l], get(f, l)) for l in a.langs.split(",")]
panels += [(f"same relation,\nother fact ({names[cl]})", get(c_same, cl))]
if not a.no_diff:
    panels += [(f"other relation\n({names[cl]})", get(c_diff, cl))]
fig, axes = plt.subplots(1, len(panels), figsize=(7.6, 1.75), sharex=True, sharey=True)
x = en.ravel()
for i, (ax, (lab, y)) in enumerate(zip(axes, panels)):
    ax.hexbin(np.clip(x, -6, 6), np.clip(y.ravel(), -6, 6), gridsize=45, bins="log",
              cmap="Blues", mincnt=1, linewidths=0)
    ax.set_title(f"{lab}\ncos {cos(en, y):.2f}", fontsize=5.5, pad=2)
    ax.tick_params(labelsize=5, length=1.5); ax.set_aspect("equal")
axes[0].set_ylabel("z in language $L$", fontsize=6)
fig.text(0.5, 0.01, "z in English (same fact)", ha="center", fontsize=6)
fig.subplots_adjust(left=0.05, right=0.995, top=0.80, bottom=0.17, wspace=0.08)
os.makedirs(a.out, exist_ok=True)
fo = os.path.join(a.out, f"dense_hexbin_controls__{f}.png"); fig.savefig(fo, dpi=300)
print("controls:", c_same, c_diff, "\nsaved", fo)
