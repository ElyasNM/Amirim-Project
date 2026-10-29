"""
analyze_binding.py
-------------------
Reads binding_results.csv (per-triplet, per-layer cosine similarities for
translation pairs vs. same-language baseline pairs) and produces:
  - summary_by_layer.csv       overall + per-language mean cosine per layer
  - summary_overall.csv        single-number summary stats (paired diff, effect size)
  - figs/binding_overall.png   translation vs baseline cosine across layers
  - figs/binding_by_lang.png   same, split by language
"""
import os
import numpy as np
import pandas as pd

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import BINDING_DIR
HERE = str(BINDING_DIR)
df = pd.read_csv(os.path.join(HERE, "binding_results.csv"))
df["diff"] = df["cos_translation"] - df["cos_baseline"]

os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)

# ---- per-layer overall summary ----
overall = df.groupby("layer")[["cos_translation", "cos_baseline", "diff"]].mean()
overall.columns = ["mean_cos_translation", "mean_cos_baseline", "mean_diff"]
overall["n"] = df.groupby("layer").size()
overall.to_csv(os.path.join(HERE, "summary_by_layer.csv"))

# ---- per-layer, per-language summary ----
by_lang = df.groupby(["layer", "lang"])[["cos_translation", "cos_baseline", "diff"]].mean().reset_index()
by_lang.to_csv(os.path.join(HERE, "summary_by_layer_lang.csv"), index=False)

# ---- headline stats ----
peak_layer = overall["mean_diff"].idxmax()
n_triplets = len(df) // df["layer"].nunique()

# paired bootstrap CI on the diff at the peak layer, across triplets
peak_df = df[df["layer"] == peak_layer]
rng = np.random.default_rng(0)
boot = []
vals = peak_df["diff"].to_numpy()
for _ in range(2000):
    sample = rng.choice(vals, size=len(vals), replace=True)
    boot.append(sample.mean())
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

# fraction of triplets where translation-pair cosine beats baseline, at peak layer
frac_translation_wins = (peak_df["diff"] > 0).mean()

with open(os.path.join(HERE, "summary_overall.txt"), "w", encoding="utf-8") as f:
    f.write(f"n_triplets: {n_triplets}\n")
    f.write(f"n_layers: {df['layer'].nunique()}\n")
    f.write(f"languages: {sorted(df['lang'].unique().tolist())}\n")
    f.write(f"relations: {sorted(df['relation'].unique().tolist())}\n\n")
    f.write(f"Layer with largest (translation - baseline) cosine gap: {peak_layer}\n")
    f.write(f"  mean cos_translation: {peak_df['cos_translation'].mean():.4f}\n")
    f.write(f"  mean cos_baseline:    {peak_df['cos_baseline'].mean():.4f}\n")
    f.write(f"  mean diff:            {peak_df['diff'].mean():.4f}  (95% bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}])\n")
    f.write(f"  fraction of triplets where translation pair > baseline pair: {frac_translation_wins:.3f}\n\n")
    f.write("Per-layer overall means:\n")
    f.write(overall.to_string())

print(open(os.path.join(HERE, "summary_overall.txt"), encoding="utf-8").read())

# ---- plots ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(overall.index, overall["mean_cos_translation"], label="translation pair (same entity, cross-lingual)", color="#2b6cb0", linewidth=2)
ax.plot(overall.index, overall["mean_cos_baseline"], label="baseline pair (same language, diff entity)", color="#c05621", linewidth=2)
ax.set_xlabel("Layer")
ax.set_ylabel("Mean cosine similarity")
ax.set_title("Llama-3.1-8B entity-binding: translation pair vs. same-language baseline")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figs", "binding_overall.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 6))
for lang, g in by_lang.groupby("lang"):
    ax.plot(g["layer"], g["cos_translation"] - g["cos_baseline"], label=lang, linewidth=1.8)
ax.axhline(0, color="gray", linewidth=1, linestyle="--")
ax.set_xlabel("Layer")
ax.set_ylabel("cos(translation pair) - cos(baseline pair)")
ax.set_title("Entity-binding gap by language, Llama-3.1-8B")
ax.legend(ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figs", "binding_by_lang.png"), dpi=150)
plt.close(fig)

print("Wrote figs/binding_overall.png and figs/binding_by_lang.png")
