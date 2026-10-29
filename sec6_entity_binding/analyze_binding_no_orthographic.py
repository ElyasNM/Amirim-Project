"""
Robustness check: exclude triplets where subject_lang is spelled identically
(or near-identically) to subject_en -- e.g. many place names are unchanged
across Latin-script languages ("Amsterdam" in EN/FR/ES). These inflate
cos_translation via shared subword tokens rather than genuine cross-lingual
entity binding. Re-run the by-language comparison on the filtered subset.
"""
import os
import json
import numpy as np
import pandas as pd

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import BINDING_DIR
HERE = str(BINDING_DIR)

with open(os.path.join(HERE, "binding_triplets.json"), encoding="utf-8") as f:
    triplets = json.load(f)

key_to_identical = {}
for t in triplets:
    key = (t["relation"], t["lang"], t["index"])
    same = t["subject_en"].strip().lower() == t["subject_lang"].strip().lower()
    key_to_identical[key] = same

n_identical = sum(key_to_identical.values())
print(f"{n_identical} / {len(triplets)} triplets have identical EN/target spelling ({n_identical/len(triplets):.1%})")

by_lang_counts = {}
for t in triplets:
    same = t["subject_en"].strip().lower() == t["subject_lang"].strip().lower()
    by_lang_counts.setdefault(t["lang"], [0, 0])
    by_lang_counts[t["lang"]][0] += same
    by_lang_counts[t["lang"]][1] += 1
print("Identical-spelling rate by language:")
for lang, (same, total) in sorted(by_lang_counts.items()):
    print(f"  {lang}: {same}/{total} = {same/total:.1%}")

df = pd.read_csv(os.path.join(HERE, "binding_results.csv"))
df["key"] = list(zip(df["relation"], df["lang"], df["index"]))
df["identical_spelling"] = df["key"].map(key_to_identical)
df["diff"] = df["cos_translation"] - df["cos_baseline"]

filtered = df[~df["identical_spelling"]].copy()
print(f"\nRows after filtering: {len(filtered)} / {len(df)}")

by_lang = filtered.groupby(["layer", "lang"])[["cos_translation", "cos_baseline", "diff"]].mean().reset_index()
by_lang.to_csv(os.path.join(HERE, "summary_by_layer_lang_filtered.csv"), index=False)

overall_f = filtered.groupby("layer")[["cos_translation", "cos_baseline", "diff"]].mean()
peak_layer = overall_f["diff"].idxmax()
peak = filtered[filtered["layer"] == peak_layer]
print(f"\n[Filtered, no identical spelling] Peak gap layer: {peak_layer}")
print(f"  mean cos_translation: {peak['cos_translation'].mean():.4f}")
print(f"  mean cos_baseline:    {peak['cos_baseline'].mean():.4f}")
print(f"  mean diff:            {peak['diff'].mean():.4f}")
print(f"  fraction translation > baseline: {(peak['diff']>0).mean():.3f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 6))
for lang, g in by_lang.groupby("lang"):
    ax.plot(g["layer"], g["cos_translation"] - g["cos_baseline"], label=lang, linewidth=1.8)
ax.axhline(0, color="gray", linewidth=1, linestyle="--")
ax.set_xlabel("Layer")
ax.set_ylabel("cos(translation pair) - cos(baseline pair)")
ax.set_title("Entity-binding gap by language (identical-spelling pairs excluded)")
ax.legend(ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figs", "binding_by_lang_filtered.png"), dpi=150)
plt.close(fig)
print("Wrote figs/binding_by_lang_filtered.png")
