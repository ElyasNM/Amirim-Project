"""Peak-layer vs layer-average cosine_z per language pair (does the layer average hide a higher per-pair peak?).

Reads <MLP_ROOT>/results/<model>/pairwise_transfer.csv (bucket=all, last_token) and, per
language pair, compares the mean over layers with the maximum over layers (aligned and control).
Local only; no cluster needed. Exact per-fact maxima would need the cluster activation maps,
since per_fact_similarity.csv.gz is already averaged over layers.

Usage: python -I peak_layer_stats.py
"""
from pathlib import Path

import pandas as pd

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import MLP_ROOT
RESULTS = MLP_ROOT / "results"
MODELS = ["meta-llama_Llama-3.1-8B", "Qwen_Qwen2.5-7B"]


def pair_stats(model):
    d = pd.read_csv(RESULTS / model / "pairwise_transfer.csv")
    d = d[(d.metric == "cosine_z") & (d.bucket == "all") & d.scope.str.startswith("layer")].copy()
    d["layer"] = d.scope.str[5:].astype(int)
    rows = []
    for (l1, l2), x in d.groupby(["lang1", "lang2"]):
        x = x.sort_values("layer")
        peak = x.loc[x.mean_aligned.idxmax()]
        rows.append(dict(
            pair=f"{l1}-{l2}",
            mean_aligned=x.mean_aligned.mean(), mean_control=x.mean_control.mean(),
            max_aligned=x.mean_aligned.max(), peak_layer=int(peak.layer),
            control_at_peak=peak.mean_control, max_control=x.mean_control.max(),
            last_layer_aligned=x.mean_aligned.iloc[-1],
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for model in MODELS:
        r = pair_stats(model)
        print(f"{model}: {len(r)} pairs")
        print(r.drop(columns="pair").mean().round(3).to_string())
        print("peak layer counts:", r.peak_layer.value_counts().sort_index().to_dict())
        print("Spearman(mean, max):", round(r.mean_aligned.corr(r.max_aligned, method="spearman"), 3))
        print()
