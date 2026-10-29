"""Language x language consistency matrix, KLAR-style: Jaccard similarity of
FAILURE sets between each pair of languages (same items, same model) —
matches the metric used in the KLAR paper's cross-lingual failure clustering.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import NUMERIC_ROOT
ROOT = NUMERIC_ROOT
RESULTS = ROOT / "results"

MKQA_26 = [
    "en", "es", "fr", "de", "it", "pt", "nl", "sv", "no", "da", "fi",
    "pl", "hu", "tr", "ru", "vi", "ms", "ar", "he", "ko", "ja",
    "zh_cn", "zh_hk", "zh_tw", "th", "km",
]

SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"

MODEL_TITLES = {"llama2-7b": "Llama-2-7B", "qwen2.5-7b": "Qwen2.5-7B"}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


for tag, title in MODEL_TITLES.items():
    rows = []
    with open(RESULTS / f"{tag}_results.jsonl", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # per-language set of (source, source_id) that the model got WRONG
    fail_sets = {lang: set() for lang in MKQA_26}
    for r in rows:
        if not r["correct"]:
            fail_sets[r["lang"]].add((r["source"], r["source_id"]))

    n = len(MKQA_26)
    mat = np.zeros((n, n))
    for i, li in enumerate(MKQA_26):
        for j, lj in enumerate(MKQA_26):
            mat[i, j] = jaccard(fail_sets[li], fail_sets[lj])

    off_diag = mat[~np.eye(n, dtype=bool)]
    vmin, vmax = off_diag.min(), off_diag.max()
    mat_masked = np.ma.masked_array(mat, mask=np.eye(n, dtype=bool))

    fig, ax = plt.subplots(figsize=(11, 9.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color=SURFACE)
    im = ax.imshow(mat_masked, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(MKQA_26, rotation=90, fontsize=8, color=SECONDARY_INK)
    ax.set_yticklabels(MKQA_26, fontsize=8, color=SECONDARY_INK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.suptitle(
        f"{title} — cross-lingual failure consistency (Jaccard)",
        x=0.02, y=0.98, ha="left", color=PRIMARY_INK, fontsize=15, fontweight="bold",
    )
    ax.set_title(
        f"Jaccard similarity of incorrect-item sets, diagonal masked, scale {vmin:.2f}-{vmax:.2f}",
        color=MUTED, fontsize=10, loc="left", pad=10,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(colors=MUTED, length=0, labelsize=8)
    cbar.outline.set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path = RESULTS / f"{tag}_consistency_matrix.png"
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out_path}")

    # quick text summary: most / least similar pairs (excluding diagonal)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((mat[i, j], MKQA_26[i], MKQA_26[j]))
    pairs.sort(reverse=True)
    print(f"  Most similar failure patterns: {pairs[0][1]}-{pairs[0][2]}={pairs[0][0]:.2f}, "
          f"{pairs[1][1]}-{pairs[1][2]}={pairs[1][0]:.2f}, {pairs[2][1]}-{pairs[2][2]}={pairs[2][0]:.2f}")
    print(f"  Least similar: {pairs[-1][1]}-{pairs[-1][2]}={pairs[-1][0]:.2f}, "
          f"{pairs[-2][1]}-{pairs[-2][2]}={pairs[-2][0]:.2f}")
