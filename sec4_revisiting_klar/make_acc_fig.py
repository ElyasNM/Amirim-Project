"""KLAR accuracy per language (paper: fig:acc) as a two-panel bar plot.

Run:  python sec4_revisiting_klar/make_acc_fig.py
Writes <DATA_ROOT>/results/figs/acc_by_language.png (column width, 300 dpi).
Numbers are copied by hand from the per-language accuracies printed by 01_fix_paper_truncation_bug_impact.py
(Llama-3.1-8B) and run_analyses_01_to_04_any_model.py (OLMo-2-7B); update them here if they change.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import RESULTS_ROOT
OUT = RESULTS_ROOT / "figs" / "acc_by_language.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

LATIN = ["fr", "es", "nl", "ca"]
NON_LATIN = ["uk", "ru", "ja", "zh", "ko"]
ACC = {
    "Llama-3.1-8B": dict(en=79.2, fr=77.0, es=76.3, nl=76.0, ca=73.2,
                         uk=72.7, ru=72.4, ja=66.6, zh=63.7, ko=63.2),
    "OLMo-2-7B": dict(en=78.2, fr=67.6, es=61.9, nl=55.4, ca=50.1,
                      uk=30.8, ru=44.1, ja=23.4, zh=25.6, ko=22.1),
}

# Reference palette, light mode: neutral ink for English, categorical slots 1-2 for the groups.
C_EN, C_LAT, C_NON = "#52514e", "#2a78d6", "#eb6834"
INK, MUTED, GRID, BASE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
GAP = 0.7  # extra space between groups

plt.rcParams.update({
    # Helvetica-style sans, the usual choice for figures in ACL papers.
    # For a Times look instead: "font.family": "serif", "font.serif": ["STIXGeneral"]
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.5,
})

# x positions: en | gap | latin | gap | non-latin
langs = ["en"] + LATIN + NON_LATIN
xs, x = [], 0.0
for i, lang in enumerate(langs):
    if lang in ("fr", "uk"):
        x += GAP
    xs.append(x)
    x += 1.0
colors = [C_EN] + [C_LAT] * len(LATIN) + [C_NON] * len(NON_LATIN)

fig, axes = plt.subplots(2, 1, figsize=(3.0, 3.3), sharex=True)
for ax, (model, acc) in zip(axes, ACC.items()):
    vals = [acc[l] for l in langs]
    ax.bar(xs, vals, width=0.72, color=colors, zorder=3)
    ax.axhline(acc["en"], color=C_EN, ls="--", lw=0.8, zorder=2)
    for xi, v in zip(xs, vals):
        # Value inside the bar top, so it never meets the dashed English line.
        ax.text(xi, v - 2, f"{v:.0f}", ha="center", va="top", fontsize=6, color="white",
                fontweight="bold", zorder=4)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Accuracy (%)", color=INK)
    ax.set_title(model, loc="left", fontsize=8, color=INK, pad=3)
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(axis="y", length=0, colors=MUTED)
    ax.tick_params(axis="x", length=0, colors=INK)

for ax in axes:  # language labels under both panels
    ax.set_xticks(xs)
    ax.set_xticklabels(langs)
    ax.tick_params(axis="x", labelbottom=True)

fig.legend(
    handles=[Patch(color=C_EN, label="English"),
             Patch(color=C_LAT, label="Latin script"),
             Patch(color=C_NON, label="Non-Latin script")],
    loc="upper center", ncol=3, frameon=False, fontsize=7.5,
    handlelength=1.0, columnspacing=1.4, bbox_to_anchor=(0.5, 1.0),
)
fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.8)
fig.savefig(OUT, dpi=300)
print("wrote", OUT)
