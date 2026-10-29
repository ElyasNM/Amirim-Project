"""Paper fig:rescue: exclusive rescue breakdown by answer form (numbers from rescue_form_breakdown.py)."""
import os
import matplotlib.pyplot as plt

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import RESULTS_ROOT
OUT = os.path.join(str(RESULTS_ROOT), "figs", "rescue_breakdown.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
models = ["Llama-3.1-8B\n($n$=3,744)", "Qwen2.5-7B\n($n$=9,522)"]
parts = [  # (label, values per model, color)
    ("target form only", [40.0, 28.4], "#1f4e8c"),
    ("both forms",       [16.1,  8.7], "#6a9fd8"),
    ("EN form only",     [ 6.0,  7.0], "#c3d9f0"),
    ("not rescued",      [38.0, 55.9], "#e6e6e6"),
]
fig, ax = plt.subplots(figsize=(3.3, 1.55))
left = [0.0, 0.0]
for label, vals, color in parts:
    bars = ax.barh(models, vals, left=left, color=color, edgecolor="white", linewidth=0.6, label=label, height=0.62)
    for b, v, l in zip(bars, vals, left):
        if v >= 6:
            ax.text(l + v / 2, b.get_y() + b.get_height() / 2, f"{v:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if color in ("#1f4e8c", "#6a9fd8") else "black")
    left = [l + v for l, v in zip(left, vals)]
ax.set_xlim(0, 100); ax.invert_yaxis()
ax.set_xlabel("% of failures", fontsize=7)
ax.tick_params(labelsize=7, length=2)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(ncol=4, fontsize=5.6, frameon=False, loc="lower center", bbox_to_anchor=(0.42, 1.0),
          handlelength=1.0, columnspacing=0.8, handletextpad=0.3)
fig.tight_layout(pad=0.2)
fig.savefig(OUT, dpi=300)
print("saved", OUT)
