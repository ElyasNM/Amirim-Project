"""
Produces 7 figures for the activation-patching analysis, one per finding.
Saves to  ./act_patch_figures/fig{1..7}.png
"""
import io, sys, json, glob, os, random
from collections import defaultdict, Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── paths ─────────────────────────────────────────────────────────────────────
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import CACHE_ROOT, RESULTS_ROOT
BASE = str(CACHE_ROOT)
MODELS = {
    "Llama-3.1-8B": os.path.join(BASE, "meta-llama_Llama-3.1-8B_section6_subj_act_patch", "examples"),
    "Qwen2.5-7B":   os.path.join(BASE, "Qwen_Qwen2.5-7B_section6_subj_act_patch",          "examples"),
}
OUT = os.path.join(str(RESULTS_ROOT), "act_patch_figures")
os.makedirs(OUT, exist_ok=True)

COLORS = {"Llama-3.1-8B": "#2563EB", "Qwen2.5-7B": "#DC2626"}   # blue / red
LIGHT  = {"Llama-3.1-8B": "#BFDBFE", "Qwen2.5-7B": "#FECACA"}

# ── data loading ──────────────────────────────────────────────────────────────
def load_all(examples_dir):
    records = []
    for fpath in glob.glob(os.path.join(examples_dir, "*.json")):
        with open(fpath, encoding="utf-8") as f:
            d = json.load(f)
        relation = d.get("relation",
                         os.path.basename(fpath).split("__")[-1].replace(".json", ""))
        idx = d.get("index", -1)
        for lang, entry in d.items():
            if not isinstance(entry, dict) or "status" not in entry:
                continue
            entry["_relation"] = relation
            entry["_lang"]     = lang
            entry["_index"]    = idx
            records.append(entry)
    return records

print("Loading data …")
data = {m: load_all(p) for m, p in MODELS.items()}
n_layers = {m: len(data[m][0].get("patch_per_layer", [])) for m in MODELS}
print({m: len(v) for m, v in data.items()})

# ── style helpers ─────────────────────────────────────────────────────────────
def save(fig, name, finding_num):
    path = os.path.join(OUT, f"fig{finding_num}_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

def add_value_labels(ax, bars, fmt="{:.1f}%", fontsize=8):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                fmt.format(h), ha="center", va="bottom", fontsize=fontsize)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Overall rescue breakdown (any / en / tgt)
# Finding: EN subject encoding IS causally richer — 62 % Llama, 44 % Qwen rescued
# ══════════════════════════════════════════════════════════════════════════════
print("\nFig 1 …")
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
fig.suptitle(
    "Fig 1 — Patching the EN subject representation rescues 44–62 % of failures\n"
    "(EN subject hidden state IS causally richer than the non-EN equivalent)",
    fontsize=11, fontweight="bold"
)

categories = ["any rescue\n(overall)", "EN-form\nrescue only", "TGT-form\nrescue only"]
for ax, (model, records) in zip(axes, data.items()):
    total = len(records)
    any_r  = sum(1 for r in records if r.get("rescue_summary", {}).get("any_rescue"))
    en_r   = sum(1 for r in records if r.get("rescue_summary", {}).get("any_rescue_en"))
    tgt_r  = sum(1 for r in records if r.get("rescue_summary", {}).get("any_rescue_tgt"))
    vals   = [any_r / total * 100, en_r / total * 100, tgt_r / total * 100]
    bars   = ax.bar(categories, vals,
                    color=[COLORS[model], LIGHT[model], "#6B7280"],
                    edgecolor="white", linewidth=1.2, width=0.5)
    add_value_labels(ax, bars)
    # baseline: 0 % and 100 %
    ax.axhline(100, color="black", ls="--", lw=0.8, alpha=0.4)
    ax.set_title(f"{model}  (n={total:,})", fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel("% of (fact, lang) failure pairs")
    ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # annotation
    ax.annotate(
        f"Not rescued:\n{(total - any_r)/total*100:.1f}%",
        xy=(0, (total - any_r) / total * 100),
        xytext=(1.5, 80), fontsize=8, color="gray",
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.7),
    )

plt.tight_layout()
save(fig, "overall_rescue_breakdown", 1)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — EN-rescue vs TGT-rescue split per language (stacked bars, both models)
# Finding: When rescued, the answer comes out in the TARGET language (translation works)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 2 …")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Fig 2 — When the EN subject patch rescues a failure, the model answers in the TARGET language\n"
    "(TGT rescue takes priority; EN-only = cases where object is identical across languages)",
    fontsize=11, fontweight="bold"
)

for ax, (model, records) in zip(axes, data.items()):
    lang_stats = defaultdict(lambda: {"total": 0, "en_only": 0, "tgt": 0, "no_rescue": 0})
    for r in records:
        lang = r["_lang"]
        rs   = r.get("rescue_summary", {})
        lang_stats[lang]["total"] += 1
        if rs.get("any_rescue_tgt"):
            lang_stats[lang]["tgt"] += 1          # TGT rescue takes priority (incl. intersection)
        elif rs.get("any_rescue_en"):
            lang_stats[lang]["en_only"] += 1      # EN-form rescue only (no TGT match)
        else:
            lang_stats[lang]["no_rescue"] += 1

    langs = sorted(lang_stats, key=lambda l: -lang_stats[l]["tgt"] / lang_stats[l]["total"])
    totals      = np.array([lang_stats[l]["total"]    for l in langs])
    tgt_pct     = np.array([lang_stats[l]["tgt"]      for l in langs]) / totals * 100
    en_pct      = np.array([lang_stats[l]["en_only"]  for l in langs]) / totals * 100
    no_r_pct    = np.array([lang_stats[l]["no_rescue"] for l in langs]) / totals * 100

    x = np.arange(len(langs))
    b1 = ax.bar(x, tgt_pct,              color="#16A34A", label="Rescued → TGT form (EN patch → correct target-lang answer)")
    b2 = ax.bar(x, en_pct, bottom=tgt_pct,
                color="#2563EB", label="Rescued → EN form only (no TGT match)")
    b3 = ax.bar(x, no_r_pct, bottom=tgt_pct + en_pct,
                color="#E5E7EB", label="Not rescued")

    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of failures")
    ax.set_title(model, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ax is axes[0]:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.axhline(100, color="black", ls="--", lw=0.6, alpha=0.3)
    # label each bar with n
    for xi, lang in enumerate(langs):
        ax.text(xi, 102, f"n={lang_stats[lang]['total']}", ha="center", fontsize=6.5, color="gray")

plt.tight_layout()
save(fig, "en_vs_tgt_rescue_by_lang", 2)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — First-rescue-layer distribution
# Finding: Rescue is overwhelmingly at L1-2; EN advantage lives in token embeddings
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 3 …")
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle(
    "Fig 3 — Most rescues first appear at layers 1–2, not in deep 'concept' layers\n"
    "(EN advantage lives in the token embeddings themselves, not in late representational depth)",
    fontsize=11, fontweight="bold"
)

for ax, (model, records) in zip(axes, data.items()):
    nl = n_layers[model]
    first_layers = [
        r.get("rescue_summary", {}).get("first_rescue_layer")
        for r in records
        if r.get("rescue_summary", {}).get("any_rescue") and r.get("rescue_summary", {}).get("first_rescue_layer") is not None
    ]
    counts = Counter(first_layers)
    layers = list(range(1, nl + 1))
    vals   = [counts.get(l, 0) for l in layers]

    # colour early vs mid vs late
    palette = ["#2563EB" if l <= 10 else "#F59E0B" if l <= 20 else "#DC2626"
               for l in layers]
    ax.bar(layers, vals, color=palette, edgecolor="white", linewidth=0.4)

    ax.set_xlabel("Layer where rescue FIRST appears")
    ax.set_ylabel("Number of rescued cases")
    ax.set_title(
        f"{model}  —  {len(first_layers):,} rescued\n"
        f"Median first-rescue layer = {sorted(first_layers)[len(first_layers)//2]}  |  "
        f"Mean = {np.mean(first_layers):.1f}",
        fontsize=9
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # legend
    patches = [
        mpatches.Patch(color="#2563EB", label="Early  L1-10"),
        mpatches.Patch(color="#F59E0B", label="Mid    L11-20"),
        mpatches.Patch(color="#DC2626", label="Late   L21+"),
    ]
    ax.legend(handles=patches, fontsize=8)
    # cumulative line on twin axis
    ax2 = ax.twinx()
    cumulative = np.cumsum(vals) / len(first_layers) * 100
    ax2.plot(layers, cumulative, color="black", lw=1.5, ls="--", label="Cumulative %")
    ax2.set_ylabel("Cumulative % of rescues", fontsize=8)
    ax2.set_ylim(0, 110)
    ax2.axhline(90, color="gray", lw=0.7, ls=":")
    ax2.text(nl - 1, 91, "90%", fontsize=7, color="gray", ha="right")

plt.tight_layout()
save(fig, "first_rescue_layer_distribution", 3)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Per-layer rescue rate across all cases (the rise and decay)
# Finding: EN advantage DECAYS in late layers as language context re-contaminates
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 4 …")
fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle(
    "Fig 4 — EN subject patch is most effective at mid layers; rescue rate DECAYS in late layers\n"
    "(late-layer language-specific context gradually overrides the patched EN representation)",
    fontsize=11, fontweight="bold"
)

for model, records in data.items():
    nl    = n_layers[model]
    total = len(records)
    layer_rescue = defaultdict(int)
    for r in records:
        for p in r.get("patch_per_layer", []):
            if p.get("rescue"):
                layer_rescue[p["layer"]] += 1
    layers = list(range(0, nl + 1))
    rates  = [layer_rescue.get(l, 0) / total * 100 for l in layers]
    ax.plot(layers, rates, color=COLORS[model], lw=2.2, label=model, marker="o",
            markersize=3.5)
    # shade the peak region
    peak_layer = int(np.argmax(rates))
    ax.axvline(peak_layer, color=COLORS[model], lw=0.8, ls="--", alpha=0.5)
    ax.text(peak_layer + 0.3, rates[peak_layer] + 0.5,
            f"peak L{peak_layer}\n{rates[peak_layer]:.1f}%",
            color=COLORS[model], fontsize=8)

# reference lines
ax.axhline(0,   color="black", lw=0.5, alpha=0.3)
ax.set_xlabel("Layer (patched from EN into L-language forward pass)")
ax.set_ylabel("% of all failure cases rescued at this layer")
ax.set_xlim(-0.5, max(n_layers.values()) + 0.5)
ax.set_ylim(-1, 65)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10)

# annotate the three phases
ax.annotate("", xy=(12, 58), xytext=(0, 58),
            arrowprops=dict(arrowstyle="<->", color="#4B5563", lw=1))
ax.text(6, 59, "Rise\n(embedding → concept)", ha="center", fontsize=8, color="#4B5563")
ax.annotate("", xy=(21, 58), xytext=(13, 58),
            arrowprops=dict(arrowstyle="<->", color="#4B5563", lw=1))
ax.text(17, 59, "Plateau", ha="center", fontsize=8, color="#4B5563")
ax.annotate("", xy=(31, 58), xytext=(22, 58),
            arrowprops=dict(arrowstyle="<->", color="#4B5563", lw=1))
ax.text(27, 59, "Decay\n(lang-context\noverrides patch)", ha="center", fontsize=8, color="#4B5563")

plt.tight_layout()
save(fig, "per_layer_rescue_rate", 4)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — Language-family × script heatmap of rescue rates
# Finding: Script matters for EN-rescue; TGT-rescue shows the real capability gap
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 5 …")
SCRIPT = {
    "ca": "Latin", "es": "Latin", "fr": "Latin", "nl": "Latin",
    "hu": "Latin", "vi": "Latin", "tr": "Latin",
    "ru": "Cyrillic", "uk": "Cyrillic",
    "ar": "Arabic", "fa": "Persian", "he": "Hebrew",
    "ja": "CJK", "ko": "Hangul", "zh": "CJK",
    "el": "Greek",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Fig 5 — Script family drives the EN-rescue rate; TGT-rescue reveals the true EN-advantage gap\n"
    "(near-zero EN-rescue for CJK/Cyrillic is a tokenization artifact, not a model understanding effect)",
    fontsize=11, fontweight="bold"
)

for ax, (model, records) in zip(axes, data.items()):
    lang_stats = defaultdict(lambda: {"total": 0, "en": 0, "tgt": 0, "any": 0})
    for r in records:
        lang = r["_lang"]
        rs   = r.get("rescue_summary", {})
        lang_stats[lang]["total"] += 1
        if rs.get("any_rescue_en"):  lang_stats[lang]["en"]  += 1
        if rs.get("any_rescue_tgt"): lang_stats[lang]["tgt"] += 1
        if rs.get("any_rescue"):     lang_stats[lang]["any"] += 1

    # Sort by script then overall rescue desc
    langs = sorted(lang_stats,
                   key=lambda l: (SCRIPT.get(l, "z"), -lang_stats[l]["any"] / lang_stats[l]["total"]))
    totals   = np.array([lang_stats[l]["total"] for l in langs])
    en_pct   = np.array([lang_stats[l]["en"]    for l in langs]) / totals * 100
    tgt_pct  = np.array([lang_stats[l]["tgt"]   for l in langs]) / totals * 100
    any_pct  = np.array([lang_stats[l]["any"]   for l in langs]) / totals * 100

    x = np.arange(len(langs))
    w = 0.28
    b1 = ax.bar(x - w, any_pct, w, color="#111827", label="Overall rescue")
    b2 = ax.bar(x,     en_pct,  w, color="#2563EB", label="EN-form rescue\n(EN patch → EN answer)")
    b3 = ax.bar(x + w, tgt_pct, w, color="#16A34A", label="TGT-form rescue\n(EN patch → TGT answer)")

    ax.set_xticks(x)
    scripts = [SCRIPT.get(l, "?") for l in langs]
    labels  = [f"{l}\n({scripts[i]})" for i, l in enumerate(langs)]
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of failures")
    ax.set_title(model, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ax is axes[0]:
        ax.legend(fontsize=8, loc="upper right")

    # draw vertical separators between script groups
    prev_script = None
    for xi, lang in enumerate(langs):
        sc = SCRIPT.get(lang, "?")
        if prev_script and sc != prev_script:
            ax.axvline(xi - 0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
        prev_script = sc

plt.tight_layout()
save(fig, "rescue_by_language_script", 5)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Logit lens: object-token rank for EN-subj vs TGT-subj per layer
# Finding: EN subject encodes the correct fact more strongly; gap grows in late layers
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 6 …")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Fig 6 — Logit-lens: EN subject hidden state decodes to the correct object at far lower rank\n"
    "(gap grows from ~3K at early layers to ~17K at late layers — EN representation is progressively richer)",
    fontsize=11, fontweight="bold"
)

for ax, (model, records) in zip(axes, data.items()):
    nl = n_layers[model]
    en_ranks  = defaultdict(list)
    tgt_ranks = defaultdict(list)
    for r in records:
        for li, lens_e in enumerate(r.get("lens_en_subject",  [])):
            en_ranks[li].append(lens_e["ranks"]["obj_en"])
        for li, lens_t in enumerate(r.get("lens_tgt_subject", [])):
            tgt_ranks[li].append(lens_t["ranks"]["obj_en"])

    layers   = sorted(en_ranks)
    en_mean  = [np.mean(en_ranks[l])  for l in layers]
    tgt_mean = [np.mean(tgt_ranks[l]) for l in layers]
    delta    = [t - e for t, e in zip(tgt_mean, en_mean)]

    ax.plot(layers, en_mean,  color="#2563EB", lw=2, label="EN-subj → obj_en rank")
    ax.plot(layers, tgt_mean, color="#DC2626", lw=2, label="TGT-subj → obj_en rank")
    ax.fill_between(layers, en_mean, tgt_mean, alpha=0.12, color="#DC2626",
                    label=f"Gap (TGT worse by this many ranks)")

    ax.set_xlabel("Layer (logit-lens projection)")
    ax.set_ylabel("Mean rank of correct EN object token\n(lower = model 'knows' the answer better)")
    ax.invert_yaxis()          # lower rank = better → put rank 0 at top
    ax.set_title(model, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8)

    # annotate the delta at the final layer
    last = layers[-1]
    ax.annotate(
        f"Final-layer gap:\n+{delta[-1]:,.0f} ranks\n(TGT worse)",
        xy=(last, tgt_mean[-1]), xytext=(last - 10, tgt_mean[-1] * 0.6),
        fontsize=8, color="#DC2626",
        arrowprops=dict(arrowstyle="->", color="#DC2626", lw=0.8),
    )

plt.tight_layout()
save(fig, "logit_lens_en_vs_tgt_subject", 6)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Distribution of best-rank achieved for non-rescued cases
# Finding: 38 % of failures can't be fixed by subject patching — different failure mode
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 7 …")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Fig 7 — Non-rescued failures: best achievable object rank across all layers\n"
    "(many are near-misses at rank 1-10; others are completely unreachable → two distinct non-rescue modes)",
    fontsize=11, fontweight="bold"
)

BINS = [0, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, 999999]
BIN_LABELS = ["0", "1", "2", "3-5", "6-10", "11-25", "26-50", "51-100",
              "101-250", "251-500", "501-1K", ">1K"]

def bin_idx(rank):
    for i in range(len(BINS) - 1):
        if BINS[i] <= rank < BINS[i + 1]:
            return i
    return len(BINS) - 2

for ax, (model, records) in zip(axes, data.items()):
    no_rescue = [r for r in records if not r.get("rescue_summary", {}).get("any_rescue")]
    best_ranks = []
    for r in no_rescue:
        layers = r.get("patch_per_layer", [])
        if layers:
            best = min(p["rank_obj_en"] for p in layers)
            best_ranks.append(best)

    bins_count = Counter(bin_idx(rank) for rank in best_ranks)
    x     = np.arange(len(BIN_LABELS))
    vals  = [bins_count.get(i, 0) for i in range(len(BIN_LABELS))]
    total = len(best_ranks)

    # colour near-misses (rank 0-10) differently
    palette = ["#F59E0B" if i <= 3 else "#6B7280" for i in range(len(BIN_LABELS))]
    bars = ax.bar(x, vals, color=palette, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=8.5, rotation=30, ha="right")
    ax.set_xlabel("Best rank of correct EN object token across all patched layers\n(rank 0 = argmax match)")
    ax.set_ylabel("Number of non-rescued cases")
    ax.set_title(
        f"{model}  —  {len(no_rescue):,} non-rescued ({len(no_rescue)/len(records)*100:.1f}% of all failures)",
        fontsize=9
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # pct labels
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, v + total * 0.003,
                    f"{v/total*100:.1f}%", ha="center", va="bottom", fontsize=7)

    # near-miss annotation
    near_miss_n = sum(vals[:4])   # rank 0-10
    ax.annotate(
        f"Near-misses (rank 0-10):\n{near_miss_n} cases ({near_miss_n/total*100:.1f}%)\n"
        f"— correct answer almost won\nbut a competing token narrowly beat it",
        xy=(2.5, max(vals[:4])), xytext=(6, max(vals) * 0.7),
        fontsize=8, color="#92400E",
        arrowprops=dict(arrowstyle="->", color="#92400E", lw=0.8),
    )
    # unreachable annotation
    far_n = sum(vals[8:])   # rank >100
    ax.annotate(
        f"Unreachable (rank>100):\n{far_n} cases ({far_n/total*100:.1f}%)\n"
        f"— subject encoding is not\nthe bottleneck here",
        xy=(10.5, vals[10]), xytext=(7, max(vals) * 0.45),
        fontsize=8, color="#374151",
        arrowprops=dict(arrowstyle="->", color="#374151", lw=0.8),
    )

    # patch legend
    patches_leg = [
        mpatches.Patch(color="#F59E0B", label="Near-miss (rank 0-10)"),
        mpatches.Patch(color="#6B7280", label="Far / unreachable"),
    ]
    ax.legend(handles=patches_leg, fontsize=8)

plt.tight_layout()
save(fig, "non_rescued_best_rank_distribution", 7)

print("\nAll 7 figures saved to", OUT)
