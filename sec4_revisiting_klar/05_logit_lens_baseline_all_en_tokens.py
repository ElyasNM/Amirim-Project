"""
05_logit_lens_baseline_all_en_tokens.py
----------------------------------------
Baseline test for the "model thinks in English" hypothesis.

The paper (Section 6) shows the correct EN token rising in rank through
middle layers and interprets this as evidence the model encodes knowledge
in an English-like space.  But they provide NO baseline: maybe all English
tokens rise, or all tokens in general.

This script loads baseline results from cacher_logitlens_baseline.py which
tracks 4 categories of tokens through logit lens for each failure case:

  1. correct_object  — correct answer in the target language
  2. correct_en      — correct answer in English
  3. random_en       — 10 random EN answer tokens from other facts (same relation)
  4. random_vocab    — 10 random vocab tokens (ASCII alpha, 3+ chars)

If the EN rise is specific to the correct answer:
  correct_en << random_en ≈ random_vocab  (only correct EN drops to low rank)

If it's an English-general phenomenon:
  correct_en ≈ random_en << random_vocab  (all EN answers rise, not just correct)

If it's a general layer phenomenon (not EN-specific):
  correct_en ≈ random_en ≈ random_vocab   (everything rises — no EN signal)

Produces:
  - figures/baseline_logit_lens.png        — 4-line comparison plot
  - figures/baseline_logit_lens_ratio.png  — correct_en rank / random_en rank
  - figures/baseline_logit_lens_median.png — median rank trajectories
  - stdout: key statistics and interpretation
"""

import os, sys, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from load_cache_and_match import NUM_LAYERS, OUT_DIR, MODEL_TAG, SHORT_NAME
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load baseline data ─────────────────────────────────────────────────────
from paths import DATA_ROOT
BASE_DIR = str(DATA_ROOT)
BASELINE_DIR = os.path.join(BASE_DIR, "cache", f"{MODEL_TAG}_baseline", "examples")

print(f"Loading baseline data from {BASELINE_DIR} ...")

cor_obj_all = []   # correct object (target-lang) ranks per layer
cor_en_all  = []   # correct EN answer ranks per layer
rand_en_all = []   # mean of 10 random EN answers, per layer
rand_voc_all = []  # mean of 10 random vocab tokens, per layer

n_loaded = 0
n_skipped = 0

for path in sorted(glob.glob(os.path.join(BASELINE_DIR, "*.json"))):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for lang_key, rec in data.items():
        if lang_key in ("index", "relation") or not isinstance(rec, dict):
            continue

        cor = rec.get("correct_object_ranks", [])
        enr = rec.get("correct_en_ranks", [])
        ren = rec.get("random_en_answer_ranks", [])
        rvo = rec.get("random_vocab_ranks", [])

        if len(cor) != NUM_LAYERS or len(enr) != NUM_LAYERS:
            n_skipped += 1
            continue
        if not ren or not rvo:
            n_skipped += 1
            continue

        # Average across the 10 random EN answer tokens per layer
        ren_arr = np.array([r for r in ren if len(r) == NUM_LAYERS])
        rvo_arr = np.array([r for r in rvo if len(r) == NUM_LAYERS])
        if len(ren_arr) == 0 or len(rvo_arr) == 0:
            n_skipped += 1
            continue

        cor_obj_all.append(cor)
        cor_en_all.append(enr)
        rand_en_all.append(ren_arr.mean(axis=0).tolist())
        rand_voc_all.append(rvo_arr.mean(axis=0).tolist())
        n_loaded += 1

print(f"  Loaded: {n_loaded} language-example pairs")
print(f"  Skipped: {n_skipped}")

if n_loaded == 0:
    print(f"\nNo baseline data found in {BASELINE_DIR}")
    print("Run klar/cacher_logitlens_baseline.py first to generate baseline cache.")
    sys.exit(0)

cor_obj = np.array(cor_obj_all)
cor_en  = np.array(cor_en_all)
rand_en = np.array(rand_en_all)
rand_voc = np.array(rand_voc_all)

# ── Mean trajectories ──────────────────────────────────────────────────────
mean_cor_obj = cor_obj.mean(axis=0)
mean_cor_en  = cor_en.mean(axis=0)
mean_rand_en = rand_en.mean(axis=0)
mean_rand_voc = rand_voc.mean(axis=0)

# Also compute medians (more robust to outliers with 128k vocab)
med_cor_obj = np.median(cor_obj, axis=0)
med_cor_en  = np.median(cor_en, axis=0)
med_rand_en = np.median(rand_en, axis=0)
med_rand_voc = np.median(rand_voc, axis=0)

# ── Print layer-by-layer ──────────────────────────────────────────────────
layers = np.arange(1, NUM_LAYERS + 1)

print(f"\n{'='*80}")
print(f"MEAN RANKS BY LAYER (n={n_loaded} failure cases)")
print(f"{'='*80}")
print(f"{'Layer':<6} {'Target':>10} {'Correct EN':>12} {'Random EN':>12} {'Random Vocab':>14}")
print("-" * 58)
for i in range(NUM_LAYERS):
    print(f"{i+1:<6} {mean_cor_obj[i]:>10.0f} {mean_cor_en[i]:>12.0f} "
          f"{mean_rand_en[i]:>12.0f} {mean_rand_voc[i]:>14.0f}")

# ── Key statistics ────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("KEY STATISTICS")
print(f"{'='*80}")

# Final layer
print(f"\nFinal layer (32) mean ranks:")
print(f"  Correct target-lang:  {mean_cor_obj[-1]:>10.1f}  (median: {med_cor_obj[-1]:.0f})")
print(f"  Correct EN answer:    {mean_cor_en[-1]:>10.1f}  (median: {med_cor_en[-1]:.0f})")
print(f"  Random EN answers:    {mean_rand_en[-1]:>10.1f}  (median: {med_rand_en[-1]:.0f})")
print(f"  Random vocab:         {mean_rand_voc[-1]:>10.1f}  (median: {med_rand_voc[-1]:.0f})")

# Mid layers (where the paper claims EN rises)
for layer_idx in [15, 19, 23, 27]:
    layer_num = layer_idx + 1
    print(f"\nLayer {layer_num} mean ranks:")
    print(f"  Correct target-lang:  {mean_cor_obj[layer_idx]:>10.1f}")
    print(f"  Correct EN answer:    {mean_cor_en[layer_idx]:>10.1f}")
    print(f"  Random EN answers:    {mean_rand_en[layer_idx]:>10.1f}")
    print(f"  Random vocab:         {mean_rand_voc[layer_idx]:>10.1f}")

# Ratio: how much better is correct EN vs random EN?
ratio_en = mean_rand_en / (mean_cor_en + 1)  # +1 to avoid div-by-zero
print(f"\n{'='*80}")
print("SELECTIVITY: Correct EN rank / Random EN rank (lower = more selective)")
print(f"{'='*80}")
for i in [15, 19, 23, 27, 31]:
    r = mean_cor_en[i] / (mean_rand_en[i] + 1)
    print(f"  Layer {i+1:>2}: correct_EN is {1/r:.1f}x lower rank than random_EN "
          f"({mean_cor_en[i]:.0f} vs {mean_rand_en[i]:.0f})")

# Crossover analysis
print(f"\n{'='*80}")
print("CROSSOVER ANALYSIS")
print(f"{'='*80}")
for i in range(NUM_LAYERS - 1):
    if mean_cor_en[i] > mean_rand_en[i] and mean_cor_en[i+1] <= mean_rand_en[i+1]:
        print(f"  Correct EN drops below random EN at layer {i+2} "
              f"({mean_cor_en[i+1]:.0f} vs {mean_rand_en[i+1]:.0f})")
        break
else:
    # Check if correct EN is always below or always above
    if mean_cor_en[-1] < mean_rand_en[-1]:
        first_below = next((i for i in range(NUM_LAYERS) if mean_cor_en[i] < mean_rand_en[i]), None)
        if first_below is not None:
            print(f"  Correct EN first below random EN at layer {first_below+1}")
    else:
        print(f"  Correct EN never drops below random EN!")

for i in range(NUM_LAYERS - 1):
    if mean_cor_en[i] > mean_rand_voc[i] and mean_cor_en[i+1] <= mean_rand_voc[i+1]:
        print(f"  Correct EN drops below random vocab at layer {i+2} "
              f"({mean_cor_en[i+1]:.0f} vs {mean_rand_voc[i+1]:.0f})")
        break

# ── Interpretation ────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}")
final_cor_en = mean_cor_en[-1]
final_rand_en = mean_rand_en[-1]
final_rand_voc = mean_rand_voc[-1]

if final_cor_en < final_rand_en * 0.1 and final_rand_en < final_rand_voc * 0.5:
    print("  RESULT: EN rise is SPECIFIC to the correct answer.")
    print("  The model specifically activates the correct EN token, not English in general.")
    print("  Random EN answers behave more like random vocab than like the correct EN answer.")
elif final_cor_en < final_rand_en * 0.1 and final_rand_en >= final_rand_voc * 0.5:
    print("  RESULT: EN rise is PARTIALLY specific.")
    print("  The correct EN answer rises much more than random EN answers,")
    print("  but random EN answers also rise somewhat compared to random vocab.")
elif final_cor_en >= final_rand_en * 0.1:
    print("  RESULT: EN rise is NOT specific to the correct answer.")
    if final_rand_en < final_rand_voc * 0.5:
        print("  All EN answer tokens rise similarly — it's an English-general phenomenon.")
    else:
        print("  All tokens rise similarly — it's a general layer phenomenon, not EN-specific.")

# ── Figure 1: Main 4-line comparison ──────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Left: all layers
ax1.semilogy(layers, mean_cor_obj + 1,  "o-",  color="steelblue", lw=2.5, ms=4,
             label="Correct target-lang token")
ax1.semilogy(layers, mean_cor_en + 1,   "s-",  color="seagreen",  lw=2.5, ms=4,
             label="Correct EN answer")
ax1.semilogy(layers, mean_rand_en + 1,  "D--", color="darkorange", lw=2.0, ms=4,
             label="Random EN answers (mean of 10)")
ax1.semilogy(layers, mean_rand_voc + 1, "^:",  color="firebrick",  lw=2.0, ms=4,
             label="Random vocab tokens (mean of 10)")
ax1.invert_yaxis()
ax1.set_xlabel("Layer", fontsize=12)
ax1.set_ylabel("Mean Rank + 1  (log scale, lower = more probable)", fontsize=11)
ax1.set_title(f"All Layers (n={n_loaded} failure cases)", fontsize=12)
ax1.legend(fontsize=9, loc="lower left")
ax1.grid(True, alpha=0.3)

# Right: zoom on layers 15-32
z = slice(14, None)
x = layers[z]
ax2.semilogy(x, mean_cor_obj[z] + 1,  "o-",  color="steelblue", lw=2.5, ms=5,
             label="Correct target-lang token")
ax2.semilogy(x, mean_cor_en[z] + 1,   "s-",  color="seagreen",  lw=2.5, ms=5,
             label="Correct EN answer")
ax2.semilogy(x, mean_rand_en[z] + 1,  "D--", color="darkorange", lw=2.0, ms=5,
             label="Random EN answers (mean of 10)")
ax2.semilogy(x, mean_rand_voc[z] + 1, "^:",  color="firebrick",  lw=2.0, ms=5,
             label="Random vocab tokens (mean of 10)")
ax2.invert_yaxis()
ax2.set_xlabel("Layer", fontsize=12)
ax2.set_title("Zoomed: Layers 15–32", fontsize=12)
ax2.legend(fontsize=9, loc="lower left")
ax2.grid(True, alpha=0.3)

fig.suptitle(
    "Baseline Test: Is the EN Token Rise Specific to the Correct Answer?\n"
    f"{SHORT_NAME} — fail_en_ok cases (EN correct, target-lang wrong)",
    fontsize=13, y=1.02,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "baseline_logit_lens.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {OUT_DIR}/baseline_logit_lens.png")

# ── Figure 2: Selectivity ratio ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Ratio: how many times lower is correct_en rank vs baselines?
# (A ratio > 1 means the correct EN is at a lower/better rank)
ratio_vs_rand_en  = (mean_rand_en + 1) / (mean_cor_en + 1)
ratio_vs_rand_voc = (mean_rand_voc + 1) / (mean_cor_en + 1)

ax.plot(layers, ratio_vs_rand_en,  "D-", color="darkorange", lw=2.5, ms=5,
        label="vs Random EN answers")
ax.plot(layers, ratio_vs_rand_voc, "^-", color="firebrick",  lw=2.5, ms=5,
        label="vs Random vocab tokens")
ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Selectivity ratio\n(baseline rank / correct EN rank)", fontsize=11)
ax.set_title(
    "How Much Does the Model Prefer the Correct EN Answer Over Baselines?\n"
    f"Ratio > 1 = correct EN has lower rank (more activated)  |  n={n_loaded}",
    fontsize=12,
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "baseline_logit_lens_ratio.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT_DIR}/baseline_logit_lens_ratio.png")

# ── Figure 3: Median trajectories (robust to outliers) ───────────────────
fig, ax = plt.subplots(figsize=(12, 7))

ax.semilogy(layers, med_cor_obj + 1,  "o-",  color="steelblue", lw=2.5, ms=5,
            label="Correct target-lang token")
ax.semilogy(layers, med_cor_en + 1,   "s-",  color="seagreen",  lw=2.5, ms=5,
            label="Correct EN answer")
ax.semilogy(layers, med_rand_en + 1,  "D--", color="darkorange", lw=2.0, ms=5,
            label="Random EN answers (median)")
ax.semilogy(layers, med_rand_voc + 1, "^:",  color="firebrick",  lw=2.0, ms=5,
            label="Random vocab tokens (median)")
ax.invert_yaxis()
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Median Rank + 1  (log scale, lower = more probable)", fontsize=11)
ax.set_title(
    "Baseline Test — Median Ranks (robust to outliers)\n"
    f"{SHORT_NAME}, fail_en_ok cases  |  n={n_loaded}",
    fontsize=13,
)
ax.legend(fontsize=10, loc="lower left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "baseline_logit_lens_median.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT_DIR}/baseline_logit_lens_median.png")
