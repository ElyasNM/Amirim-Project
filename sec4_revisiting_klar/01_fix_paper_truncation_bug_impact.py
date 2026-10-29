"""
01_fix_paper_truncation_bug_impact.py
--------------------------------------
Quantifies the prediction-extraction truncation bug.

The bug: `decoded[len(few_shot_prompt):]` chops the start of predictions
because tokenizer.decode(tokenizer.encode(text)) can differ in length.

Produces:
  - figures/corrected_accuracy.png
  - figures/corrected_histogram.png
  - stdout: per-language accuracy table, histogram, classification of failures
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from load_cache_and_match import *  # noqa: F401,F403 (SHORT_NAME, OUT_DIR, etc.)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

facts = load_facts()

# ── Classify every failure ──────────────────────────────────────────────────
stats = defaultdict(lambda: defaultdict(int))
total_records = total_matches = total_failures = 0

for key, ld in facts.items():
    for lang, rec in ld.items():
        total_records += 1
        target = " " + rec["object"]
        pred = rec["prediction"]

        if match_original(pred, target):
            total_matches += 1
            continue

        total_failures += 1
        p = norm(pred)
        t = norm(rec["object"])

        if len(p) > 0 and t.endswith(p) and p != t:
            stats[lang]["truncated"] += 1
        elif len(p) == 0:
            stats[lang]["empty"] += 1
        else:
            stats[lang]["genuine"] += 1
        stats[lang]["total_fail"] += 1


print(f"Total records: {total_records}")
print(f"Total matches: {total_matches}")
print(f"Total failures: {total_failures}")
print()
print(f"{'Lang':<6} {'Fail':>6} {'Trunc':>6} {'Empty':>6} {'Real':>6} {'Trunc%':>8}")
print("-" * 44)
for lang in LANGS:
    s = stats[lang]
    t = s["truncated"]; e = s["empty"]; g = s["genuine"]; tot = s["total_fail"]
    pct = t / tot * 100 if tot else 0
    print(f"{lang:<6} {tot:>6} {t:>6} {e:>6} {g:>6} {pct:>7.1f}%")

# ── Accuracy: original vs fixed ────────────────────────────────────────────
corr_o = defaultdict(int)
corr_f = defaultdict(int)
total = defaultdict(int)
fail_o = defaultdict(set)
fail_f = defaultdict(set)

for key, ld in facts.items():
    for lang, rec in ld.items():
        total[lang] += 1
        target = " " + rec["object"]
        if match_original(rec["prediction"], target):
            corr_o[lang] += 1
        else:
            fail_o[lang].add(key)
        if match_fixed(rec["prediction"], target, rec):
            corr_f[lang] += 1
        else:
            fail_f[lang].add(key)

print()
print(f"{'Lang':<6} {'Original':>10} {'Fixed':>10} {'Recovered':>10}")
print("-" * 40)
for lang in LANGS:
    t = total[lang]
    o = corr_o[lang] / t * 100
    f = corr_f[lang] / t * 100
    r = corr_f[lang] - corr_o[lang]
    print(f"{lang:<6} {o:>9.1f}% {f:>9.1f}% {r:>+10}")

# ── Figure 1: side-by-side accuracy bars ────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(LANGS))
w = 0.35
orig_accs = [corr_o[l] / total[l] * 100 for l in LANGS]
fix_accs  = [corr_f[l] / total[l] * 100 for l in LANGS]
b1 = ax.bar(x - w/2, orig_accs, w, label="Original (buggy)", color="#d9534f", alpha=0.8)
b2 = ax.bar(x + w/2, fix_accs,  w, label="Truncation-fixed", color="#5cb85c", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(LANGS, fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title(f"Per-Language Accuracy: Original vs Truncation-Fixed\n({SHORT_NAME} on KLAR)", fontsize=12)
ax.legend(fontsize=11); ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)
for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=8)
for b in b2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "corrected_accuracy.png"), dpi=150)
plt.close(fig)
print(f"\nSaved {OUT_DIR}/corrected_accuracy.png")

# ── Figure 2: side-by-side histograms ───────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
xs = list(range(13))
h_o = [sum(1 for k in facts if sum(1 for l in LANGS if k in fail_o[l]) == n) for n in xs]
h_f = [sum(1 for k in facts if sum(1 for l in LANGS if k in fail_f[l]) == n) for n in xs]

ax1.bar(xs, h_o, color="#d9534f", edgecolor="white", alpha=0.8)
ax1.set_title("Original (buggy matching)", fontsize=11)
ax1.set_xlabel("#Languages failing per fact"); ax1.set_ylabel("# Facts"); ax1.set_xticks(xs)
ax2.bar(xs, h_f, color="#5cb85c", edgecolor="white", alpha=0.8)
ax2.set_title("Truncation-fixed matching", fontsize=11)
ax2.set_xlabel("#Languages failing per fact"); ax2.set_ylabel("# Facts"); ax2.set_xticks(xs)
for ax in [ax1, ax2]: ax.set_ylim(0, max(max(h_o), max(h_f)) * 1.1)
fig.suptitle("Failure Distribution: How many languages fail per fact?", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "corrected_histogram.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}/corrected_histogram.png")
