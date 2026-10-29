"""
02_english_bias_test.py
-----------------------
Evaluate whether the model shows an English accuracy advantage.

Uses truncation-aware fixed matching to label each language prediction as
correct or failed, then computes:
  - how often English-correct facts still fail in non-English languages
  - the English vs non-English accuracy gap by relation
  - pairwise Jaccard similarity of language failure sets

Produces:
  - figures/english_bias_test.png    — distribution of non-EN failures among EN-correct facts
  - figures/en_gap_by_relation.png   — EN advantage per relation type
  - figures/jaccard_fixed.png        — pairwise Jaccard similarity of failure sets (fixed matching)
  - stdout: tables
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from load_cache_and_match import *
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

facts = load_facts()

# ── Compute failures with fixed matching ────────────────────────────────────
fail = defaultdict(set)
corr = defaultdict(int)
total = defaultdict(int)

for key, ld in facts.items():
    for lang, rec in ld.items():
        total[lang] += 1
        target = " " + rec["object"]
        if match_fixed(rec["prediction"], target, rec):
            corr[lang] += 1
        else:
            fail[lang].add(key)

# ── English bias distribution ───────────────────────────────────────────────
en_ok = [k for k in facts if k not in fail["en"]]
en_fail = [k for k in facts if k in fail["en"]]
print(f"EN correct (fixed): {len(en_ok)} / {len(facts)} ({len(en_ok)/len(facts)*100:.1f}%)")
print(f"EN wrong   (fixed): {len(en_fail)} / {len(facts)}")

dist = defaultdict(int)
for k in en_ok:
    n = sum(1 for l in NON_EN if k in fail[l])
    dist[n] += 1

print(f"\nAmong EN-correct facts: how many non-EN langs also fail?")
print(f"{'#NonEN fail':>12} {'Count':>6} {'Pct':>7}")
print("-" * 28)
for n in range(12):
    c = dist.get(n, 0)
    print(f"{n:>12} {c:>6} {c/len(en_ok)*100:>6.1f}%")

# ── Figure: EN bias distribution ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
xs = list(range(12))
ys = [dist.get(n, 0) for n in xs]
colors = ["#5cb85c" if n <= 3 else "#f0ad4e" if n <= 7 else "#d9534f" for n in xs]
ax.bar(xs, ys, color=colors, edgecolor="white")
ax.set_xlabel("#Non-English languages that also fail", fontsize=11)
ax.set_ylabel("#Facts (where EN is correct)", fontsize=11)
ax.set_title("Among facts English gets RIGHT: how many other languages fail?\n(If English-biased, we'd see mass at 8-11)", fontsize=11)
ax.set_xticks(xs)
for i, v in enumerate(ys):
    ax.text(i, v + 5, f"{v/len(en_ok)*100:.0f}%", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "english_bias_test.png"), dpi=150)
plt.close(fig)
print(f"\nSaved {OUT_DIR}/english_bias_test.png")

# ── Per-relation EN gap ─────────────────────────────────────────────────────
rel_stats = defaultdict(lambda: defaultdict(lambda: {"ok": 0, "n": 0}))
for key, ld in facts.items():
    for lang, rec in ld.items():
        target = " " + rec["object"]
        rel_stats[key[1]][lang]["n"] += 1
        if match_fixed(rec["prediction"], target, rec):
            rel_stats[key[1]][lang]["ok"] += 1

gaps = []
for rel in sorted(rel_stats):
    en_a = rel_stats[rel]["en"]["ok"] / rel_stats[rel]["en"]["n"] * 100
    ne = [rel_stats[rel][l]["ok"] / rel_stats[rel][l]["n"] * 100 for l in NON_EN if rel_stats[rel][l]["n"] > 0]
    ne_avg = sum(ne) / len(ne) if ne else 0
    gaps.append((rel, en_a - ne_avg, en_a, ne_avg))

gaps.sort(key=lambda x: x[1])

print(f"\n{'Relation':<28} {'EN%':>6} {'NonEN%':>8} {'Gap':>8}")
print("-" * 54)
for rel, gap, ea, na in sorted(gaps, key=lambda x: -x[1]):
    print(f"{rel:<28} {ea:>5.1f}% {na:>7.1f}% {gap:>+7.1f}%")

fig, ax = plt.subplots(figsize=(10, 7))
rels = [g[0] for g in gaps]
vals = [g[1] for g in gaps]
colors = ["#d9534f" if v > 10 else "#5cb85c" if v < 0 else "#f0ad4e" for v in vals]
ax.barh(range(len(rels)), vals, color=colors, edgecolor="white")
ax.set_yticks(range(len(rels))); ax.set_yticklabels(rels, fontsize=9)
ax.set_xlabel("EN accuracy - avg non-EN accuracy (pp)", fontsize=11)
ax.set_title("English Advantage by Relation Type\n(Red = large EN advantage, Green = EN disadvantage)", fontsize=11)
ax.axvline(0, color="black", linewidth=0.5)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "en_gap_by_relation.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}/en_gap_by_relation.png")

# ── Jaccard heatmap ─────────────────────────────────────────────────────────
n = len(LANGS)
jaccard = np.zeros((n, n))
for i, la in enumerate(LANGS):
    for j, lb in enumerate(LANGS):
        inter = len(fail[la] & fail[lb])
        union = len(fail[la] | fail[lb])
        jaccard[i, j] = inter / union if union else 0

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(jaccard, vmin=0, vmax=1, cmap="RdYlGn_r")
ax.set_xticks(range(n)); ax.set_xticklabels(LANGS, rotation=45, ha="right")
ax.set_yticks(range(n)); ax.set_yticklabels(LANGS)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{jaccard[i,j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax, label="Jaccard similarity of failure sets")
ax.set_title("Pairwise Jaccard Similarity of Failures (fixed matching)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "jaccard_fixed.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}/jaccard_fixed.png")
