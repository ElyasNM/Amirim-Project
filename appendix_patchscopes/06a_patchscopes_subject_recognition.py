"""
06a_patchscopes_subject_recognition.py
---------------------------------------
Analyze patchscope results: does the model recognize the SUBJECT at each layer?

Produces:
  - figures/patchscopes_per_layer.png          — subject recognition success rate per layer
  - figures/patchscopes_by_language.png        — success rate broken down by language
  - figures/patchscopes_first_success_hist.png — histogram of first-success layer (if any successes)
  - stdout: overall success rate, per-language breakdown, per-relation breakdown
"""

import os, sys, json, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from load_cache_and_match import OUT_DIR, NUM_LAYERS, MODEL_TAG, SHORT_NAME
import numpy as np
from collections import Counter

from paths import DATA_ROOT
BASE_DIR = str(DATA_ROOT)
PS_DIR = os.path.join(BASE_DIR, "cache", f"{MODEL_TAG}_patchscopes", "examples")

total = 0
success = 0
by_lang = Counter()
by_lang_total = Counter()
first_layers = []
last_layers = []
per_layer_success = [0] * NUM_LAYERS
per_layer_total = [0] * NUM_LAYERS
by_relation = Counter()
by_relation_total = Counter()
example_texts = []

for path in sorted(glob.glob(os.path.join(PS_DIR, "*.json"))):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    relation = data.get("relation", "")
    for lang, rec in data.items():
        if lang in ("index", "relation") or not isinstance(rec, dict):
            continue
        ps = rec.get("patchscopes", {})
        if not ps:
            continue
        total += 1
        by_lang_total[lang] += 1
        by_relation_total[relation] += 1
        matches = ps.get("per_layer_match", [])
        texts = ps.get("per_layer_text", [])
        for i, m in enumerate(matches):
            if i < NUM_LAYERS:
                per_layer_total[i] += 1
                if m:
                    per_layer_success[i] += 1
        if ps.get("patchscopes_success"):
            success += 1
            by_lang[lang] += 1
            by_relation[relation] += 1
            fl = ps.get("first_success_layer")
            ll = ps.get("last_success_layer")
            if fl is not None:
                first_layers.append(fl)
            if ll is not None:
                last_layers.append(ll)
        if len(example_texts) < 20:
            example_texts.append({
                "lang": lang,
                "subject": rec.get("subject", ""),
                "relation": relation,
                "success": ps.get("patchscopes_success", False),
                "texts": texts,
            })

print("=" * 70)
print("PATCHSCOPES ANALYSIS: Subject Recognition Through Layers")
print("=" * 70)
print(f"Total examples: {total}")
print(f"Patchscopes success: {success} ({success/total*100:.1f}%)")
print(f"Patchscopes fail: {total-success} ({(total-success)/total*100:.1f}%)")

print("\n--- Per Language ---")
print(f"{'Lang':<6} {'Success':>8} {'Total':>8} {'Rate':>8}")
for lang in sorted(by_lang_total):
    s = by_lang[lang]
    t = by_lang_total[lang]
    print(f"{lang:<6} {s:>8} {t:>8} {s/t*100:>7.1f}%")

print("\n--- Per Relation ---")
print(f"{'Relation':<30} {'Success':>8} {'Total':>8} {'Rate':>8}")
for rel in sorted(by_relation_total):
    s = by_relation[rel]
    t = by_relation_total[rel]
    print(f"{rel:<30} {s:>8} {t:>8} {s/t*100:>7.1f}%")

if first_layers:
    arr = np.array(first_layers)
    print("\n--- First Success Layer (when patchscopes succeeds) ---")
    print(f"  mean={arr.mean():.1f}, median={np.median(arr):.0f}, min={arr.min()}, max={arr.max()}")
    bins = [(1,5),(6,10),(11,15),(16,20),(21,25),(26,32)]
    for lo, hi in bins:
        c = ((arr >= lo) & (arr <= hi)).sum()
        print(f"  layers {lo:>2}-{hi:>2}: {c:>5} ({c/len(arr)*100:.1f}%)")

print("\n--- Per-Layer Success Rate ---")
print(f"{'Layer':<6} {'Success':>8} {'Total':>8} {'Rate':>8}")
for i in range(NUM_LAYERS):
    if per_layer_total[i] > 0:
        rate = per_layer_success[i] / per_layer_total[i] * 100
        print(f"{i+1:<6} {per_layer_success[i]:>8} {per_layer_total[i]:>8} {rate:>7.1f}%")

print("\n--- Example Patchscope Outputs (first 20) ---")
for ex in example_texts:
    tag = "OK" if ex["success"] else "FAIL"
    print(f"\n[{tag}] {ex['lang']} | {ex['subject']} | {ex['relation']}")
    for i, t in enumerate(ex["texts"]):
        marker = " <--" if i < len(ex.get("texts", [])) else ""
        print(f"  L{i+1:>2}: {t}")

# ── Figures ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure 1: per-layer success rate
layers = np.arange(1, NUM_LAYERS + 1)
rates = [per_layer_success[i] / per_layer_total[i] * 100 if per_layer_total[i] > 0 else 0 for i in range(NUM_LAYERS)]

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(layers, rates, color="steelblue", alpha=0.8)
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Subject Recognition Rate (%)", fontsize=12)
ax.set_title(f"Patchscopes: At Which Layer Does the Model Recognize the Subject?\n{SHORT_NAME}, fail_en_ok cases | n={total}", fontsize=13)
ax.set_xticks(layers)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "patchscopes_per_layer.png"), dpi=150)
plt.close(fig)
print(f"\nSaved {OUT_DIR}/patchscopes_per_layer.png")

# Figure 2: per-language success rate
langs = sorted(by_lang_total.keys())
lang_rates = [by_lang[l] / by_lang_total[l] * 100 for l in langs]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(langs, lang_rates, color="seagreen", alpha=0.8)
ax.set_xlabel("Language", fontsize=12)
ax.set_ylabel("Subject Recognition Rate (%)", fontsize=12)
ax.set_title(f"Patchscopes Success Rate by Language | n={total}", fontsize=13)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "patchscopes_by_language.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}/patchscopes_by_language.png")

# Figure 3: histogram of first success layer
if first_layers:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(first_layers, bins=range(1, NUM_LAYERS + 2), color="darkorange", alpha=0.8, edgecolor="black")
    ax.set_xlabel("First Success Layer", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"When Does Subject Recognition First Succeed?\nn={len(first_layers)} successful cases", fontsize=13)
    ax.set_xticks(range(1, NUM_LAYERS + 1, 2))
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "patchscopes_first_success_hist.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/patchscopes_first_success_hist.png")
