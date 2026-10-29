"""
06b_patchscopes_by_script_and_layer.py
---------------------------------------
Deeper analysis of patchscope results.

The basic subject_match gives 15.2% success, but looking at outputs qualitatively,
the model often shows clear entity recognition without repeating the name literally.

This script analyzes:
1. Script effect: Latin-script subjects vs non-Latin (CJK/Cyrillic)
   - Hypothesis: patchscopes succeed more when the subject is Latin-script
     because the model's internal representation is English-aligned
2. What the model actually outputs in "failures" — garbage or semantic content?
3. Per-layer quality: even when name doesn't match, how related is the output?
4. Shared-name effect: subjects that are the same string across languages
   (e.g. "Madrid" is "Madrid" in most languages)

Produces:
  - figures/patchscopes_three_metrics.png  — exact/partial/semantic match rates per layer
  - figures/patchscopes_script_effect.png  — Latin vs non-Latin script success comparison
  - figures/patchscopes_per_layer_deep.png — per-layer breakdown with all three metrics
  - stdout: script effect stats, failure output analysis
"""

import os, sys, json, glob, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from load_cache_and_match import OUT_DIR, NUM_LAYERS, NON_EN, MODEL_TAG, CACHE_DIR
import numpy as np
from collections import Counter, defaultdict

from paths import DATA_ROOT
BASE_DIR = str(DATA_ROOT)
PS_DIR = os.path.join(BASE_DIR, "cache", f"{MODEL_TAG}_patchscopes", "examples")
ORIG_DIR = CACHE_DIR


def is_latin(s):
    """Check if string is primarily Latin-script."""
    if not s:
        return False
    latin = sum(1 for c in s if c.isascii() and c.isalpha())
    total = sum(1 for c in s if c.isalpha())
    if total == 0:
        return False
    return latin / total > 0.5


def has_semantic_content(texts, subject, relation):
    """Check if patchscope outputs show entity recognition beyond name matching.
    Returns (has_semantic, best_evidence)."""
    subject_lower = subject.lower()
    subject_words = set(subject_lower.split())
    # Remove very short words (articles, prepositions)
    subject_words = {w for w in subject_words if len(w) > 2}

    best_evidence = ""
    for t in texts:
        t_lower = t.lower().strip()
        if not t_lower or len(t_lower) < 2:
            continue
        # Check for subject word overlap (partial name recognition)
        t_words = set(t_lower.replace(",", " ").replace(".", " ").split())
        overlap = subject_words & t_words
        if overlap and len(list(overlap)[0]) >= 3:
            return True, t
        # Check for typical entity-description patterns
        desc_patterns = [
            "was born", "was the", "was a", "is the", "is a",
            "died in", "born in", "founded", "dynasty",
            "the first", "the son", "the king", "the founder",
        ]
        for pat in desc_patterns:
            if pat in t_lower:
                if not best_evidence:
                    best_evidence = t
    if best_evidence:
        return True, best_evidence
    return False, ""


# ── Load patchscope data + original data ──────────────────────────────────
print("Loading patchscope results...")
records = []

for path in sorted(glob.glob(os.path.join(PS_DIR, "*.json"))):
    with open(path, encoding="utf-8") as f:
        ps_data = json.load(f)

    idx = ps_data.get("index")
    rel = ps_data.get("relation", "")

    # Load original cache for EN subject name comparison
    orig_path = os.path.join(ORIG_DIR, os.path.basename(path))
    en_subject = None
    if os.path.exists(orig_path):
        with open(orig_path, encoding="utf-8") as f:
            orig_data = json.load(f)
        en_rec = orig_data.get("en", {})
        if isinstance(en_rec, dict):
            en_subject = en_rec.get("subject", "")

    for lang, rec in ps_data.items():
        if lang in ("index", "relation") or not isinstance(rec, dict):
            continue
        ps = rec.get("patchscopes", {})
        if not ps:
            continue
        subject = rec.get("subject", "")
        texts = ps.get("per_layer_text", [])
        matches = ps.get("per_layer_match", [])

        # Classify subject script
        subj_latin = is_latin(subject)

        # Check semantic recognition
        has_sem, sem_evidence = has_semantic_content(texts, subject, rel)

        # Also check: does EN subject name appear in any output?
        en_name_found = False
        if en_subject:
            en_lower = en_subject.lower()
            en_words = {w for w in en_lower.split() if len(w) > 2}
            for t in texts:
                t_lower = t.lower()
                # Check full name or significant first/last name
                if en_lower in t_lower:
                    en_name_found = True
                    break
                t_words = set(t_lower.replace(",", " ").split())
                if en_words & t_words:
                    en_name_found = True
                    break

        records.append({
            "lang": lang,
            "relation": rel,
            "subject": subject,
            "en_subject": en_subject,
            "subject_latin": subj_latin,
            "name_match": ps.get("patchscopes_success", False),
            "semantic_match": has_sem,
            "en_name_found": en_name_found,
            "sem_evidence": sem_evidence,
            "texts": texts,
            "matches": matches,
            "first_success": ps.get("first_success_layer"),
        })

print(f"Total records: {len(records)}")

# ── Analysis 1: Script effect ────────────────────────────────────────────
print("\n" + "=" * 70)
print("ANALYSIS 1: Script Effect (Latin vs Non-Latin subjects)")
print("=" * 70)

latin = [r for r in records if r["subject_latin"]]
nonlatin = [r for r in records if not r["subject_latin"]]

print(f"\nLatin-script subjects:     {len(latin)}")
print(f"  Name match:     {sum(r['name_match'] for r in latin):>5} "
      f"({sum(r['name_match'] for r in latin)/len(latin)*100:.1f}%)")
print(f"  Semantic match: {sum(r['semantic_match'] for r in latin):>5} "
      f"({sum(r['semantic_match'] for r in latin)/len(latin)*100:.1f}%)")
print(f"  EN name found:  {sum(r['en_name_found'] for r in latin):>5} "
      f"({sum(r['en_name_found'] for r in latin)/len(latin)*100:.1f}%)")

print(f"\nNon-Latin subjects:        {len(nonlatin)}")
print(f"  Name match:     {sum(r['name_match'] for r in nonlatin):>5} "
      f"({sum(r['name_match'] for r in nonlatin)/len(nonlatin)*100:.1f}%)")
print(f"  Semantic match: {sum(r['semantic_match'] for r in nonlatin):>5} "
      f"({sum(r['semantic_match'] for r in nonlatin)/len(nonlatin)*100:.1f}%)")
print(f"  EN name found:  {sum(r['en_name_found'] for r in nonlatin):>5} "
      f"({sum(r['en_name_found'] for r in nonlatin)/len(nonlatin)*100:.1f}%)")

# Per language with script info
print(f"\nPer-language breakdown:")
print(f"{'Lang':<5} {'Script':<8} {'N':>5} {'Name%':>7} {'Sem%':>7} {'EN-name%':>9}")
by_lang = defaultdict(list)
for r in records:
    by_lang[r["lang"]].append(r)
for lang in sorted(by_lang):
    recs = by_lang[lang]
    n = len(recs)
    nm = sum(r["name_match"] for r in recs)
    sm = sum(r["semantic_match"] for r in recs)
    en = sum(r["en_name_found"] for r in recs)
    script = "latin" if recs[0]["subject_latin"] else "non-lat"
    # Actually check - some languages use Latin for some subjects
    n_latin = sum(r["subject_latin"] for r in recs)
    script = f"{n_latin}/{n}"
    print(f"{lang:<5} {script:<8} {n:>5} {nm/n*100:>6.1f}% {sm/n*100:>6.1f}% {en/n*100:>8.1f}%")


# ── Analysis 2: What do "failures" actually output? ──────────────────────
print("\n" + "=" * 70)
print("ANALYSIS 2: What Do Failures Actually Output?")
print("=" * 70)

failures = [r for r in records if not r["name_match"]]
n_fail = len(failures)

n_semantic = sum(r["semantic_match"] for r in failures)
n_en_name = sum(r["en_name_found"] for r in failures)
n_garbage = sum(1 for r in failures if not r["semantic_match"] and not r["en_name_found"])

print(f"\nAmong {n_fail} name-match failures:")
print(f"  Has semantic content:   {n_semantic:>5} ({n_semantic/n_fail*100:.1f}%)")
print(f"  EN name found:          {n_en_name:>5} ({n_en_name/n_fail*100:.1f}%)")
print(f"  Pure garbage:           {n_garbage:>5} ({n_garbage/n_fail*100:.1f}%)")

# Revised success metric: name match OR semantic recognition
all_recognition = sum(r["name_match"] or r["semantic_match"] for r in records)
print(f"\n** REVISED RECOGNITION RATE (name OR semantic): "
      f"{all_recognition}/{len(records)} ({all_recognition/len(records)*100:.1f}%) **")
print(f"   (vs strict name match: {sum(r['name_match'] for r in records)/len(records)*100:.1f}%)")

# ── Analysis 3: Same-name subjects ───────────────────────────────────────
print("\n" + "=" * 70)
print("ANALYSIS 3: Same-Name vs Different-Name Subjects")
print("=" * 70)
# Subjects where the non-EN name equals the EN name (e.g., "Madrid" everywhere)
same_name = [r for r in records if r["en_subject"] and
             r["subject"].lower().strip() == r["en_subject"].lower().strip()]
diff_name = [r for r in records if r["en_subject"] and
             r["subject"].lower().strip() != r["en_subject"].lower().strip()]

if same_name:
    print(f"\nSame name as EN: {len(same_name)}")
    print(f"  Name match: {sum(r['name_match'] for r in same_name)/len(same_name)*100:.1f}%")
    print(f"  Semantic:   {sum(r['semantic_match'] for r in same_name)/len(same_name)*100:.1f}%")
if diff_name:
    print(f"\nDifferent name from EN: {len(diff_name)}")
    print(f"  Name match: {sum(r['name_match'] for r in diff_name)/len(diff_name)*100:.1f}%")
    print(f"  Semantic:   {sum(r['semantic_match'] for r in diff_name)/len(diff_name)*100:.1f}%")


# ── Analysis 4: Examples of semantic recognition without name match ──────
print("\n" + "=" * 70)
print("ANALYSIS 4: Examples of Semantic Recognition Without Name Match")
print("=" * 70)

sem_no_name = [r for r in records if r["semantic_match"] and not r["name_match"]]
# Show diverse examples
shown_subjects = set()
count = 0
for r in sem_no_name:
    if r["en_subject"] in shown_subjects:
        continue
    shown_subjects.add(r["en_subject"])
    print(f"\n  {r['lang']} | Subject: {r['subject']} (EN: {r['en_subject']})")
    print(f"  Relation: {r['relation']}")
    print(f"  Evidence: '{r['sem_evidence']}'")
    # Show a few layer outputs
    mid = len(r["texts"]) // 2
    samples = [(0, r["texts"][0]), (mid, r["texts"][mid]), (-1, r["texts"][-1])]
    for idx, t in samples:
        print(f"    L{idx+1 if idx >= 0 else NUM_LAYERS}: {t}")
    count += 1
    if count >= 15:
        break


# ── Analysis 5: Per-layer semantic recognition rate ──────────────────────
print("\n" + "=" * 70)
print("ANALYSIS 5: Per-Layer Entity Recognition (name OR semantic)")
print("=" * 70)

# For each layer, check what fraction of outputs show any sign of recognition
# This is harder to do per-layer for semantic, so let's check EN name per layer
per_layer_en_name = [0] * NUM_LAYERS
for r in records:
    if not r["en_subject"]:
        continue
    en_lower = r["en_subject"].lower()
    en_words = {w for w in en_lower.split() if len(w) > 2}
    for i, t in enumerate(r["texts"]):
        if i >= NUM_LAYERS:
            break
        t_lower = t.lower()
        t_words = set(t_lower.replace(",", " ").split())
        if en_lower in t_lower or (en_words & t_words):
            per_layer_en_name[i] += 1

print(f"{'Layer':<6} {'EN-name':>8} {'Rate':>8}")
for i in range(NUM_LAYERS):
    rate = per_layer_en_name[i] / len(records) * 100
    print(f"{i+1:<6} {per_layer_en_name[i]:>8} {rate:>7.1f}%")


# ── Figures ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure 1: Three metrics compared by language
langs = sorted(by_lang.keys())
n_langs = len(langs)
x = np.arange(n_langs)
w = 0.25

name_rates = []
sem_rates = []
en_rates = []
for lang in langs:
    recs = by_lang[lang]
    n = len(recs)
    name_rates.append(sum(r["name_match"] for r in recs) / n * 100)
    sem_rates.append(sum(r["semantic_match"] for r in recs) / n * 100)
    en_rates.append(sum(r["en_name_found"] for r in recs) / n * 100)

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - w, name_rates, w, label="Exact name match", color="steelblue")
ax.bar(x, sem_rates, w, label="Semantic recognition", color="seagreen")
ax.bar(x + w, en_rates, w, label="EN name found", color="darkorange")
ax.set_xlabel("Language", fontsize=12)
ax.set_ylabel("Recognition Rate (%)", fontsize=12)
ax.set_title("Patchscopes: Three Recognition Metrics by Language\n"
             "Strict name match vs semantic content vs EN name in output",
             fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(langs)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "patchscopes_three_metrics.png"), dpi=150)
plt.close(fig)
print(f"\nSaved {OUT_DIR}/patchscopes_three_metrics.png")

# Figure 2: Latin vs non-Latin script effect
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, group, title in [
    (axes[0], latin, f"Latin-script subjects (n={len(latin)})"),
    (axes[1], nonlatin, f"Non-Latin subjects (n={len(nonlatin)})"),
]:
    vals = [
        sum(r["name_match"] for r in group) / len(group) * 100,
        sum(r["semantic_match"] for r in group) / len(group) * 100,
        sum(r["en_name_found"] for r in group) / len(group) * 100,
    ]
    colors = ["steelblue", "seagreen", "darkorange"]
    labels = ["Name match", "Semantic", "EN name"]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Rate (%)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)

fig.suptitle("Script Effect on Patchscope Success", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "patchscopes_script_effect.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}/patchscopes_script_effect.png")

# Figure 3: Per-layer EN-name recognition rate
fig, ax = plt.subplots(figsize=(12, 6))
layers = np.arange(1, NUM_LAYERS + 1)
en_name_rates = [per_layer_en_name[i] / len(records) * 100 for i in range(NUM_LAYERS)]

# Also overlay the strict name match from script 06
strict_match = [0] * NUM_LAYERS
for r in records:
    for i, m in enumerate(r["matches"]):
        if i < NUM_LAYERS and m:
            strict_match[i] += 1
strict_rates = [strict_match[i] / len(records) * 100 for i in range(NUM_LAYERS)]

ax.plot(layers, strict_rates, "o-", color="steelblue", lw=2, ms=4, label="Strict name match")
ax.plot(layers, en_name_rates, "s-", color="darkorange", lw=2, ms=4, label="EN name found in output")
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Recognition Rate (%)", fontsize=12)
ax.set_title("Per-Layer Subject Recognition: Strict vs EN-Name Matching\n"
             f"n={len(records)} fail_en_ok cases", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(layers)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "patchscopes_per_layer_deep.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}/patchscopes_per_layer_deep.png")
