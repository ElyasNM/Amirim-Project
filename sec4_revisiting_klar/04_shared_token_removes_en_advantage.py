"""
04_shared_token_removes_en_advantage.py
----------------------------------------
Q1: If the model "thinks in English," why isn't EN massively better?
Q2: Why is place_of_birth the biggest EN gap? What makes it unique?

Hypothesis for Q1: Many answers share the same token across languages
(e.g., "Madrid" is "Madrid" in EN, ES, FR). When they share a token,
there's no translation needed, so no EN advantage.

Hypothesis for Q2: place_of_birth has more diverse/obscure answers that
differ across languages. Capitals are well-known cities with similar
names; birthplaces are obscure towns with language-specific names.

Produces:
  - figures/same_vs_diff_token_accuracy.png — accuracy split by whether EN/target share a token
  - stdout: token-sharing rates, accuracy tables
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))

from load_cache_and_match import *
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

facts = load_facts()

# =====================================================================
# Q1: How often do EN and target language share the same first token?
# =====================================================================

same_tok = defaultdict(int)
diff_tok = defaultdict(int)
same_tok_acc = defaultdict(lambda: {"ok": 0, "n": 0})
diff_tok_acc = defaultdict(lambda: {"ok": 0, "n": 0})

for key, ld in facts.items():
    en_rec = ld.get("en")
    if not en_rec:
        continue
    for lang in NON_EN:
        rec = ld.get(lang)
        if not rec:
            continue
        ll = rec.get("logit_lens", {})
        ctid = ll.get("correct_object_token_id")
        etid = ll.get("correct_en_token_id")
        if ctid is None or etid is None:
            continue

        target = " " + rec["object"]
        ok = match_fixed(rec["prediction"], target, rec)

        if ctid == etid:
            same_tok[lang] += 1
            same_tok_acc[lang]["n"] += 1
            same_tok_acc[lang]["ok"] += int(ok)
        else:
            diff_tok[lang] += 1
            diff_tok_acc[lang]["n"] += 1
            diff_tok_acc[lang]["ok"] += int(ok)

print("=" * 70)
print("Q1: SAME-TOKEN vs DIFFERENT-TOKEN accuracy")
print("    (Does sharing a token with EN eliminate the gap?)")
print("=" * 70)
print(f"{'Lang':<6} {'Same#':>6} {'Same%':>8} {'SameAcc':>9}  {'Diff#':>6} {'DiffAcc':>9}  {'Gap':>7}")
print("-" * 65)
for lang in NON_EN:
    sn = same_tok_acc[lang]["n"]
    so = same_tok_acc[lang]["ok"]
    sa = so / sn * 100 if sn else 0
    dn = diff_tok_acc[lang]["n"]
    do = diff_tok_acc[lang]["ok"]
    da = do / dn * 100 if dn else 0
    frac = sn / (sn + dn) * 100 if (sn + dn) else 0
    print(f"{lang:<6} {sn:>6} {frac:>7.1f}% {sa:>8.1f}%  {dn:>6} {da:>8.1f}%  {sa-da:>+6.1f}%")

# Overall: EN accuracy for reference
en_corr = sum(1 for k, ld in facts.items()
              if "en" in ld and match_fixed(ld["en"]["prediction"], " " + ld["en"]["object"], ld["en"]))
print(f"\nEN accuracy: {en_corr}/{len(facts)} = {en_corr/len(facts)*100:.1f}%")

# Aggregate same vs diff
all_same_ok = sum(same_tok_acc[l]["ok"] for l in NON_EN)
all_same_n  = sum(same_tok_acc[l]["n"] for l in NON_EN)
all_diff_ok = sum(diff_tok_acc[l]["ok"] for l in NON_EN)
all_diff_n  = sum(diff_tok_acc[l]["n"] for l in NON_EN)
print(f"\nALL non-EN same-token: {all_same_ok}/{all_same_n} = {all_same_ok/all_same_n*100:.1f}%")
print(f"ALL non-EN diff-token: {all_diff_ok}/{all_diff_n} = {all_diff_ok/all_diff_n*100:.1f}%")
print(f"Gap: {all_same_ok/all_same_n*100 - all_diff_ok/all_diff_n*100:+.1f} pp")

# =====================================================================
# Q2: What makes place_of_birth unique?
# =====================================================================
print()
print("=" * 70)
print("Q2: WHY is place_of_birth the biggest EN gap?")
print("=" * 70)

# For each geographic relation, compute:
# 1. What fraction of answers share a token with EN?
# 2. How many unique answer entities are there?
# 3. Average answer "popularity" (how often the same answer appears)

geo_rels_ordered = ["place_of_birth", "place_of_death", "headquarters_location",
                    "location_of_formation", "capital", "capital_of"]

for rel in geo_rels_ordered:
    total_n = 0
    same_n = 0
    ok_n = 0
    en_ok_n = 0
    en_total = 0
    unique_en_answers = set()
    all_en_answers = []

    for key, ld in facts.items():
        if key[1] != rel:
            continue
        en_rec = ld.get("en")
        if not en_rec:
            continue

        unique_en_answers.add(en_rec["object"].lower().strip())
        all_en_answers.append(en_rec["object"].lower().strip())
        en_total += 1
        if match_fixed(en_rec["prediction"], " " + en_rec["object"], en_rec):
            en_ok_n += 1

        for lang in NON_EN:
            rec = ld.get(lang)
            if not rec:
                continue
            ll = rec.get("logit_lens", {})
            ctid = ll.get("correct_object_token_id")
            etid = ll.get("correct_en_token_id")
            total_n += 1
            if ctid is not None and etid is not None and ctid == etid:
                same_n += 1
            if match_fixed(rec["prediction"], " " + rec["object"], rec):
                ok_n += 1

    same_pct = same_n / total_n * 100 if total_n else 0
    nonenl_acc = ok_n / total_n * 100 if total_n else 0
    en_acc = en_ok_n / en_total * 100 if en_total else 0
    n_unique = len(unique_en_answers)
    avg_freq = len(all_en_answers) / n_unique if n_unique else 0

    print(f"\n  {rel}:")
    print(f"    EN accuracy:         {en_acc:.1f}%")
    print(f"    Non-EN accuracy:     {nonenl_acc:.1f}%")
    print(f"    EN gap:              {en_acc - nonenl_acc:+.1f} pp")
    print(f"    Same first token:    {same_pct:.1f}% of (fact x lang) pairs")
    print(f"    Unique EN answers:   {n_unique}")
    print(f"    Avg answer freq:     {avg_freq:.1f} (facts per unique answer)")

# =====================================================================
# Figure: same-token vs diff-token accuracy by language
# =====================================================================
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(NON_EN))
w = 0.35
sa = [same_tok_acc[l]["ok"]/same_tok_acc[l]["n"]*100 if same_tok_acc[l]["n"] else 0 for l in NON_EN]
da = [diff_tok_acc[l]["ok"]/diff_tok_acc[l]["n"]*100 if diff_tok_acc[l]["n"] else 0 for l in NON_EN]
ax.bar(x - w/2, sa, w, label="Same first token as EN", color="#5cb85c", alpha=0.8)
ax.bar(x + w/2, da, w, label="Different first token from EN", color="#d9534f", alpha=0.8)
ax.axhline(79.2, color="black", linestyle="--", alpha=0.5, label="EN accuracy (79.2%)")
ax.set_xticks(x); ax.set_xticklabels(NON_EN, fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Non-EN accuracy: when the answer token is the SAME as EN vs DIFFERENT\n"
             "(Explains why EN advantage is small: most answers share tokens)", fontsize=11)
ax.legend(fontsize=10); ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "same_vs_diff_token_accuracy.png"), dpi=150)
plt.close(fig)
print(f"\nSaved {OUT_DIR}/same_vs_diff_token_accuracy.png")
