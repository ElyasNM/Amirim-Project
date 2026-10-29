"""
run_multimodel_analysis.py
--------------------------
Runs analyses 01-04 (truncation bug, EN bias, logit lens, same-token)
on any model's cached data. Parameterized by model directory.

Usage:
    python run_multimodel_analysis.py <cache_dir_name>
    e.g.:  python run_multimodel_analysis.py allenai_OLMo-2-1124-7B_section6
"""

import os, sys, json, glob, unicodedata
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Args ───────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python run_multimodel_analysis.py <cache_dir_name>")
    sys.exit(1)

CACHE_DIR_NAME = sys.argv[1]
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import DATA_ROOT
BASE_DIR = str(DATA_ROOT)
CACHE_DIR = os.path.join(BASE_DIR, "cache", CACHE_DIR_NAME, "examples")

# Read metadata for model name
meta_path = os.path.join(BASE_DIR, "cache", CACHE_DIR_NAME, "metadata.json")
with open(meta_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)
MODEL_NAME = metadata["model_name"]
NUM_LAYERS = metadata["num_layers"]
SHORT_NAME = MODEL_NAME.split("/")[-1]

# Output dir
OUT_DIR = os.path.join(BASE_DIR, "results", CACHE_DIR_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

print(f"=" * 70)
print(f"MULTI-MODEL ANALYSIS: {MODEL_NAME}")
print(f"Cache: {CACHE_DIR}")
print(f"Output: {OUT_DIR}")
print(f"Layers: {NUM_LAYERS}")
print(f"=" * 70)

# ── Constants ──────────────────────────────────────────────────────────────
LANGS = ["ca", "en", "es", "fr", "hu", "ja", "ko", "nl", "ru", "uk", "vi", "zh"]
NON_EN = [l for l in LANGS if l != "en"]

GEO_RELS = {
    "place_of_birth", "place_of_death",
    "headquarters_location", "location_of_formation",
    "capital", "capital_of",
}


# ── Matching ───────────────────────────────────────────────────────────────
def norm(s):
    return unicodedata.normalize("NFC", s.lower().strip())

def _pfx(a, b):
    a, b = norm(a), norm(b)
    return len(a) > 0 and b.startswith(a)

def match_original(pred, target):
    return _pfx(pred, target) or _pfx(target, pred)

def match_fixed(pred, target, rec):
    if _pfx(pred, target) or _pfx(target, pred):
        return True
    p, t = norm(pred), norm(target)
    ll = rec.get("logit_lens", {})
    ptid = ll.get("predicted_token_id")
    ctid = ll.get("correct_object_token_id")
    if len(p) > 4 and t.endswith(p) and p != t:
        return True
    if len(p) > 2 and t.endswith(p) and p != t and ptid is not None and ptid == ctid:
        return True
    if 0 < len(p) <= 2 and ptid is not None and ptid == ctid:
        return True
    return False


# ── Data loading ───────────────────────────────────────────────────────────
def load_facts():
    facts = {}
    for path in glob.glob(os.path.join(CACHE_DIR, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        key = (data["index"], data["relation"])
        facts[key] = {}
        for lang in (k for k in data if k not in ("index", "relation") and isinstance(data[k], dict)):
            if lang in LANGS:
                facts[key][lang] = data[lang]
    return facts


print("\nLoading facts...")
facts = load_facts()
print(f"Loaded {len(facts)} facts")

# Also detect extra languages beyond LANGS
extra_langs = set()
for path in glob.glob(os.path.join(CACHE_DIR, "*.json"))[:5]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for k in data:
        if k not in ("index", "relation") and isinstance(data[k], dict) and k not in LANGS:
            extra_langs.add(k)
if extra_langs:
    print(f"Extra languages available (not in core 12): {sorted(extra_langs)}")

# ── Results collector (for cross-model comparison later) ───────────────────
results = {"model": MODEL_NAME, "short_name": SHORT_NAME, "num_layers": NUM_LAYERS}


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 01: Truncation Bug Impact
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 01: TRUNCATION BUG IMPACT")
print("=" * 70)

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
print(f"Total matches (original): {total_matches}")
print(f"Total failures (original): {total_failures}")
print()
print(f"{'Lang':<6} {'Fail':>6} {'Trunc':>6} {'Empty':>6} {'Real':>6} {'Trunc%':>8}")
print("-" * 44)
for lang in LANGS:
    s = stats[lang]
    t = s["truncated"]; e = s["empty"]; g = s["genuine"]; tot = s["total_fail"]
    pct = t / tot * 100 if tot else 0
    print(f"{lang:<6} {tot:>6} {t:>6} {e:>6} {g:>6} {pct:>7.1f}%")

# Accuracy: original vs fixed
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
acc_orig = {}
acc_fixed = {}
for lang in LANGS:
    t = total[lang]
    o = corr_o[lang] / t * 100
    fx = corr_f[lang] / t * 100
    r = corr_f[lang] - corr_o[lang]
    acc_orig[lang] = o
    acc_fixed[lang] = fx
    print(f"{lang:<6} {o:>9.1f}% {fx:>9.1f}% {r:>+10}")

results["accuracy_original"] = dict(acc_orig)
results["accuracy_fixed"] = dict(acc_fixed)
results["total_truncation_recovered"] = sum(corr_f[l] - corr_o[l] for l in LANGS)

# Figure: accuracy bars
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(LANGS))
w = 0.5
fix_accs = [acc_fixed[l] for l in LANGS]
b2 = ax.bar(x, fix_accs, w, label="Truncation-fixed", color="#5cb85c", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(LANGS, fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title(f"Per-Language Accuracy ({SHORT_NAME} on KLAR)", fontsize=12)
ax.legend(fontsize=11); ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)
for b in b2: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "01_corrected_accuracy.png"), dpi=150)
plt.close(fig)
print(f"\nSaved {OUT_DIR}/01_corrected_accuracy.png")

# Histogram
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
fig.suptitle(f"Failure Distribution ({SHORT_NAME}): How many languages fail per fact?", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "01_corrected_histogram.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}/01_corrected_histogram.png")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 02: English Bias Test
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 02: ENGLISH BIAS TEST")
print("=" * 70)

fail = defaultdict(set)
corr = defaultdict(int)
total2 = defaultdict(int)

for key, ld in facts.items():
    for lang, rec in ld.items():
        total2[lang] += 1
        target = " " + rec["object"]
        if match_fixed(rec["prediction"], target, rec):
            corr[lang] += 1
        else:
            fail[lang].add(key)

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

results["en_correct"] = len(en_ok)
results["en_correct_pct"] = len(en_ok) / len(facts) * 100
results["en_bias_dist"] = {n: dist.get(n, 0) for n in range(12)}
results["zero_nonen_fail_pct"] = dist.get(0, 0) / len(en_ok) * 100 if len(en_ok) > 0 else 0

# EN bias figure
fig, ax = plt.subplots(figsize=(10, 5))
xs = list(range(12))
ys = [dist.get(n, 0) for n in xs]
colors = ["#5cb85c" if n <= 3 else "#f0ad4e" if n <= 7 else "#d9534f" for n in xs]
ax.bar(xs, ys, color=colors, edgecolor="white")
ax.set_xlabel("#Non-English languages that also fail", fontsize=11)
ax.set_ylabel("#Facts (where EN is correct)", fontsize=11)
ax.set_title(f"Among facts EN gets RIGHT: how many other langs fail? ({SHORT_NAME})\n"
             "(If English-biased, we'd see mass at 8-11)", fontsize=11)
ax.set_xticks(xs)
for i, v in enumerate(ys):
    if len(en_ok) > 0:
        ax.text(i, v + 5, f"{v/len(en_ok)*100:.0f}%", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_english_bias_test.png"), dpi=150)
plt.close(fig)
print(f"\nSaved {OUT_DIR}/02_english_bias_test.png")

# Per-relation EN gap
rel_stats = defaultdict(lambda: defaultdict(lambda: {"ok": 0, "n": 0}))
for key, ld in facts.items():
    for lang, rec in ld.items():
        target = " " + rec["object"]
        rel_stats[key[1]][lang]["n"] += 1
        if match_fixed(rec["prediction"], target, rec):
            rel_stats[key[1]][lang]["ok"] += 1

gaps = []
for rel in sorted(rel_stats):
    en_a = rel_stats[rel]["en"]["ok"] / rel_stats[rel]["en"]["n"] * 100 if rel_stats[rel]["en"]["n"] > 0 else 0
    ne = [rel_stats[rel][l]["ok"] / rel_stats[rel][l]["n"] * 100 for l in NON_EN if rel_stats[rel][l]["n"] > 0]
    ne_avg = sum(ne) / len(ne) if ne else 0
    gaps.append((rel, en_a - ne_avg, en_a, ne_avg))

gaps.sort(key=lambda x: x[1])

print(f"\n{'Relation':<28} {'EN%':>6} {'NonEN%':>8} {'Gap':>8}")
print("-" * 54)
for rel, gap, ea, na in sorted(gaps, key=lambda x: -x[1]):
    print(f"{rel:<28} {ea:>5.1f}% {na:>7.1f}% {gap:>+7.1f}%")

results["en_gap_by_relation"] = {rel: {"en": ea, "nonen": na, "gap": gap} for rel, gap, ea, na in gaps}

# EN gap figure
fig, ax = plt.subplots(figsize=(10, 7))
rels = [g[0] for g in gaps]
vals = [g[1] for g in gaps]
colors = ["#d9534f" if v > 10 else "#5cb85c" if v < 0 else "#f0ad4e" for v in vals]
ax.barh(range(len(rels)), vals, color=colors, edgecolor="white")
ax.set_yticks(range(len(rels))); ax.set_yticklabels(rels, fontsize=9)
ax.set_xlabel("EN accuracy - avg non-EN accuracy (pp)", fontsize=11)
ax.set_title(f"English Advantage by Relation Type ({SHORT_NAME})", fontsize=11)
ax.axvline(0, color="black", linewidth=0.5)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_en_gap_by_relation.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}/02_en_gap_by_relation.png")

# Jaccard heatmap
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
ax.set_title(f"Pairwise Jaccard Similarity of Failures ({SHORT_NAME})", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_jaccard_fixed.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}/02_jaccard_fixed.png")

# Jaccard heatmap — correct overlap (success alignment, like Fig 2 in arxiv 2504.04264)
correct = {la: set(facts.keys()) - fail[la] for la in LANGS}
jaccard_corr = np.zeros((n, n))
for i, la in enumerate(LANGS):
    for j, lb in enumerate(LANGS):
        inter = len(correct[la] & correct[lb])
        union = len(correct[la] | correct[lb])
        jaccard_corr[i, j] = inter / union if union else 0

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(jaccard_corr, vmin=0, vmax=1, cmap="RdYlGn")
ax.set_xticks(range(n)); ax.set_xticklabels(LANGS, rotation=45, ha="right")
ax.set_yticks(range(n)); ax.set_yticklabels(LANGS)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{jaccard_corr[i,j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax, label="Jaccard similarity of correct sets")
ax.set_title(f"Pairwise Jaccard Similarity of Correct Predictions ({SHORT_NAME})", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_jaccard_correct_overlap.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}/02_jaccard_correct_overlap.png")

# Directed correct-transfer matrix: row=source, cell(i,j) = |correct[i]∩correct[j]| / |correct[i]|
transfer = np.zeros((n, n))
for i, la in enumerate(LANGS):
    for j, lb in enumerate(LANGS):
        transfer[i, j] = len(correct[la] & correct[lb]) / len(correct[la]) if correct[la] else 0

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(transfer, vmin=0, vmax=1, cmap="RdYlGn")
ax.set_xticks(range(n)); ax.set_xticklabels(LANGS, rotation=45, ha="right")
ax.set_yticks(range(n)); ax.set_yticklabels(LANGS)
ax.set_xlabel("Target language (Y)"); ax.set_ylabel("Source language (X)")
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{transfer[i,j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax, label="Fraction of X-correct facts also correct in Y")
ax.set_title(f"Correct Transfer: of X's correct facts, fraction also correct in Y\n({SHORT_NAME})", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_correct_transfer.png"), dpi=150)
plt.close(fig)
print(f"Saved {OUT_DIR}/02_correct_transfer.png")

# Top Jaccard pairs
results["jaccard_top_pairs"] = {}
for i, la in enumerate(LANGS):
    for j, lb in enumerate(LANGS):
        if i < j:
            results["jaccard_top_pairs"][f"{la}-{lb}"] = round(jaccard[i, j], 3)


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 03: Logit Lens Analysis
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 03: LOGIT LENS ANALYSIS")
print("=" * 70)

groups = defaultdict(list)
groups_all = defaultdict(list)

for key, ld in facts.items():
    en_rec = ld.get("en")
    if not en_rec:
        continue
    en_ok_flag = match_fixed(en_rec["prediction"], " " + en_rec["object"], en_rec)
    rel_type = "geo" if key[1] in GEO_RELS else "nongeo"

    for lang in NON_EN:
        rec = ld.get(lang)
        if not rec:
            continue
        ll = rec.get("logit_lens", {})
        cor = ll.get("correct_object_ranks", [])
        enr = ll.get("correct_en_ranks", [])
        pdr = ll.get("predicted_token_ranks", [])
        if len(cor) != NUM_LAYERS or len(enr) != NUM_LAYERS or len(pdr) != NUM_LAYERS:
            continue

        entry = {
            "cor": cor, "enr": enr, "pdr": pdr,
            "lang": lang, "rel": key[1],
            "pred_tok": ll.get("predicted_token", ""),
            "en_tok": ll.get("correct_en_token", ""),
            "obj_tok": ll.get("correct_object_token", ""),
        }

        is_correct = match_fixed(rec["prediction"], " " + rec["object"], rec)
        if is_correct:
            status = "success"
        elif en_ok_flag:
            status = "fail_en_ok"
        else:
            status = "fail_en_bad"

        groups[(status, rel_type)].append(entry)
        groups_all[status].append(entry)

for g, entries in sorted(groups_all.items()):
    print(f"{g}: {len(entries)} records")

print("\n=== FINAL LAYER MEAN RANKS ===")
print(f"{'Group':<25} {'n':>6} {'Target':>8} {'EN':>8} {'Pred':>8}")
print("-" * 60)
for rtype in ["geo", "nongeo"]:
    for status in ["success", "fail_en_ok", "fail_en_bad"]:
        entries = groups[(status, rtype)]
        if not entries:
            continue
        ct = np.mean([e["cor"][-1] for e in entries])
        et = np.mean([e["enr"][-1] for e in entries])
        pt = np.mean([e["pdr"][-1] for e in entries])
        print(f"{rtype:>6}_{status:<18} {len(entries):>6} {ct:>8.1f} {et:>8.1f} {pt:>8.1f}")

# Layer-by-layer
succ = groups_all["success"]
fail_group = groups_all["fail_en_ok"]

if succ and fail_group:
    cor_s = np.mean([e["cor"] for e in succ], axis=0)
    enr_s = np.mean([e["enr"] for e in succ], axis=0)
    cor_f = np.mean([e["cor"] for e in fail_group], axis=0)
    enr_f = np.mean([e["enr"] for e in fail_group], axis=0)

    print(f"\n=== LAYER-BY-LAYER RANKS (layers {NUM_LAYERS-12}-{NUM_LAYERS}) ===")
    print(f"{'Layer':<6} {'S_Tgt':>10} {'S_EN':>10} {'F_Tgt':>10} {'F_EN':>10}")
    print("-" * 50)
    start_layer = max(0, NUM_LAYERS - 13)
    for i in range(start_layer, NUM_LAYERS):
        print(f"{i+1:<6} {cor_s[i]:>10.0f} {enr_s[i]:>10.0f} {cor_f[i]:>10.0f} {enr_f[i]:>10.0f}")

    # Crossover
    for label, c, e in [("Success", cor_s, enr_s), ("Fail_en_ok", cor_f, enr_f)]:
        for i in range(15, NUM_LAYERS):
            if c[i] < e[i]:
                print(f"\n{label}: target overtakes EN at layer {i+1} (tgt={c[i]:.0f}, en={e[i]:.0f})")
                break

    results["crossover_success"] = None
    results["crossover_fail"] = None
    for i in range(15, NUM_LAYERS):
        if cor_s[i] < enr_s[i] and results["crossover_success"] is None:
            results["crossover_success"] = i + 1
        if cor_f[i] < enr_f[i] and results["crossover_fail"] is None:
            results["crossover_fail"] = i + 1

    # Failure characterization
    en_fallback = 0
    for e in fail_group:
        pt = norm(e["pred_tok"])
        et = norm(e["en_tok"])
        if pt and et and (pt.startswith(et) or et.startswith(pt)):
            en_fallback += 1

    total_f = len(fail_group)
    near_miss = sum(1 for e in fail_group if e["cor"][-1] <= 5)
    near20 = sum(1 for e in fail_group if e["cor"][-1] <= 20)

    print(f"\n=== FAILURE CHARACTERIZATION (fail_en_ok, n={total_f}) ===")
    print(f"  English fallback:     {en_fallback} ({en_fallback/total_f*100:.1f}%)")
    print(f"  Something else:       {total_f - en_fallback} ({(total_f-en_fallback)/total_f*100:.1f}%)")
    print(f"  Target at rank 0-5:   {near_miss} ({near_miss/total_f*100:.1f}%)")
    print(f"  Target at rank 0-20:  {near20} ({near20/total_f*100:.1f}%)")

    results["en_fallback_pct"] = en_fallback / total_f * 100
    results["near_miss_pct"] = near_miss / total_f * 100
    results["near_miss_20_pct"] = near20 / total_f * 100
    results["final_layer_target_rank_fail"] = float(np.mean([e["cor"][-1] for e in fail_group]))
    results["final_layer_en_rank_fail"] = float(np.mean([e["enr"][-1] for e in fail_group]))
    results["final_layer_target_rank_success"] = float(np.mean([e["cor"][-1] for e in succ]))

    # Figure 1: success vs failure
    pdr_s = np.mean([e["pdr"] for e in succ], axis=0)
    pdr_f = np.mean([e["pdr"] for e in fail_group], axis=0)

    layers = np.arange(1, NUM_LAYERS + 1)
    z = slice(max(0, NUM_LAYERS - 13), None)
    x_plot = layers[z]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    for ax, c, e, p, title, nn in [
        (ax1, cor_s, enr_s, pdr_s, "SUCCESS", len(succ)),
        (ax2, cor_f, enr_f, pdr_f, "FAILURE (EN correct)", len(fail_group)),
    ]:
        ax.semilogy(x_plot, np.array(c[z]) + 1, "o-", color="steelblue", lw=2.5, ms=5, label="Correct target-lang token")
        ax.semilogy(x_plot, np.array(e[z]) + 1, "s--", color="seagreen", lw=2.5, ms=5, label="Correct EN token")
        ax.semilogy(x_plot, np.array(p[z]) + 1, "^:", color="firebrick", lw=2.5, ms=5, label="Actually predicted token")
        ax.invert_yaxis()
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_title(f"{title} (n={nn})", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("Mean Rank+1 (log, lower = more probable)", fontsize=11)
    fig.suptitle(
        f"Logit Lens: Language Transition ({SHORT_NAME})\n"
        "EN token rises first, target-lang follows",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_logit_lens_definitive.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved {OUT_DIR}/03_logit_lens_definitive.png")

    # Figure 2: geo vs nongeo
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey="row")
    for row, rtype in enumerate(["geo", "nongeo"]):
        for col, status in enumerate(["success", "fail_en_ok", "fail_en_bad"]):
            ax = axes[row][col]
            entries = groups[(status, rtype)]
            if not entries:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue
            c = np.mean([e["cor"] for e in entries], axis=0)
            e_arr = np.mean([e["enr"] for e in entries], axis=0)
            p = np.mean([e["pdr"] for e in entries], axis=0)
            ax.semilogy(layers, c + 1, color="steelblue", lw=2.5, label="Correct target-lang")
            ax.semilogy(layers, e_arr + 1, color="seagreen", lw=2.5, ls="--", label="Correct EN")
            ax.semilogy(layers, p + 1, color="firebrick", lw=2.5, ls=":", label="Predicted (actual)")
            ax.invert_yaxis()
            rtype_label = "GEOGRAPHIC" if rtype == "geo" else "NON-GEOGRAPHIC"
            status_label = {"success": "SUCCESS", "fail_en_ok": "FAIL (EN ok)", "fail_en_bad": "FAIL (EN bad)"}
            ax.set_title(f"{rtype_label}\n{status_label[status]} (n={len(entries)})", fontsize=10)
            ax.set_xlabel("Layer")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Mean Rank+1 (log)")
    axes[1][0].set_ylabel("Mean Rank+1 (log)")
    fig.suptitle(f"Logit Lens by Relation Type ({SHORT_NAME})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "03_logit_lens_geo_vs_nongeo.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/03_logit_lens_geo_vs_nongeo.png")
else:
    print("WARNING: Not enough data for logit lens analysis")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 04: Why EN Advantage is Small (Same-Token Analysis)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS 04: SAME-TOKEN vs DIFFERENT-TOKEN")
print("=" * 70)

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

print(f"{'Lang':<6} {'Same#':>6} {'Same%':>8} {'SameAcc':>9}  {'Diff#':>6} {'DiffAcc':>9}  {'Gap':>7}")
print("-" * 65)
for lang in NON_EN:
    sn = same_tok_acc[lang]["n"]
    so = same_tok_acc[lang]["ok"]
    sa = so / sn * 100 if sn else 0
    dn = diff_tok_acc[lang]["n"]
    do_ = diff_tok_acc[lang]["ok"]
    da = do_ / dn * 100 if dn else 0
    frac = sn / (sn + dn) * 100 if (sn + dn) else 0
    print(f"{lang:<6} {sn:>6} {frac:>7.1f}% {sa:>8.1f}%  {dn:>6} {da:>8.1f}%  {sa-da:>+6.1f}%")

en_corr = sum(1 for k, ld in facts.items()
              if "en" in ld and match_fixed(ld["en"]["prediction"], " " + ld["en"]["object"], ld["en"]))
print(f"\nEN accuracy: {en_corr}/{len(facts)} = {en_corr/len(facts)*100:.1f}%")

all_same_ok = sum(same_tok_acc[l]["ok"] for l in NON_EN)
all_same_n = sum(same_tok_acc[l]["n"] for l in NON_EN)
all_diff_ok = sum(diff_tok_acc[l]["ok"] for l in NON_EN)
all_diff_n = sum(diff_tok_acc[l]["n"] for l in NON_EN)

same_acc_all = all_same_ok / all_same_n * 100 if all_same_n else 0
diff_acc_all = all_diff_ok / all_diff_n * 100 if all_diff_n else 0
print(f"\nALL non-EN same-token: {all_same_ok}/{all_same_n} = {same_acc_all:.1f}%")
print(f"ALL non-EN diff-token: {all_diff_ok}/{all_diff_n} = {diff_acc_all:.1f}%")
print(f"Gap: {same_acc_all - diff_acc_all:+.1f} pp")
print(f"Same-token fraction: {all_same_n / (all_same_n + all_diff_n) * 100:.1f}%")

results["same_token_acc"] = same_acc_all
results["diff_token_acc"] = diff_acc_all
results["same_token_fraction"] = all_same_n / (all_same_n + all_diff_n) * 100 if (all_same_n + all_diff_n) > 0 else 0

# Per-geo-relation
print("\nPer geographic relation:")
geo_rels_ordered = ["place_of_birth", "place_of_death", "headquarters_location",
                    "location_of_formation", "capital", "capital_of"]
for rel in geo_rels_ordered:
    total_n = 0; same_n = 0; ok_n = 0; en_ok_n = 0; en_total = 0
    unique_en = set(); all_en = []
    for key, ld in facts.items():
        if key[1] != rel:
            continue
        en_rec = ld.get("en")
        if not en_rec:
            continue
        unique_en.add(en_rec["object"].lower().strip())
        all_en.append(en_rec["object"].lower().strip())
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
    if total_n == 0:
        continue
    same_pct = same_n / total_n * 100
    nonenl_acc = ok_n / total_n * 100
    en_acc = en_ok_n / en_total * 100 if en_total else 0
    print(f"  {rel}: EN={en_acc:.1f}% NonEN={nonenl_acc:.1f}% gap={en_acc-nonenl_acc:+.1f}pp same_tok={same_pct:.1f}% unique={len(unique_en)}")

# Figure
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(NON_EN))
w = 0.35
sa = [same_tok_acc[l]["ok"]/same_tok_acc[l]["n"]*100 if same_tok_acc[l]["n"] else 0 for l in NON_EN]
da = [diff_tok_acc[l]["ok"]/diff_tok_acc[l]["n"]*100 if diff_tok_acc[l]["n"] else 0 for l in NON_EN]
en_acc_val = en_corr / len(facts) * 100
ax.bar(x - w/2, sa, w, label="Same first token as EN", color="#5cb85c", alpha=0.8)
ax.bar(x + w/2, da, w, label="Different first token from EN", color="#d9534f", alpha=0.8)
ax.axhline(en_acc_val, color="black", linestyle="--", alpha=0.5, label=f"EN accuracy ({en_acc_val:.1f}%)")
ax.set_xticks(x); ax.set_xticklabels(NON_EN, fontsize=11)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title(f"Same-Token vs Different-Token Accuracy ({SHORT_NAME})\n"
             "(Explains why EN advantage is small)", fontsize=11)
ax.legend(fontsize=10); ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "04_same_vs_diff_token_accuracy.png"), dpi=150)
plt.close(fig)
print(f"\nSaved {OUT_DIR}/04_same_vs_diff_token_accuracy.png")


# ═══════════════════════════════════════════════════════════════════════════
# Save results JSON for cross-model comparison
# ═══════════════════════════════════════════════════════════════════════════
results_path = os.path.join(OUT_DIR, "results_summary.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n{'=' * 70}")
print(f"ALL DONE. Results saved to {results_path}")
print(f"Figures saved to {OUT_DIR}/")
print(f"{'=' * 70}")
