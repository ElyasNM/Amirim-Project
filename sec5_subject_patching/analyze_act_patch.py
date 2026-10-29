"""
Activation patching analysis: cross-language subject patching results.

Hypothesis: If the model fails in language L but succeeds in EN, patching
the EN subject representation into the L forward should "rescue" the prediction
(produce the correct object). High rescue rate => EN subject encoding is richer.
"""
import json
import os
import glob
import sys
import io
from collections import defaultdict
import math

# force UTF-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─── Config ───────────────────────────────────────────────────────────────────
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import CACHE_ROOT
BASE = str(CACHE_ROOT)
MODELS = {
    "Llama-3.1-8B": os.path.join(BASE, "meta-llama_Llama-3.1-8B_section6_subj_act_patch", "examples"),
    "Qwen2.5-7B":   os.path.join(BASE, "Qwen_Qwen2.5-7B_section6_subj_act_patch",          "examples"),
}
N_LAYERS = 32   # will be inferred per-file too


def load_all(examples_dir):
    records = []
    for fpath in glob.glob(os.path.join(examples_dir, "*.json")):
        with open(fpath, encoding="utf-8") as f:
            d = json.load(f)
        relation = d.get("relation", os.path.basename(fpath).split("__")[-1].replace(".json",""))
        idx = d.get("index", -1)
        for lang, entry in d.items():
            if not isinstance(entry, dict) or "status" not in entry:
                continue
            entry["_relation"] = relation
            entry["_lang"] = lang
            entry["_index"] = idx
            records.append(entry)
    return records


def analyse(model_name, examples_dir):
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}   ({examples_dir})")
    print(f"{'='*70}")

    records = load_all(examples_dir)
    print(f"  Total (fact, lang) pairs loaded: {len(records)}")
    if not records:
        print("  No data!")
        return

    n_layers = len(records[0].get("patch_per_layer", [])) or N_LAYERS

    # ── 1. Overall rescue rate ─────────────────────────────────────────────
    total = len(records)
    any_rescue     = sum(1 for r in records if r.get("rescue_summary", {}).get("any_rescue", False))
    rescue_en_only = sum(1 for r in records if r.get("rescue_summary", {}).get("any_rescue_en", False))
    rescue_tgt_only= sum(1 for r in records if r.get("rescue_summary", {}).get("any_rescue_tgt", False))

    print(f"\n--- 1. Overall rescue rate (any layer) ---")
    print(f"  any_rescue      : {any_rescue}/{total} = {any_rescue/total*100:.1f}%")
    print(f"  any_rescue_en   : {rescue_en_only}/{total} = {rescue_en_only/total*100:.1f}%")
    print(f"  any_rescue_tgt  : {rescue_tgt_only}/{total} = {rescue_tgt_only/total*100:.1f}%")

    # ── 2. Rescue rate by language ─────────────────────────────────────────
    print(f"\n--- 2. Rescue rate by language ---")
    lang_stats = defaultdict(lambda: {"total":0,"rescue":0,"rescue_en":0,"rescue_tgt":0})
    for r in records:
        lang = r["_lang"]
        rs = r.get("rescue_summary", {})
        lang_stats[lang]["total"] += 1
        if rs.get("any_rescue"):    lang_stats[lang]["rescue"] += 1
        if rs.get("any_rescue_en"): lang_stats[lang]["rescue_en"] += 1
        if rs.get("any_rescue_tgt"):lang_stats[lang]["rescue_tgt"] += 1
    for lang in sorted(lang_stats):
        s = lang_stats[lang]
        t = s["total"]
        print(f"  {lang:6s}: {s['rescue']:4d}/{t} = {s['rescue']/t*100:5.1f}%  "
              f"(en-rescue={s['rescue_en']/t*100:4.1f}%, tgt-rescue={s['rescue_tgt']/t*100:4.1f}%)")

    # ── 3. Rescue rate by relation ─────────────────────────────────────────
    print(f"\n--- 3. Rescue rate by relation ---")
    rel_stats = defaultdict(lambda: {"total":0,"rescue":0})
    for r in records:
        rel = r["_relation"]
        rs = r.get("rescue_summary", {})
        rel_stats[rel]["total"] += 1
        if rs.get("any_rescue"): rel_stats[rel]["rescue"] += 1
    for rel in sorted(rel_stats, key=lambda x: -rel_stats[x]["rescue"]/rel_stats[x]["total"]):
        s = rel_stats[rel]
        t = s["total"]
        print(f"  {rel:35s}: {s['rescue']:4d}/{t:4d} = {s['rescue']/t*100:5.1f}%")

    # ── 4. First rescue layer distribution ────────────────────────────────
    print(f"\n--- 4. First rescue layer distribution ---")
    first_layers = [r["rescue_summary"]["first_rescue_layer"]
                    for r in records
                    if r.get("rescue_summary", {}).get("any_rescue")]
    if first_layers:
        from collections import Counter
        layer_counts = Counter(first_layers)
        print(f"  Early (layers 1-10) : {sum(c for l,c in layer_counts.items() if l<=10)}")
        print(f"  Mid   (layers 11-20): {sum(c for l,c in layer_counts.items() if 11<=l<=20)}")
        print(f"  Late  (layers 21-31): {sum(c for l,c in layer_counts.items() if l>=21)}")
        print(f"  Mean first rescue layer: {sum(first_layers)/len(first_layers):.1f}")
        print(f"  Median:                  {sorted(first_layers)[len(first_layers)//2]}")
        # histogram per layer
        print("  Per-layer count (rescues first appearing at that layer):")
        for layer in range(1, n_layers+1):
            c = layer_counts.get(layer, 0)
            bar = '#' * min(c, 60)
            print(f"    L{layer:2d}: {c:4d}  {bar}")

    # ── 5. Per-layer rescue rate (aggregated over all records) ────────────
    print(f"\n--- 5. Per-layer rescue rate (fraction of cases rescued at that layer) ---")
    layer_rescue = defaultdict(int)
    for r in records:
        for p in r.get("patch_per_layer", []):
            if p.get("rescue"):
                layer_rescue[p["layer"]] += 1
    for layer in range(0, n_layers+1):
        c = layer_rescue.get(layer, 0)
        rate = c / total * 100
        bar = '#' * int(rate / 2)
        print(f"    L{layer:2d}: {c:4d}/{total}  {rate:5.1f}%  {bar}")

    # ── 6. Logit lens — EN vs TGT subject decoding quality ────────────────
    print(f"\n--- 6. Logit lens: object token rank EN-subj vs TGT-subj (mean per layer) ---")
    en_ranks  = defaultdict(list)
    tgt_ranks = defaultdict(list)
    for r in records:
        for layer_idx, lens_e in enumerate(r.get("lens_en_subject", [])):
            en_ranks[layer_idx].append(lens_e["ranks"]["obj_en"])
        for layer_idx, lens_t in enumerate(r.get("lens_tgt_subject", [])):
            tgt_ranks[layer_idx].append(lens_t["ranks"]["obj_en"])  # rank of EN object in TGT-subj lens

    print(f"  {'Layer':>6}  {'mean_rank(EN-subj->obj_en)':>26}  {'mean_rank(TGT-subj->obj_en)':>27}  {'delta(TGT-EN)':>14}")
    for layer in range(n_layers):
        if not en_ranks[layer] or not tgt_ranks[layer]:
            continue
        me = sum(en_ranks[layer]) / len(en_ranks[layer])
        mt = sum(tgt_ranks[layer]) / len(tgt_ranks[layer])
        print(f"  {layer:>6}  {me:>26.0f}  {mt:>28.0f}  {mt-me:>+12.0f}")

    # ── 7. Spot check: cases where EN patches rescue but TGT-subj cannot ──
    print(f"\n--- 7. Spot-check: 10 random cases ---")
    import random
    random.seed(42)
    sample = random.sample(records, min(10, len(records)))
    for r in sample:
        rs = r.get("rescue_summary", {})
        print(f"  [{r['_lang']:4s}] {r.get('subject_en','?')[:30]:30s} | obj={r.get('object_en','?'):15s} "
              f"| pred_unpatched={r.get('prediction_unpatched','?'):12s} "
              f"| rescue={rs.get('any_rescue','?')} layers={rs.get('rescue_layers_en','?')}")


if __name__ == "__main__":
    for model_name, examples_dir in MODELS.items():
        analyse(model_name, examples_dir)
    print("\nDone.")
