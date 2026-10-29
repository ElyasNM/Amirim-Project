"""Rescue rates when early layers are excluded (answers: is the subject-patching effect all in L1?).
Reads existing per-layer patching outputs; no model run needed.
Denominator = all KLAR failures; failures without patch data count as not rescued (same as rescue_form_breakdown.py)."""
import json, glob, os, sys
sys.stdout.reconfigure(encoding="utf-8")
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import CACHE_ROOT
BASE = str(CACHE_ROOT)
MODELS = {"Llama-3.1-8B": "meta-llama_Llama-3.1-8B_section6_subj_act_patch",
          "Qwen2.5-7B": "Qwen_Qwen2.5-7B_section6_subj_act_patch"}
BANDS = {"any layer": lambda l: True, "L>=3": lambda l: l >= 3, "L>=6": lambda l: l >= 6,
         "L10-20": lambda l: 10 <= l <= 20, "L>=21": lambda l: l >= 21}

for name, d in MODELS.items():
    recs, n_all = [], 0
    for f in glob.glob(os.path.join(BASE, d, "examples", "*.json")):
        for lang, e in json.load(open(f, encoding="utf-8")).items():
            if isinstance(e, dict) and e.get("status") == "fail_en_ok":
                n_all += 1
                if e.get("patch_per_layer"):
                    recs.append(e["patch_per_layer"])
    n = n_all
    print(f"\n{name}: n={n} failures ({n - len(recs)} without patch data), layers {recs[0][0]['layer']}..{recs[0][-1]['layer']}")
    print(f"  {'band':10s} {'any':>6s} {'tgt-form':>9s} {'EN-form':>8s}")
    for band, ok in BANDS.items():
        a = sum(any(p["rescue"] for p in r if ok(p["layer"])) for r in recs)
        t = sum(any(p["match_tgt"] for p in r if ok(p["layer"])) for r in recs)
        e = sum(any(p["match_en"] for p in r if ok(p["layer"])) for r in recs)
        print(f"  {band:10s} {100*a/n:6.1f} {100*t/n:9.1f} {100*e/n:8.1f}")
    resc = [r for r in recs if any(p["rescue"] for p in r)]
    only12 = sum(not any(p["rescue"] for p in r if p["layer"] >= 3) for r in resc)
    print(f"  rescued only at L1-2: {only12}/{len(resc)} = {100*only12/len(resc):.1f}% of rescued")
