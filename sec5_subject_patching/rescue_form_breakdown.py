"""Exact rescue breakdown by answer form (EN vs target), incl. exclusive categories.
Denominator = all KLAR failures (fail_en_ok); records without patch data count as not rescued."""
import json, glob, os, sys
sys.stdout.reconfigure(encoding="utf-8")
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import CACHE_ROOT
BASE = str(CACHE_ROOT)
for name, d in {"Llama-3.1-8B": "meta-llama_Llama-3.1-8B_section6_subj_act_patch",
                "Qwen2.5-7B": "Qwen_Qwen2.5-7B_section6_subj_act_patch"}.items():
    c = dict(n=0, nodata=0, any=0, en=0, tgt=0, both=0, en_only=0, tgt_only=0, same_form=0, rescue_ne_or=0)
    for f in glob.glob(os.path.join(BASE, d, "examples", "*.json")):
        for lang, e in json.load(open(f, encoding="utf-8")).items():
            if not isinstance(e, dict) or e.get("status") != "fail_en_ok": continue
            c["n"] += 1; P = e.get("patch_per_layer") or []
            if not P: c["nodata"] += 1; continue
            a = any(p["rescue"] for p in P); en = any(p["match_en"] for p in P); tg = any(p["match_tgt"] for p in P)
            c["rescue_ne_or"] += any(p["rescue"] != (p["match_en"] or p["match_tgt"]) for p in P)
            c["same_form"] += e.get("object_en", "").strip().lower() == e.get("object_tgt", "").strip().lower()
            c["any"] += a; c["en"] += en; c["tgt"] += tg; c["both"] += en and tg
            c["en_only"] += en and not tg; c["tgt_only"] += tg and not en
    n = c["n"]; pct = lambda k: f"{100*c[k]/n:5.1f}%"
    print(f"\n{name}: n={n} failures ({c['nodata']} without patch data)")
    for k in ["any", "tgt", "en", "both", "tgt_only", "en_only"]: print(f"  {k:9s} {c[k]:5d}  {pct(k)}")
    print(f"  not rescued {n-c['any']:5d}  {100*(n-c['any'])/n:5.1f}%")
    print(f"  [checks] layers where rescue != (match_en or match_tgt): {c['rescue_ne_or']} records; EN object == target object: {c['same_form']}")
