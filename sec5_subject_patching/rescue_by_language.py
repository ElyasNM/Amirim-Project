"""Per-language rescue rate (Llama-3.1-8B), split by answer form. Denominator: all failures (no patch data = not rescued)."""
import json, glob, os, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import CACHE_ROOT
BASE = str(CACHE_ROOT)
s = defaultdict(lambda: [0, 0, 0, 0])  # n, any, tgt, en
for f in glob.glob(os.path.join(BASE, "meta-llama_Llama-3.1-8B_section6_subj_act_patch", "examples", "*.json")):
    for lang, e in json.load(open(f, encoding="utf-8")).items():
        if not (isinstance(e, dict) and e.get("status") == "fail_en_ok"): continue
        p = e.get("patch_per_layer") or []; x = s[lang]; x[0] += 1
        x[1] += any(q["rescue"] for q in p); x[2] += any(q["match_tgt"] for q in p); x[3] += any(q["match_en"] for q in p)
print(f"{'lang':5}{'n':>5}{'any%':>7}{'tgt%':>7}{'en%':>7}")
for l, (n, a, t, e) in sorted(s.items(), key=lambda kv: -kv[1][1] / kv[1][0]):
    print(f"{l:5}{n:>5}{100*a/n:7.1f}{100*t/n:7.1f}{100*e/n:7.1f}")
