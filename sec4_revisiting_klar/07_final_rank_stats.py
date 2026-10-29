"""Final-layer rank of the correct target-language token on failures: mean vs median, and top-k shares
(ranks are 0-based, so rank <= 5 means top 6). Same failure set as 06_en_fallback_per_language_check.py."""
import importlib.util, json, glob, os, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
spec = importlib.util.spec_from_file_location("chk", os.path.join(os.path.dirname(__file__), "06_en_fallback_per_language_check.py"))
src = open(spec.origin, encoding="utf-8").read().split("stats = defaultdict")[0]
ns = {"__file__": spec.origin}; exec(src, ns)
ranks = []
for path in glob.glob(os.path.join(ns["CACHE"], "*.json")):
    d = json.load(open(path, encoding="utf-8")); en = d.get("en")
    if not isinstance(en, dict) or not ns["match_fixed"](en["prediction"], " " + en["object"], en): continue
    for lang, rec in d.items():
        if lang in ("en", "index", "relation") or not isinstance(rec, dict): continue
        cor = rec.get("logit_lens", {}).get("correct_object_ranks", [])
        if not cor or ns["match_fixed"](rec["prediction"], " " + rec["object"], rec): continue
        ranks.append(cor[-1])
r = np.array(ranks)
print(f"n={len(r)} mean0={r.mean():.1f} (mean rank+1={r.mean()+1:.1f}) median0={np.median(r):.0f}")
for k in [1, 5, 6, 10]: print(f"top-{k}: {100*np.mean(r < k):.1f}%")
