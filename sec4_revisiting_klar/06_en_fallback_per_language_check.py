import json, glob, os, sys, unicodedata
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import DATA_ROOT
CACHE = os.path.join(str(DATA_ROOT), "cache", "meta-llama_Llama-3.1-8B_section6", "examples")
def norm(s): return unicodedata.normalize("NFC", (s or "").lower().strip())
def pfx(a, b):
    a, b = norm(a), norm(b); return len(a) > 0 and b.startswith(a)
def match_fixed(pred, target, rec):
    if pfx(pred, target) or pfx(target, pred): return True
    p, t = norm(pred), norm(target); ll = rec.get("logit_lens", {})
    ptid, ctid = ll.get("predicted_token_id"), ll.get("correct_object_token_id")
    if len(p) > 4 and t.endswith(p) and p != t: return True
    if len(p) > 2 and t.endswith(p) and p != t and ptid is not None and ptid == ctid: return True
    if 0 < len(p) <= 2 and ptid is not None and ptid == ctid: return True
    return False
stats = defaultdict(lambda: defaultdict(int)); succ_min = []
for path in glob.glob(os.path.join(CACHE, "*.json")):
    d = json.load(open(path, encoding="utf-8")); en = d.get("en")
    if not isinstance(en, dict) or not match_fixed(en["prediction"], " " + en["object"], en): continue
    for lang, rec in d.items():
        if lang in ("en", "index", "relation") or not isinstance(rec, dict): continue
        ll = rec.get("logit_lens", {}); cor = ll.get("correct_object_ranks", [])
        if not cor: continue
        if match_fixed(rec["prediction"], " " + rec["object"], rec):
            succ_min.append(cor[-1]); continue
        s = stats[lang]; s["n"] += 1
        pt, et, ot = norm(ll.get("predicted_token", "")), norm(ll.get("correct_en_token", "")), norm(ll.get("correct_object_token", ""))
        if pt and et and (pt.startswith(et) or et.startswith(pt)): s["fb_tok"] += 1
        if pfx(en["object"], rec["prediction"]) or pfx(rec["prediction"], en["object"]): s["fb_full"] += 1
        if et and et == ot: s["same_tok"] += 1
        if cor[-1] == min(cor[-1], 0) or cor[-1] == 0: s["rank0"] += 1
        if cor[-1] <= 5: s["le5"] += 1
print("min final rank among successes:", min(succ_min), "share rank==0:", sum(r == 0 for r in succ_min) / len(succ_min))
tot = defaultdict(int)
print(f"{'lang':5}{'n':>6}{'fb_tok%':>9}{'fb_full%':>10}{'rank0%':>8}{'le5%':>7}{'sameTok%':>9}")
for lang in sorted(stats):
    s = stats[lang]
    for k, v in s.items(): tot[k] += v
    print(f"{lang:5}{s['n']:>6}{100*s['fb_tok']/s['n']:>9.1f}{100*s['fb_full']/s['n']:>10.1f}{100*s['rank0']/s['n']:>8.1f}{100*s['le5']/s['n']:>7.1f}{100*s['same_tok']/s['n']:>9.1f}")
s = tot
print(f"{'ALL':5}{s['n']:>6}{100*s['fb_tok']/s['n']:>9.1f}{100*s['fb_full']/s['n']:>10.1f}{100*s['rank0']/s['n']:>8.1f}{100*s['le5']/s['n']:>7.1f}{100*s['same_tok']/s['n']:>9.1f}")
