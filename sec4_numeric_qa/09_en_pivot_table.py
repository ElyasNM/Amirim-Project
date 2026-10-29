"""Numeric-QA table (paper: tab:numeric).
English-pivoted as in KLAR: among items answered correctly in English, the share of (item, language) pairs
answered incorrectly in the other language. Items come from MKQA (native) and Mintaka."""
import json, os, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import NUMERIC_ROOT
RES = str(NUMERIC_ROOT / "results")
NON_LATIN = {"ar", "he", "ja", "km", "ko", "ru", "th", "zh_cn", "zh_hk", "zh_tw"}
SOURCES = ("mkqa", "mintaka")
for model in ["llama2-7b", "qwen2.5-7b"]:
    corr, src = defaultdict(dict), {}
    for line in open(os.path.join(RES, f"{model}_results.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        k = (r["source"], r["source_id"]); corr[k][r["lang"]] = r["correct"]; src[k] = r["source"]
    en_ok = [k for k, v in corr.items() if v.get("en")]
    langs = sorted({l for v in corr.values() for l in v})
    def rate(keys, keep):
        pairs = [(k, l) for k in keys for l in corr[k] if l != "en" and keep(l)]
        return 100 * sum(not corr[k][l] for k, l in pairs) / len(pairs), len(pairs)
    allr = rate(en_ok, lambda l: True); lat = rate(en_ok, lambda l: l not in NON_LATIN)
    nl = rate(en_ok, lambda l: l in NON_LATIN); nat = rate([k for k in en_ok if src[k] == "mkqa"], lambda l: True)
    by_src = {s: sum(src[k] == s for k in corr) for s in SOURCES}
    print(f"{model}: items {len(corr)} {by_src} langs {len(langs)} | EN acc {100*len(en_ok)/len(corr):.1f} | "
          f"EN-correct {len(en_ok)} (native {sum(src[k]=='mkqa' for k in en_ok)}) | fail all {allr[0]:.1f} | "
          f"Latin {lat[0]:.1f} | non-Latin {nl[0]:.1f} | native-only {nat[0]:.1f}")
