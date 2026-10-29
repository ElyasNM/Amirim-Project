"""Apples-to-apples check: subject-token vs last-token sharing (cosine_z transfer_score, Llama-3.1-8B).
Restricts to layers captured in BOTH runs (15-28), same bucket, same pairs."""
import csv, os, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import MLP_ROOT
R = str(MLP_ROOT / "results")
LATIN = {"ca", "en", "es", "fr", "hu", "nl", "vi"}
def load(tag):
    out = {}
    for r in csv.DictReader(open(os.path.join(R, tag, "pairwise_transfer.csv"), encoding="utf-8")):
        if r["metric"] == "cosine_z":
            out[(r["lang1"], r["lang2"], r["scope"], r["bucket"])] = (float(r["transfer_score"]), int(r["n_facts"]))
    return out
S, T = load("meta-llama_Llama-3.1-8B_subj"), load("meta-llama_Llama-3.1-8B")
layers = [f"layer{l}" for l in range(15, 29)]
pairs = sorted({(a, b) for a, b, s, k in S if (a, b, s, k) in T and s in layers})
def avg(D, ps, ls, bucket="both"):
    v = [D[(a, b, l, bucket)][0] for a, b in ps for l in ls if (a, b, l, bucket) in D]
    return sum(v) / len(v)
print(f"{len(pairs)} pairs, common layers 15-28, bucket=both")
for lab, ps in [("all pairs", pairs),
                ("same script (both Latin or both non-Latin... strict: both Latin)", [p for p in pairs if p[0] in LATIN and p[1] in LATIN]),
                ("cross-script (Latin vs non-Latin)", [p for p in pairs if (p[0] in LATIN) != (p[1] in LATIN)]),
                ("en-zh", [("en", "zh")]), ("en-ja", [("en", "ja")]), ("en-ru", [("en", "ru")]), ("en-es", [("en", "es")]), ("en-fr", [("en", "fr")])]:
    s, t = avg(S, ps, layers), avg(T, ps, layers)
    print(f"  {lab:60s} subject {s:.3f} | last token {t:.3f} | ratio {s/t:.2f}")
print("\nper layer, all pairs (subject | last token):")
for l in layers: print(f"  {l:8s} {avg(S, pairs, [l]):.3f} | {avg(T, pairs, [l]):.3f}")
print("\nscope=mean as reported (different bands!), en-zh:", S.get(("en","zh","mean","both")), T.get(("en","zh","mean","both")))
peakS = max(range(12, 29), key=lambda l: avg(S, pairs, [f"layer{l}"])); peakT = max(range(15, 32), key=lambda l: avg(T, pairs, [f"layer{l}"]))
print("peak layer subject:", peakS, round(avg(S, pairs, [f'layer{peakS}']), 3), "| last token:", peakT, round(avg(T, pairs, [f'layer{peakT}']), 3))
