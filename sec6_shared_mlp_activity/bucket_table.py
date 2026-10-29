"""Values of tab:strict per knowledge bucket (all / both / only_one / neither): mean over pairs of per-pair mean over facts."""
import pandas as pd
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import MLP_ROOT
f = pd.read_csv(MLP_ROOT / "relation_controls" / "relation_controls_meta-llama_Llama-3.1-8B.csv.gz")
cols = ["aligned", "same_rel", "random", "diff_rel"]
def tab(d):
    return d.groupby(["lang1", "lang2"])[cols].mean().mean()
out = {"all": tab(f)}
for b in ["both", "only_one", "neither"]:
    out[b] = tab(f[f.bucket == b])
t = pd.DataFrame(out).round(3); print(t.to_string())
print("\nfacts per bucket (pair-fact rows):", f.bucket.value_counts().to_dict())
print("share aligned > same_rel by bucket:", f.assign(w=f.aligned > f.same_rel).groupby("bucket").w.mean().round(3).to_dict())
