"""Per-layer mean (over all 66 pairs) of aligned vs controls, and the fact-specific gap aligned - same_rel."""
import pandas as pd
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import MLP_ROOT
d = pd.read_csv(MLP_ROOT / "relation_controls" / "relation_controls_layers_meta-llama_Llama-3.1-8B.csv")
m = d.groupby("layer")[["aligned", "random", "same_rel", "same_obj_samerel", "same_obj_diffrel"]].mean()
m["gap_vs_same_rel"] = m.aligned - m.same_rel
print(m.round(3).to_string())
f = pd.read_csv(MLP_ROOT / "relation_controls" / "relation_controls_meta-llama_Llama-3.1-8B.csv.gz")
print("\nper-fact share with aligned > same_rel:", round(float((f.aligned > f.same_rel).mean()), 3))
r = f.groupby("relation")[["aligned", "same_rel"]].mean(); r["gap"] = r.aligned - r.same_rel
print(r.sort_values("gap").round(3).to_string())
