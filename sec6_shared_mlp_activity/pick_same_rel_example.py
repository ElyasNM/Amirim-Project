"""Rank exported facts by (mean cross-lingual same-fact cos) vs (same-relation control cos, EN fact vs other fact in each L)."""
import numpy as np, sys
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import MLP_ROOT
Z = np.load(MLP_ROOT / "dense_export" / "dense_maps_meta-llama_Llama-3.1-8B.npz", allow_pickle=True)
S = [str(s) for s in Z["stems"]]; L = ["es", "ru", "zh", "ja"]
g = lambda s, l: Z[f"z__{s}__{l}"].astype(np.float32).ravel()
cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
rel = lambda s: s.split("__", 1)[1]
rows = []
for f in S:
    en = g(f, "en"); al = np.mean([cos(en, g(f, l)) for l in L])
    for c in [s for s in S if s != f and rel(s) == rel(f)]:
        ctl = {l: cos(en, g(c, l)) for l in L}
        rows.append((al - max(ctl.values()), f, c, al, ctl))
for gap, f, c, al, ctl in sorted(rows, reverse=True)[:12]:
    print(f"{f:32s} vs {c:32s} same-fact mean {al:.3f}  ctrl " + " ".join(f"{l}:{v:+.2f}" for l, v in ctl.items()))
print("n same-relation (fact, control) combos:", len(rows))
