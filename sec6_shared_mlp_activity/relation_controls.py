"""
relation_controls.py  (runs on the cluster, CPU only)
--------------------------------------------------
Stricter controls for cross-lingual MLP sharing (paper: tab:strict).
The paper's control pairs fact f in L1 with a RANDOM other fact in L2, which is
almost always a different relation. Here the partner g is drawn from:

  random            any other fact                     (the paper's control; sanity: ~0)
  diff_rel          other relation
  same_rel          same relation, other fact
  same_rel_diffobj  same relation, different (EN) object
  same_obj_samerel  same relation, same object, other subject
  same_obj_diffrel  different relation, same object

cosine_z as in analyze_mlp_similarity.py: z-score each neuron per language and
layer over all facts, cosine per layer, then average over layers.

Outputs (in --dest):
  relation_controls_<TAG>.csv.gz   per (pair, fact): layer-mean cosine_z per condition + knowledge bucket
  relation_controls_layers_<TAG>.csv  per (pair, layer, condition): mean over facts
  relation_controls_summary_<TAG>.txt
"""
import argparse, glob, gzip, json, os, sys, time, unicodedata
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--out_dir", required=True)
p.add_argument("--dest", required=True)
p.add_argument("--k", type=int, default=5, help="partners per fact per condition")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

MAPS = os.path.join(args.out_dir, "maps")
TAG = os.path.basename(os.path.normpath(args.out_dir))
with np.load(os.path.join(MAPS, "model_meta.npz"), allow_pickle=False) as z:
    LAYERS = [int(x) for x in z["layers"]]
    D = int(z["d_mlp"])
NL = len(LAYERS)
files = sorted(f for f in glob.glob(os.path.join(MAPS, "*.npz")) if not f.endswith("model_meta.npz"))
NF = len(files)
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)
log(f"{NF} facts, layers {LAYERS[0]}-{LAYERS[-1]}, d_mlp {D}")

norm = lambda s: unicodedata.normalize("NFC", (s or "").strip().lower())

# ── load everything (last_token only) ─────────────────────────────────────
rel, obj, keys, known = [], [], [], []
langs = None
acts = {}
for i, fp in enumerate(files):
    with np.load(fp, allow_pickle=True) as z:
        m = json.loads(z["meta"].item())
        if langs is None:
            langs = sorted(k for k in z.files if k != "meta")
            acts = {l: np.full((NF, NL, D), np.nan, np.float16) for l in langs}
        pl = m.get("per_lang", {})
        for l in langs:
            if l in z.files:
                acts[l][i] = z[l][:, 0, :D]
    keys.append(f"{m['index']}__{m['relation']}")
    rel.append(m["relation"])
    obj.append(norm(pl.get("en", {}).get("object")))
    known.append({l: bool(pl.get(l, {}).get("known", False)) for l in langs})
    if i % 250 == 0:
        log(f"  loaded {i}/{NF}")
rel, obj = np.array(rel), np.array(obj)
valid = {l: ~np.isnan(acts[l]).any(axis=(1, 2)) for l in langs}
KN = {l: np.array([k[l] for k in known]) for l in langs}
log("langs", langs, "| valid per lang", {l: int(v.sum()) for l, v in valid.items()})

# ── partner sets (fixed across pairs and layers) ──────────────────────────
rng = np.random.default_rng(args.seed)
idx = np.arange(NF)
CONDS = ["random", "diff_rel", "same_rel", "same_rel_diffobj", "same_obj_samerel", "same_obj_diffrel"]
partners = {c: np.full((NF, args.k), -1, np.int64) for c in CONDS}
for f in range(NF):
    other = idx != f
    pools = {
        "random": other,
        "diff_rel": rel != rel[f],
        "same_rel": other & (rel == rel[f]),
        "same_rel_diffobj": (rel == rel[f]) & (obj != obj[f]),
        "same_obj_samerel": other & (rel == rel[f]) & (obj == obj[f]) & (obj[f] != ""),
        "same_obj_diffrel": (rel != rel[f]) & (obj == obj[f]) & (obj[f] != ""),
    }
    for c, msk in pools.items():
        cand = idx[msk]
        if cand.size:
            partners[c][f] = rng.choice(cand, size=args.k, replace=cand.size < args.k)
log("facts with >=1 partner:", {c: int((partners[c][:, 0] >= 0).sum()) for c in CONDS})

# ── similarity sweep ──────────────────────────────────────────────────────
pairs = [(a, b) for i, a in enumerate(langs) for b in langs[i + 1:]]
NC = 1 + len(CONDS)                        # 0 = aligned
SUM = np.zeros((len(pairs), NF, NC))
CNT = np.zeros((len(pairs), NF, NC))
LAYER_MEAN = np.full((len(pairs), NL, NC), np.nan)

for lo in range(NL):
    U = {}
    for l in langs:                        # z-score per language/layer over valid facts, then unit-normalize
        a = acts[l][:, lo, :].astype(np.float32)
        v = valid[l]
        mu = a[v].mean(0); sd = a[v].std(0); sd[sd < 1e-6] = 1.0
        zz = (a - mu) / sd
        zz /= (np.linalg.norm(zz, axis=1, keepdims=True) + 1e-12)
        zz[~v] = np.nan
        U[l] = zz
    for pi, (L1, L2) in enumerate(pairs):
        A, B = U[L1], U[L2]
        vals = np.full((NF, NC), np.nan)
        vals[:, 0] = np.einsum("fd,fd->f", A, B)
        for ci, c in enumerate(CONDS, start=1):
            P = partners[c]
            acc = np.zeros(NF); n = np.zeros(NF)
            for j in range(args.k):
                g = P[:, j]
                ok = g >= 0
                s = np.full(NF, np.nan)
                s[ok] = np.einsum("fd,fd->f", A[ok], B[g[ok]])
                good = np.isfinite(s)
                acc[good] += s[good]; n[good] += 1
            with np.errstate(invalid="ignore", divide="ignore"):
                vals[:, ci] = np.where(n > 0, acc / n, np.nan)
        good = np.isfinite(vals)
        SUM[pi][good] += vals[good]; CNT[pi][good] += 1
        with np.errstate(invalid="ignore"):
            LAYER_MEAN[pi, lo] = np.nanmean(vals, axis=0)
    log(f"layer {LAYERS[lo]} done")

with np.errstate(invalid="ignore", divide="ignore"):
    MEAN = np.where(CNT == NL, SUM / np.maximum(CNT, 1), np.nan)   # require all layers

# ── outputs ───────────────────────────────────────────────────────────────
os.makedirs(args.dest, exist_ok=True)
cols = ["aligned"] + CONDS
with gzip.open(os.path.join(args.dest, f"relation_controls_{TAG}.csv.gz"), "wt", encoding="utf-8") as fh:
    fh.write("lang1,lang2,fact,relation,object,bucket," + ",".join(cols) + "\n")
    for pi, (L1, L2) in enumerate(pairs):
        for f in range(NF):
            if not np.isfinite(MEAN[pi, f, 0]):
                continue
            k1, k2 = KN[L1][f], KN[L2][f]
            bk = "both" if k1 and k2 else "neither" if not (k1 or k2) else "only_one"
            fh.write(f"{L1},{L2},{keys[f]},{rel[f]},\"{obj[f]}\",{bk}," +
                     ",".join(f"{x:.4f}" if np.isfinite(x) else "" for x in MEAN[pi, f]) + "\n")
with open(os.path.join(args.dest, f"relation_controls_layers_{TAG}.csv"), "w", encoding="utf-8") as fh:
    fh.write("lang1,lang2,layer," + ",".join(cols) + "\n")
    for pi, (L1, L2) in enumerate(pairs):
        for lo in range(NL):
            fh.write(f"{L1},{L2},{LAYERS[lo]}," + ",".join(f"{x:.4f}" for x in LAYER_MEAN[pi, lo]) + "\n")

out = []
def summ(name, sel_pairs, bucket=None):
    per_pair = []
    for pi in sel_pairs:
        L1, L2 = pairs[pi]
        m = np.isfinite(MEAN[pi, :, 0])
        if bucket:
            k1, k2 = KN[L1], KN[L2]
            m &= {"both": k1 & k2, "neither": ~k1 & ~k2, "only_one": k1 ^ k2}[bucket]
        if m.sum() < 20:
            continue
        per_pair.append([np.nanmean(MEAN[pi, m, c]) for c in range(NC)])
    pp = np.array(per_pair)
    out.append(f"{name:28s} pairs={len(pp):3d}  " + "  ".join(f"{c}={v:+.3f}" for c, v in zip(cols, np.nanmean(pp, 0))))

allp = list(range(len(pairs)))
enp = [i for i, (a, b) in enumerate(pairs) if "en" in (a, b)]
out.append(f"{TAG}  facts={NF}  langs={langs}  k={args.k}  layers {LAYERS[0]}-{LAYERS[-1]}")
out.append("mean over language pairs of the per-pair mean over facts (layer-averaged cosine_z)")
for nm, sel in [("all pairs", allp), ("EN pairs", enp)]:
    for b in [None, "both", "neither"]:
        summ(f"{nm} / {b or 'all facts'}", sel, b)
out.append("partners available: " + str({c: int((partners[c][:, 0] >= 0).sum()) for c in CONDS}))
txt = "\n".join(out)
print(txt, flush=True)
open(os.path.join(args.dest, f"relation_controls_summary_{TAG}.txt"), "w", encoding="utf-8").write(txt + "\n")
log("saved")
