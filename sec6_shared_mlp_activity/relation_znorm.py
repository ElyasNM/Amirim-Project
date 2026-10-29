"""
relation_znorm.py  (runs on the cluster, CPU only)
----------------------------------------------
Motivation: normalize according to the facts in the same language and same relation.

Recomputes cosine_z with the per-neuron mean/std taken over facts of the same
(language, layer, RELATION) instead of (language, layer). Both variants run in the same job:

  lang      z-score per (language, layer)              the paper's cosine_z; sanity: ~0.63 aligned / ~0.001 random (Llama)
  langrel   z-score per (language, layer, relation)    relations with < --min_rel_facts valid facts are dropped from BOTH variants

Controls per variant: `random` (any other fact; the paper's control) and `same_rel`
(other fact of the same relation; the strict control).

Outputs (in --dest):
  relation_znorm_layers_<TAG>.csv   per (pair, layer, variant, condition): mean over facts
  relation_znorm_summary_<TAG>.txt  layer-averaged and peak-layer means over pairs, plus per-relation aligned means
"""
import argparse, glob, json, os, time
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--out_dir", required=True)
p.add_argument("--dest", required=True)
p.add_argument("--k", type=int, default=5, help="partners per fact per condition")
p.add_argument("--min_rel_facts", type=int, default=20)
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

# -- load (last_token only) ---------------------------------------------------
rel, langs, acts = [], None, {}
for i, fp in enumerate(files):
    with np.load(fp, allow_pickle=True) as z:
        m = json.loads(z["meta"].item())
        if langs is None:
            langs = sorted(k for k in z.files if k != "meta")
            acts = {l: np.full((NF, NL, D), np.nan, np.float16) for l in langs}
        for l in langs:
            if l in z.files:
                acts[l][i] = z[l][:, 0, :D]
    rel.append(m["relation"])
    if i % 250 == 0:
        log(f"  loaded {i}/{NF}")
rel = np.array(rel)
valid = {l: ~np.isnan(acts[l]).any(axis=(1, 2)) for l in langs}
log("langs", langs, "| valid per lang", {l: int(v.sum()) for l, v in valid.items()})

# -- partners (fixed across pairs/layers/variants) ----------------------------
rng = np.random.default_rng(args.seed)
idx = np.arange(NF)
CONDS = ["random", "same_rel"]
partners = {c: np.full((NF, args.k), -1, np.int64) for c in CONDS}
for f in range(NF):
    for c, msk in (("random", idx != f), ("same_rel", (idx != f) & (rel == rel[f]))):
        cand = idx[msk]
        if cand.size:
            partners[c][f] = rng.choice(cand, size=args.k, replace=cand.size < args.k)

# facts kept: relation has enough valid facts in every language (so both variants use the same facts)
rel_ok = np.ones(NF, bool)
for r in np.unique(rel):
    sel = rel == r
    if min(int((valid[l] & sel).sum()) for l in langs) < args.min_rel_facts:
        rel_ok[sel] = False
rels_kept = np.unique(rel[rel_ok])
log(f"facts kept (relations with >= {args.min_rel_facts} valid facts in every language): "
    f"{int(rel_ok.sum())}/{NF}; relations kept: {len(rels_kept)}/{len(np.unique(rel))}")

# -- sweep --------------------------------------------------------------------
pairs = [(a, b) for i, a in enumerate(langs) for b in langs[i + 1:]]
VARIANTS = ["lang", "langrel"]
COLS = ["aligned"] + CONDS
OUT = np.full((len(pairs), NL, len(VARIANTS), len(COLS)), np.nan)
REL_ALIGNED = {v: {r: np.full((len(pairs), NL), np.nan) for r in rels_kept} for v in VARIANTS}


def unit(zz, v):
    zz /= (np.linalg.norm(zz, axis=1, keepdims=True) + 1e-12)
    zz[~v] = np.nan
    return zz


for lo in range(NL):
    U = {v: {} for v in VARIANTS}
    for l in langs:
        a = acts[l][:, lo, :].astype(np.float32)
        v = valid[l] & rel_ok
        mu = a[v].mean(0); sd = a[v].std(0); sd[sd < 1e-6] = 1.0
        U["lang"][l] = unit((a - mu) / sd, v)
        zz = np.full_like(a, np.nan)
        for r in rels_kept:
            s = v & (rel == r)
            mu = a[s].mean(0); sd = a[s].std(0); sd[sd < 1e-6] = 1.0
            zz[s] = (a[s] - mu) / sd
        U["langrel"][l] = unit(zz, v)
    for pi, (L1, L2) in enumerate(pairs):
        for vi, var in enumerate(VARIANTS):
            A, B = U[var][L1], U[var][L2]
            al = np.einsum("fd,fd->f", A, B)
            vals = [al]
            for c in CONDS:
                acc = np.zeros(NF); n = np.zeros(NF)
                for j in range(args.k):
                    g = partners[c][:, j]
                    ok = g >= 0
                    s = np.full(NF, np.nan)
                    s[ok] = np.einsum("fd,fd->f", A[ok], B[g[ok]])
                    good = np.isfinite(s)
                    acc[good] += s[good]; n[good] += 1
                vals.append(np.where(n > 0, acc / np.maximum(n, 1), np.nan))
            vals = np.stack(vals, 1)
            OUT[pi, lo, vi] = np.nanmean(vals, axis=0)
            for r in rels_kept:
                m = (rel == r) & np.isfinite(al)
                if m.any():
                    REL_ALIGNED[var][r][pi, lo] = al[m].mean()
    log(f"layer {LAYERS[lo]} done")

# -- outputs ------------------------------------------------------------------
os.makedirs(args.dest, exist_ok=True)
with open(os.path.join(args.dest, f"relation_znorm_layers_{TAG}.csv"), "w", encoding="utf-8") as fh:
    fh.write("lang1,lang2,layer,variant," + ",".join(COLS) + "\n")
    for pi, (L1, L2) in enumerate(pairs):
        for lo in range(NL):
            for vi, var in enumerate(VARIANTS):
                fh.write(f"{L1},{L2},{LAYERS[lo]},{var}," + ",".join(f"{x:.4f}" for x in OUT[pi, lo, vi]) + "\n")

lines = [f"{TAG}  facts kept={int(rel_ok.sum())}/{NF}  relations={len(rels_kept)}  langs={langs}  k={args.k}",
         "mean over language pairs (cosine_z); layer-avg = mean over layers; "
         "peak = max over layers of the pair mean, then mean over pairs"]
for vi, var in enumerate(VARIANTS):
    avg = np.nanmean(OUT[:, :, vi], axis=1)                       # [pairs, cols]
    peak = np.nanmax(OUT[:, :, vi], axis=1)
    lines.append(f"{var:8s} layer-avg " + "  ".join(f"{c}={v:+.3f}" for c, v in zip(COLS, np.nanmean(avg, 0))))
    lines.append(f"{var:8s} peak      " + "  ".join(f"{c}={v:+.3f}" for c, v in zip(COLS, np.nanmean(peak, 0))))
lines.append("per-relation aligned cosine_z (layer-avg, mean over pairs): relation  lang  langrel")
for r in rels_kept:
    a = [np.nanmean(REL_ALIGNED[v][r]) for v in VARIANTS]
    lines.append(f"  {r:32s} {a[0]:+.3f}  {a[1]:+.3f}")
txt = "\n".join(lines)
print(txt, flush=True)
open(os.path.join(args.dest, f"relation_znorm_summary_{TAG}.txt"), "w", encoding="utf-8").write(txt + "\n")
log("saved")
