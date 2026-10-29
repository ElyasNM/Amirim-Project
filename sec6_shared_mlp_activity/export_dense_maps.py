"""
export_dense_maps.py  (runs on the cluster, CPU only)
--------------------------------------------------
Exports z-scored FULL MLP activation maps (all neurons, all captured layers) for a
few candidate facts, so the dense "fingerprint" / hexbin figures can be drawn locally.

z-score = per language, per layer, per neuron: (a - mean) / std, with mean/std over
--n_stats random facts of that language (same definition as cosine_z in Phase 2).

Output (small, ~3 MB per fact): <dest>/dense_maps_<TAG>.npz
    z__<stem>__<lang>   float16 [n_layers, d_mlp]   z-scored map
    known__<stem>       json {lang: bool}
    layers              int   [n_layers]
    stems               str   candidate fact stems (in order)
    n_stats             int

Usage:
  python -u export_dense_maps.py --out_dir /path/to/lab_dir/transfer_similarity/out/meta-llama_Llama-3.1-8B \
      --langs en,es,ru,zh,ja --n_stats 300 --n_cand 20 --extra 6178__capital,6444__capital --dest <dir>
"""
import argparse, glob, json, os, random, sys
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--out_dir", required=True)
p.add_argument("--langs", default="en,es,ru,zh,ja")
p.add_argument("--position", default="last_token")
p.add_argument("--n_stats", type=int, default=300)
p.add_argument("--n_cand", type=int, default=20)
p.add_argument("--extra", default="")
p.add_argument("--dest", required=True)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

LANGS = args.langs.split(",")
MAPS = os.path.join(args.out_dir, "maps")
with np.load(os.path.join(MAPS, "model_meta.npz"), allow_pickle=False) as z:
    LAYERS = [int(x) for x in z["layers"]]
    POS = [str(x) for x in z["positions"]]
    D = int(z["d_mlp"])
PIDX = POS.index(args.position)
files = sorted(f for f in glob.glob(os.path.join(MAPS, "*.npz")) if not f.endswith("model_meta.npz"))
print(f"{len(files)} fact files, layers {LAYERS[0]}-{LAYERS[-1]}, d_mlp {D}", flush=True)


def load(fp):
    """-> ({lang: [L, D] float32 or None}, {lang: known})"""
    with np.load(fp, allow_pickle=True) as z:
        meta = json.loads(z["meta"].item()).get("per_lang", {})
        acts = {}
        for l in LANGS:
            if l in z.files:
                a = z[l][:, PIDX, :D].astype(np.float32)
                acts[l] = None if np.isnan(a).any() else a
            else:
                acts[l] = None
        return acts, {l: bool(meta[l]["known"]) if l in meta else False for l in LANGS}


rng = random.Random(args.seed)
stat_files = rng.sample(files, min(args.n_stats, len(files)))
s1 = {l: np.zeros((len(LAYERS), D), np.float64) for l in LANGS}
s2 = {l: np.zeros((len(LAYERS), D), np.float64) for l in LANGS}
cnt = {l: 0 for l in LANGS}
cand_pool = []
for i, fp in enumerate(stat_files):
    acts, known = load(fp)
    for l in LANGS:
        if acts[l] is not None:
            s1[l] += acts[l]; s2[l] += acts[l].astype(np.float64) ** 2; cnt[l] += 1
    if all(acts[l] is not None and known[l] for l in LANGS):
        cand_pool.append(fp)
    if i % 50 == 0:
        print(f"  stats {i}/{len(stat_files)}", flush=True)
mu = {l: s1[l] / cnt[l] for l in LANGS}
sd = {l: np.sqrt(np.maximum(s2[l] / cnt[l] - mu[l] ** 2, 1e-12)) for l in LANGS}
print("stat counts:", cnt, "| candidates known in all langs:", len(cand_pool), flush=True)

stem = lambda fp: os.path.basename(fp)[:-4]
chosen = rng.sample(cand_pool, min(args.n_cand, len(cand_pool)))
for e in [x for x in args.extra.split(",") if x]:
    fp = os.path.join(MAPS, e + ".npz")
    if os.path.exists(fp) and fp not in chosen:
        chosen.append(fp)

out = {"layers": np.array(LAYERS), "n_stats": np.int64(args.n_stats)}
stems = []
for fp in chosen:
    acts, known = load(fp)
    s = stem(fp); stems.append(s)
    out[f"known__{s}"] = np.array(json.dumps(known))
    for l in LANGS:
        if acts[l] is not None:
            out[f"z__{s}__{l}"] = ((acts[l] - mu[l]) / sd[l]).astype(np.float16)
out["stems"] = np.array(stems)
os.makedirs(args.dest, exist_ok=True)
tag = os.path.basename(os.path.normpath(args.out_dir))
dst = os.path.join(args.dest, f"dense_maps_{tag}.npz")
np.savez_compressed(dst, **out)
print("saved", dst, f"{os.path.getsize(dst)/1e6:.1f} MB,", len(stems), "facts", flush=True)
