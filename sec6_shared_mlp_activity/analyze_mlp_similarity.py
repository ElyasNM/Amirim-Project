"""
analyze_mlp_similarity.py
-------------------------
Phase 2 (CPU) of the cross-lingual "transfer via shared MLP paths" experiment
-- SIMILARITY STAGE ONLY, no ablation.

Turns the per-(fact, language) MLP activation maps from Phase 1 into a
*transfer score* per language pair, defined so the raw similarity number is
interpretable:

    aligned(f, L1, L2) = sim( map[f, L1] , map[f, L2] )                 same fact
    control(f, L1, L2) = mean_r sim( map[f, L1] , map[perm_r(f), L2] )  random fact
    transfer(L1, L2)   = mean_f aligned  -  mean_f control

Why the control:
  Two questions in one language and a *different* question in the other already
  share MLP machinery (formatting, "answer a factual query" routing). `control`
  measures that language-pair baseline; aligned - control isolates the
  *fact-specific* co-activation a shared storage/enrichment path would produce.
  We z-score each neuron across facts (per language, per layer) so a few
  always-huge neurons cannot push every similarity to ~1 -- for `cosine_z` a
  genuine null sits near 0.

Broken down by knowledge state of the pair:
    both      model answers the fact correctly in L1 AND L2
    only_one  correct in exactly one
    neither   correct in neither
Def-2 hypothesis: `both` shows a larger transfer score than `neither`.

Outputs (under --results_dir):
    pairwise_transfer.csv        long: pair x position x scope x metric x bucket
    per_fact_similarity.csv.gz   fact-level aligned/control (scope=mean, 2 metrics)
    summary.txt                  sanity checks + ranked pairs + bucket contrast
    figs/*.png                   heatmap / per-layer / per-bucket / distributions

Usage:
  python -u analyze_mlp_similarity.py \
      --out_dir     transfer_similarity/out/meta-llama_Llama-3.1-8B \
      --results_dir results/meta-llama_Llama-3.1-8B_section6/transfer_similarity \
      --topk 100 --n_control_perms 3     2>&1
"""

import os
import sys
import gzip
import json
import glob
import argparse
from collections import OrderedDict, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

from common import parse_layers

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ── Args ───────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--out_dir", required=True, help="Phase-1 --out_dir (contains maps/)")
p.add_argument("--results_dir", required=True)
p.add_argument("--langs", default=None, help="restrict/order langs (comma list)")
p.add_argument("--positions", default=None,
               help="subset of captured positions (default: all)")
p.add_argument("--layers", default=None,
               help="subset of captured layer INDICES, e.g. '20-31' (default: all)")
p.add_argument("--topk", type=int, default=100,
               help="K for top-K active-neuron overlap / Jaccard, per layer")
p.add_argument("--n_control_perms", type=int, default=3)
p.add_argument("--lang_cache", type=int, default=3,
               help="how many languages' map stacks to hold in RAM at once")
p.add_argument("--neuron_scale", default="none", choices=["none", "write_norm"],
               help="scale each neuron by its down_proj column norm before "
                    "cosine (adds metric cosine_w)")
p.add_argument("--focus_langs", default="en",
               help="comma list; per-layer + distribution plots cover pairs "
                    "including any of these")
p.add_argument("--min_facts", type=int, default=20,
               help="skip a (pair,bucket) cell with fewer valid facts than this")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

MAPS_DIR = os.path.join(args.out_dir, "maps")
FIGS = os.path.join(args.results_dir, "figs")
os.makedirs(FIGS, exist_ok=True)
RNG = np.random.default_rng(args.seed)


# ── model_meta + manifest ─────────────────────────────────────────────────
with np.load(os.path.join(MAPS_DIR, "model_meta.npz"), allow_pickle=False) as z:
    CAP_LAYERS = [int(x) for x in z["layers"]]        # layer indices captured
    CAP_POS = [str(x) for x in z["positions"]]
    D_MLP = int(z["d_mlp"])
    WRITE_NORM = z["write_norm"].astype(np.float32)   # [n_cap_layers, d_mlp]
    CAPTURE = str(z["capture"])
    MODEL_NAME = str(z["model_name"])

manifest = {}
mpath = os.path.join(args.out_dir, "cache_metadata.json")
if os.path.exists(mpath):
    manifest = json.load(open(mpath, encoding="utf-8"))
MODEL_TAG = os.path.basename(os.path.normpath(args.out_dir))
FEAT_W = int(manifest.get("feat_width", D_MLP))

# layer subselection (indices into the captured set)
if args.layers:
    keep = set(parse_layers(args.layers, max(CAP_LAYERS) + 1))
    USE_LI = [i for i, li in enumerate(CAP_LAYERS) if li in keep]
else:
    USE_LI = list(range(len(CAP_LAYERS)))
USE_LAYERS = [CAP_LAYERS[i] for i in USE_LI]
NL = len(USE_LI)
WN = WRITE_NORM[USE_LI]                               # [NL, d_mlp]

POSITIONS = ([x.strip() for x in args.positions.split(",")]
             if args.positions else list(CAP_POS))
CAP_POS_IX = [CAP_POS.index(pn) for pn in POSITIONS]

CAN_SCALE = args.neuron_scale == "write_norm" and CAPTURE == "neurons" and FEAT_W == D_MLP
if args.neuron_scale == "write_norm" and not CAN_SCALE:
    print("WARN: --neuron_scale write_norm ignored (needs capture='neurons').")

print(f"model      : {MODEL_NAME}  (tag {MODEL_TAG})")
print(f"capture    : {CAPTURE}  feat_width={FEAT_W}")
print(f"layers used: {USE_LAYERS}")
print(f"positions  : {POSITIONS}")


# ── fact index + knowledge flags (read only 'meta' from each npz) ─────────
fact_files = sorted(f for f in glob.glob(os.path.join(MAPS_DIR, "*.npz"))
                    if os.path.basename(f) != "model_meta.npz")
print(f"fact files : {len(fact_files)}")

FACT_KEYS = []
KNOWN = defaultdict(dict)            # lang -> {fact_key: bool}
LANGS_SEEN = set()
for fp in tqdm(fact_files, desc="index", file=sys.stderr):
    with np.load(fp, allow_pickle=True) as z:
        if "meta" not in z.files:
            continue
        meta = json.loads(z["meta"].item())
        present = [k for k in z.files if k != "meta"]
    key = (meta["index"], meta["relation"])
    FACT_KEYS.append(key)
    for lang, m in meta.get("per_lang", {}).items():
        if lang in present:
            LANGS_SEEN.add(lang)
            KNOWN[lang][key] = bool(m.get("known", False))

LANGS = ([l.strip() for l in args.langs.split(",")] if args.langs
         else sorted(LANGS_SEEN))
LANGS = [l for l in LANGS if l in LANGS_SEEN]
FACT_POS = {k: i for i, k in enumerate(FACT_KEYS)}
NF = len(FACT_KEYS)
print(f"languages  : {LANGS}")
print(f"facts      : {NF}")


# ── lazy per-language stack loader (LRU) ─────────────────────────────────
# entry: acts  float16 [NF, NL, nPos, FEAT_W]   (NaN where position unresolved)
#        valid bool    [NF, nPos]
#        zmean/zstd float32 [NL, nPos, FEAT_W]
_CACHE = OrderedDict()


def load_lang(lang):
    if lang in _CACHE:
        _CACHE.move_to_end(lang)
        return _CACHE[lang]

    acts = None
    valid = np.zeros((NF, len(POSITIONS)), dtype=bool)
    for fp in fact_files:
        with np.load(fp, allow_pickle=True) as z:
            if lang not in z.files or "meta" not in z.files:
                continue
            key = tuple(json.loads(z["meta"].item())[k] for k in ("index", "relation"))
            a = z[lang]                                   # [nCapL, nCapPos, FEAT_W] f16
        a = a[np.ix_(USE_LI, CAP_POS_IX)]                 # [NL, nPos, FEAT_W]
        fi = FACT_POS[key]
        if acts is None:
            acts = np.full((NF,) + a.shape, np.nan, dtype=np.float16)
        acts[fi] = a
        valid[fi] = ~np.isnan(a).any(axis=(0, 2))       # per-position validity

    zmean = np.empty((NL, len(POSITIONS), FEAT_W), dtype=np.float32)
    zstd = np.empty_like(zmean)
    for l in range(NL):                                   # per-layer to bound memory
        sl = acts[:, l].astype(np.float32)               # [NF, nPos, FEAT_W]
        with np.errstate(invalid="ignore"):
            zmean[l] = np.nanmean(sl, axis=0)
            zstd[l] = np.nanstd(sl, axis=0)
    zstd[zstd < 1e-6] = 1.0

    entry = {"acts": acts, "valid": valid, "zmean": zmean, "zstd": zstd}
    _CACHE[lang] = entry
    _CACHE.move_to_end(lang)
    while len(_CACHE) > max(1, args.lang_cache):
        _CACHE.popitem(last=False)
    return entry


# ── similarity primitives ([n, feat] float32) ───────────────────────────
def _cos(a, b):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return np.einsum("fd,fd->f", a, b) / (na * nb + 1e-12)


def _pearson(a, b):
    return _cos(a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True))


def _overlap_jaccard(a, b, k):
    k = min(k, a.shape[1])
    ta = np.argpartition(a, -k, axis=1)[:, -k:]
    tb = np.argpartition(b, -k, axis=1)[:, -k:]
    ov = np.empty(a.shape[0], dtype=np.float32)
    ja = np.empty(a.shape[0], dtype=np.float32)
    for i in range(a.shape[0]):
        inter = np.intersect1d(ta[i], tb[i], assume_unique=False).size
        ov[i] = inter / k
        ja[i] = inter / (2 * k - inter)
    return ov, ja


METRICS = ["cosine_raw", "cosine_z", "pearson", "overlap_topk", "jaccard_topk"]
if CAN_SCALE:
    METRICS.append("cosine_w")
CONCAT_METRICS = ["cosine_raw", "cosine_z", "pearson"] + (["cosine_w"] if CAN_SCALE else [])


def layer_metrics(a1, a2, zm1, zs1, zm2, zs2, wn):
    """a1,a2: [n, feat] float32 for one (layer, position). -> {metric: [n]}."""
    out = {
        "cosine_raw": _cos(a1, a2),
        "pearson": _pearson(a1, a2),
        "cosine_z": _cos((a1 - zm1) / zs1, (a2 - zm2) / zs2),
    }
    ov, ja = _overlap_jaccard(a1, a2, args.topk)
    out["overlap_topk"] = ov
    out["jaccard_topk"] = ja
    if CAN_SCALE:
        out["cosine_w"] = _cos(a1 * wn, a2 * wn)
    return out


def transformed(a, zm, zs, wn, metric):
    if metric == "cosine_z":
        return (a - zm) / zs
    if metric == "cosine_w":
        return a * wn
    if metric == "pearson":
        return a - a.mean(1, keepdims=True)
    return a                                             # cosine_raw


# ── aggregation ─────────────────────────────────────────────────────────
def agg(aligned, control, buckets):
    rows = []
    for bname, mask in buckets.items():
        m = mask & np.isfinite(aligned) & np.isfinite(control)
        n = int(m.sum())
        if n < args.min_facts:
            continue
        al, co = aligned[m], control[m]
        diff = al - co
        pooled = np.sqrt((al.var(ddof=1) + co.var(ddof=1)) / 2) + 1e-12
        wp = ""
        if _HAVE_SCIPY and n >= 10 and np.any(diff != 0):
            try:
                wp = f"{float(wilcoxon(al, co, zero_method='wilcox').pvalue):.3g}"
            except Exception:
                wp = ""
        rows.append(dict(
            bucket=bname, n_facts=n,
            mean_aligned=float(al.mean()), sd_aligned=float(al.std(ddof=1)),
            mean_control=float(co.mean()), sd_control=float(co.std(ddof=1)),
            transfer_score=float(diff.mean()),
            cohens_d=float(diff.mean() / pooled),
            wilcoxon_p=wp,
        ))
    return rows


# ── main sweep ─────────────────────────────────────────────────────────
pairs = [(LANGS[i], LANGS[j])
         for i in range(len(LANGS)) for j in range(i + 1, len(LANGS))]
print(f"\nlanguage pairs: {len(pairs)}")
focus = {x.strip() for x in args.focus_langs.split(",") if x.strip()}

long_rows = []
per_fact_rows = []
per_layer_transfer = defaultdict(dict)   # (L1,L2,pos,metric) -> {bucket: [NL]}
dist_cache = {}                          # (L1,L2,pos,metric) -> (aligned_both, control_both)


def make_perms():
    out = []
    for _ in range(args.n_control_perms):
        pm = RNG.permutation(NF)
        if np.any(pm == np.arange(NF)):
            pm = np.roll(pm, 1)          # guarantees no fixed point for NF > 1
        out.append(pm)
    return out


for (L1, L2) in tqdm(pairs, desc="pairs", file=sys.stderr):
    e1, e2 = load_lang(L1), load_lang(L2)
    A1, A2 = e1["acts"], e2["acts"]                       # [NF, NL, nPos, FEAT_W] f16
    kn1 = np.array([KNOWN[L1].get(k, False) for k in FACT_KEYS])
    kn2 = np.array([KNOWN[L2].get(k, False) for k in FACT_KEYS])
    perms = make_perms()

    for pi, pn in enumerate(POSITIONS):
        vmask = e1["valid"][:, pi] & e2["valid"][:, pi]
        idx = np.where(vmask)[0]
        if idx.size < args.min_facts:
            continue

        buckets = {
            "both":     (kn1 & kn2)[idx],
            "only_one": (kn1 ^ kn2)[idx],
            "neither":  (~kn1 & ~kn2)[idx],
            "all":      np.ones(idx.size, dtype=bool),
        }

        mean_al = {m: np.zeros(idx.size) for m in METRICS}
        mean_co = {m: np.zeros(idx.size) for m in METRICS}
        cc = {m: dict(dot_al=0.0, n1=0.0, n2=0.0, dot_co=0.0, n2c=0.0)
              for m in CONCAT_METRICS}

        for lo in range(NL):
            a1 = A1[idx, lo, pi, :].astype(np.float32)
            a2 = A2[idx, lo, pi, :].astype(np.float32)
            zm1, zs1 = e1["zmean"][lo, pi], e1["zstd"][lo, pi]
            zm2, zs2 = e2["zmean"][lo, pi], e2["zstd"][lo, pi]
            wn = WN[lo] if CAN_SCALE else None

            al = layer_metrics(a1, a2, zm1, zs1, zm2, zs2, wn)

            # control: mean over perms, NaN where permuted partner invalid here
            co_stacks = {m: [] for m in METRICS}
            for pm in perms:
                jj = pm[idx]
                a2p = A2[jj, lo, pi, :].astype(np.float32)
                bad = np.isnan(a2p).any(axis=1)
                a2p_safe = np.where(bad[:, None], 0.0, a2p)
                cm = layer_metrics(a1, a2p_safe, zm1, zs1, zm2, zs2, wn)
                for m in METRICS:
                    v = cm[m].astype(np.float64)
                    v[bad] = np.nan
                    co_stacks[m].append(v)
            with np.errstate(invalid="ignore"):
                co = {m: np.nanmean(np.stack(co_stacks[m]), axis=0) for m in METRICS}

            for m in METRICS:
                for r in agg(al[m], co[m], buckets):
                    r.update(model=MODEL_TAG, lang1=L1, lang2=L2, position=pn,
                             scope=f"layer{USE_LAYERS[lo]}", metric=m)
                    long_rows.append(r)
                mean_al[m] += np.nan_to_num(al[m]) / NL
                mean_co[m] += np.nan_to_num(co[m]) / NL

            if focus & {L1, L2}:
                for m in METRICS:
                    tr = {x["bucket"]: x["transfer_score"] for x in agg(al[m], co[m], buckets)}
                    d = per_layer_transfer[(L1, L2, pn, m)]
                    for bk in ("both", "neither"):
                        d.setdefault(bk, [np.nan] * NL)[lo] = tr.get(bk, np.nan)

            # concat accumulation (dot / squared norms summed across layers)
            for m in CONCAT_METRICS:
                x1 = transformed(a1, zm1, zs1, wn, m)
                x2 = transformed(a2, zm2, zs2, wn, m)
                cc[m]["dot_al"] = cc[m]["dot_al"] + np.einsum("fd,fd->f", x1, x2)
                cc[m]["n1"] = cc[m]["n1"] + np.einsum("fd,fd->f", x1, x1)
                cc[m]["n2"] = cc[m]["n2"] + np.einsum("fd,fd->f", x2, x2)
                pm = perms[0]
                a2p = A2[pm[idx], lo, pi, :].astype(np.float32)
                a2p = np.where(np.isnan(a2p).any(axis=1)[:, None], 0.0, a2p)
                x2c = transformed(a2p, zm2, zs2, wn, m)
                cc[m]["dot_co"] = cc[m]["dot_co"] + np.einsum("fd,fd->f", x1, x2c)
                cc[m]["n2c"] = cc[m]["n2c"] + np.einsum("fd,fd->f", x2c, x2c)

        for m in METRICS:
            for r in agg(mean_al[m], mean_co[m], buckets):
                r.update(model=MODEL_TAG, lang1=L1, lang2=L2, position=pn,
                         scope="mean", metric=m)
                long_rows.append(r)
            if pi == 0 and m in ("cosine_z", "overlap_topk"):
                for i_loc in range(idx.size):
                    bk = ("both" if buckets["both"][i_loc]
                          else "only_one" if buckets["only_one"][i_loc]
                          else "neither")
                    k = FACT_KEYS[idx[i_loc]]
                    per_fact_rows.append(dict(
                        model=MODEL_TAG, lang1=L1, lang2=L2, position=pn, metric=m,
                        index=k[0], relation=k[1], bucket=bk,
                        aligned=float(mean_al[m][i_loc]),
                        control=float(mean_co[m][i_loc])))

        for m in CONCAT_METRICS:
            al_c = cc[m]["dot_al"] / (np.sqrt(cc[m]["n1"] * cc[m]["n2"]) + 1e-12)
            co_c = cc[m]["dot_co"] / (np.sqrt(cc[m]["n1"] * cc[m]["n2c"]) + 1e-12)
            for r in agg(al_c, co_c, buckets):
                r.update(model=MODEL_TAG, lang1=L1, lang2=L2, position=pn,
                         scope="concat", metric=m)
                long_rows.append(r)
            if (focus & {L1, L2}) and m == "cosine_z":
                dist_cache[(L1, L2, pn, m)] = (al_c[buckets["both"]], co_c[buckets["both"]])


# ── write tables ───────────────────────────────────────────────────────
os.makedirs(args.results_dir, exist_ok=True)
COLS = ["model", "lang1", "lang2", "position", "scope", "metric", "bucket",
        "n_facts", "mean_aligned", "sd_aligned", "mean_control", "sd_control",
        "transfer_score", "cohens_d", "wilcoxon_p"]


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return f"{v}"


csv_path = os.path.join(args.results_dir, "pairwise_transfer.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write(",".join(COLS) + "\n")
    for r in long_rows:
        f.write(",".join(_fmt(r.get(c, "")) for c in COLS) + "\n")
print(f"\nwrote {csv_path}  ({len(long_rows)} rows)")

pf_path = os.path.join(args.results_dir, "per_fact_similarity.csv.gz")
pc = ["model", "lang1", "lang2", "position", "metric", "index", "relation",
      "bucket", "aligned", "control"]
with gzip.open(pf_path, "wt", encoding="utf-8") as f:
    f.write(",".join(pc) + "\n")
    for r in per_fact_rows:
        f.write(",".join(_fmt(r[c]) for c in pc) + "\n")
print(f"wrote {pf_path}  ({len(per_fact_rows)} rows)")


# ── plots ──────────────────────────────────────────────────────────────
def rows_where(**cond):
    for r in long_rows:
        if all(r.get(k) == v for k, v in cond.items()):
            yield r


def plot_heatmap(position, metric, scope="mean", bucket="both"):
    M = np.full((len(LANGS), len(LANGS)), np.nan)
    for r in rows_where(position=position, metric=metric, scope=scope, bucket=bucket):
        i, j = LANGS.index(r["lang1"]), LANGS.index(r["lang2"])
        M[i, j] = M[j, i] = r["transfer_score"]
    if np.all(np.isnan(M)):
        return
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(LANGS), 1 + 0.6 * len(LANGS)))
    vmax = np.nanmax(np.abs(M)) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(LANGS))); ax.set_xticklabels(LANGS, rotation=90)
    ax.set_yticks(range(len(LANGS))); ax.set_yticklabels(LANGS)
    for i in range(len(LANGS)):
        for j in range(len(LANGS)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"transfer score (aligned - control)\n{metric}  pos={position}  "
                 f"scope={scope}  bucket={bucket}\n{MODEL_TAG}")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fp = os.path.join(FIGS, f"heatmap__{position}__{metric}__{bucket}.png")
    fig.savefig(fp, dpi=140); plt.close(fig); print(" ", fp)


def plot_by_layer(position, metric):
    items = [(k, v) for k, v in per_layer_transfer.items()
             if k[2] == position and k[3] == metric]
    if not items:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for (L1, L2, _, _), bybk in sorted(items):
        if "both" in bybk:
            ax.plot(USE_LAYERS, bybk["both"], marker="o", ms=3, label=f"{L1}-{L2} both")
        if "neither" in bybk:
            ax.plot(USE_LAYERS, bybk["neither"], marker="x", ms=3, ls="--",
                    alpha=0.6, label=f"{L1}-{L2} neither")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("layer"); ax.set_ylabel("transfer score")
    ax.set_title(f"per-layer transfer score  {metric}  pos={position}\n{MODEL_TAG}")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fp = os.path.join(FIGS, f"by_layer__{position}__{metric}.png")
    fig.savefig(fp, dpi=140); plt.close(fig); print(" ", fp)


def plot_by_bucket(position, metric, scope="mean"):
    order = ["both", "only_one", "neither"]
    vals = {b: [r["transfer_score"]
                for r in rows_where(position=position, metric=metric,
                                    scope=scope, bucket=b)]
            for b in order}
    fig, ax = plt.subplots(figsize=(5, 4))
    xs = range(len(order))
    means = [np.mean(vals[b]) if vals[b] else np.nan for b in order]
    errs = [np.std(vals[b]) if vals[b] else 0 for b in order]
    ax.bar(xs, means, yerr=errs, capsize=4, color=["#2c7fb8", "#7fcdbb", "#c7e9b4"])
    ax.set_xticks(list(xs)); ax.set_xticklabels(order)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("transfer score (mean +/- sd over pairs)")
    ax.set_title(f"transfer by knowledge state  {metric}  pos={position}\n{MODEL_TAG}")
    fig.tight_layout()
    fp = os.path.join(FIGS, f"by_bucket__{position}__{metric}.png")
    fig.savefig(fp, dpi=140); plt.close(fig); print(" ", fp)


def plot_dist(position, metric):
    for (L1, L2, pn, m), (al, co) in dist_cache.items():
        if pn != position or m != metric:
            continue
        al = al[np.isfinite(al)]; co = co[np.isfinite(co)]
        if al.size < 5:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        bins = np.linspace(min(al.min(), co.min()), max(al.max(), co.max()), 40)
        ax.hist(co, bins=bins, alpha=0.55, label="control (random fact)", color="#999999")
        ax.hist(al, bins=bins, alpha=0.55, label="aligned (same fact)", color="#2c7fb8")
        ax.axvline(co.mean(), color="#555555", ls="--")
        ax.axvline(al.mean(), color="#1f5d8c", ls="--")
        ax.set_title(f"{L1}-{L2}  {metric} concat  pos={position}  bucket=both\n{MODEL_TAG}")
        ax.set_xlabel(metric); ax.legend()
        fig.tight_layout()
        fp = os.path.join(FIGS, f"dist__{L1}-{L2}__{position}__{metric}.png")
        fig.savefig(fp, dpi=140); plt.close(fig); print(" ", fp)


print("\nplots:")
for pos in POSITIONS:
    for met in ("cosine_z", "overlap_topk"):
        plot_heatmap(pos, met)
        plot_by_bucket(pos, met)
        plot_by_layer(pos, met)
    plot_dist(pos, "cosine_z")


# ── summary.txt ────────────────────────────────────────────────────────
L = []
L.append(f"Cross-lingual MLP activation-map similarity  --  {MODEL_TAG}")
L.append(f"model={MODEL_NAME}  capture={CAPTURE}  feat_width={FEAT_W}")
L.append(f"layers={USE_LAYERS}")
L.append(f"positions={POSITIONS}  topk={args.topk}  control_perms={args.n_control_perms}"
         f"  scipy={_HAVE_SCIPY}")
L.append("")
L.append("[SANITY] mean similarity over ALL facts (scope=mean):")
L.append("  cosine_z control should sit near 0; raw/overlap can be >0 but the")
L.append("  aligned-minus-control gap is what matters.")
for pos in POSITIONS:
    for met in METRICS:
        cs = [r["mean_control"] for r in rows_where(position=pos, metric=met,
              scope="mean", bucket="all")]
        al = [r["mean_aligned"] for r in rows_where(position=pos, metric=met,
              scope="mean", bucket="all")]
        if cs:
            L.append(f"  {pos:12s} {met:13s} control={np.mean(cs):+.4f}"
                     f"  aligned={np.mean(al):+.4f}")
L.append("")
L.append("[RANKED PAIRS] transfer_score  bucket=both  scope=mean  metric=cosine_z:")
for pos in POSITIONS:
    L.append(f"  -- position={pos} --")
    rr = sorted(rows_where(position=pos, metric="cosine_z", scope="mean", bucket="both"),
                key=lambda r: -r["transfer_score"])
    for r in rr:
        star = "*" if (r["wilcoxon_p"] not in ("", None)
                       and float(r["wilcoxon_p"]) < 0.01) else " "
        L.append(f"   {star} {r['lang1']}-{r['lang2']:<3s} "
                 f"transfer={r['transfer_score']:+.4f} d={r['cohens_d']:+.2f} "
                 f"aligned={r['mean_aligned']:.4f} control={r['mean_control']:.4f} "
                 f"n={r['n_facts']} p={r['wilcoxon_p']}")
L.append("")
L.append("[KNOWLEDGE-STATE CONTRAST] mean transfer_score over all pairs (scope=mean):")
for pos in POSITIONS:
    for met in ("cosine_z", "overlap_topk"):
        cells = []
        for bk in ("both", "only_one", "neither"):
            vs = [r["transfer_score"] for r in rows_where(position=pos, metric=met,
                  scope="mean", bucket=bk)]
            cells.append(f"{bk}={np.mean(vs):+.4f}" if vs else f"{bk}=NA")
        L.append(f"  {pos:12s} {met:13s} " + "   ".join(cells))
L.append("")
L.append("INTERPRETATION")
L.append("  transfer_score > 0, rising with layer  -> fact-specific MLP co-activation")
L.append("    beyond the language-pair baseline: consistent with a shared")
L.append("    enrichment/storage path (Def-2).")
L.append("  both >> neither  -> that co-activation tracks whether the model actually")
L.append("    HAS the fact in both languages (the wanted signal).")
L.append("  both ~= neither or transfer_score ~= 0  -> cross-language answer")
L.append("    agreement is NOT explained by shared MLP paths at these sites; look")
L.append("    upstream to shared representations (Def-1).")
L.append("  Correlational only. It licenses the ablation stage on the pairs/layers")
L.append("  with the largest positive transfer_score.")

sp = os.path.join(args.results_dir, "summary.txt")
open(sp, "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
print(f"\nwrote {sp}")
print(f"figs -> {FIGS}")
