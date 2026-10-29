"""
sweep_ablation.py
------------------
Ablation stage, K-sweep (GPU, single model load). Llama-3.1-8B, English only.

Fixes ONE sample of target facts + their (fresh-sampled, then fixed) control
triples, exactly as run_ablation.py would (same seed/order), then re-ablates
each target at growing K (top-K z-scored neurons per layer, all 17 layers at
once): K = k_start, 2*k_start, 4*k_start, ... doubling until control survival
drops below --control_floor of its K=k_start value, then bisects a few times
between the last "safe" K and the first "unsafe" K to localize the knee.

Baseline (non-ablated) predictions for every fact used are computed ONCE and
cached -- they don't depend on K.

Outputs (under --out_dir):
  sweep_summary.csv   one row per K tested: break rate, control survival, phase
  sweep_detail.csv    one row per (target fact, K): full outcome

Usage:
  python -u sweep_ablation.py \
      --model_name meta-llama/Llama-3.1-8B \
      --cache_dir  /path/to/lab_dir/cached/meta-llama_Llama-3.1-8B_section6/examples \
      --maps_dir   /path/to/lab_dir/transfer_similarity/out/meta-llama_Llama-3.1-8B/maps \
      --out_dir    /path/to/lab_dir/transfer_similarity/results/meta-llama_Llama-3.1-8B/ablation \
      --n_facts 200 --n_control 3 --k_start 40 --control_floor 0.90 --seed 42
"""

import os
import sys
import csv
import json
import glob
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

from common import load_cache, match_fixed, get_decoder_layers, get_mlp_down_proj

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model_name", required=True)
p.add_argument("--cache_dir", required=True)
p.add_argument("--maps_dir", required=True)
p.add_argument("--out_dir", required=True)
p.add_argument("--n_facts", type=int, default=200)
p.add_argument("--n_control", type=int, default=3)
p.add_argument("--k_start", type=int, default=40)
p.add_argument("--control_floor", type=float, default=0.90,
               help="fraction of the k_start control-survival rate below which "
                    "we call the decline 'significant' and stop doubling")
p.add_argument("--max_new_tokens", type=int, default=10)
p.add_argument("--n_bisect", type=int, default=3)
p.add_argument("--lang", default="en")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--rank_metric", default="zscore", choices=["zscore", "write_norm"],
               help="zscore: rank neurons by per-neuron z-scored activation "
                    "(population mean/std over all EN facts). write_norm: rank "
                    "by raw activation * down_proj write-norm (||W_down[:,j]||) "
                    "-- a proxy for how much the neuron's firing actually pushes "
                    "into the residual stream, i.e. closer to a causal-relevance "
                    "measure than the purely statistical z-score.")
p.add_argument("--out_prefix", default="sweep",
               help="output files are <out_prefix>_summary.csv / _detail.csv "
                    "(use a distinct prefix per --rank_metric to avoid overwriting)")
args = p.parse_args()

RNG = np.random.default_rng(args.seed)
os.makedirs(args.out_dir, exist_ok=True)


# ── Population mean/std per (layer, neuron), same as localize_topk_mask.py ─
mm = np.load(os.path.join(args.maps_dir, "model_meta.npz"), allow_pickle=True)
LAYERS = [int(x) for x in mm["layers"]]
POSITIONS = [str(x) for x in mm["positions"]]
D_MLP = int(mm["d_mlp"])
NL = len(LAYERS)
POS_IX = POSITIONS.index("last_token")
K_MAX = D_MLP
print(f"layers={LAYERS}  d_mlp={D_MLP}  rank_metric={args.rank_metric}")

fact_files = sorted(glob.glob(os.path.join(args.maps_dir, "*.npz")))
fact_files = [f for f in fact_files if os.path.basename(f) != "model_meta.npz"]

zmean = zstd = None
write_norm = None
if args.rank_metric == "zscore":
    print(f"{len(fact_files)} fact files; computing EN population mean/std ...")
    n_sum = np.zeros((NL, D_MLP), dtype=np.float64)
    n_sumsq = np.zeros((NL, D_MLP), dtype=np.float64)
    n_count = 0
    for fp in tqdm(fact_files, desc="population"):
        with np.load(fp, allow_pickle=True) as z:
            if args.lang not in z.files:
                continue
            a = z[args.lang][:, POS_IX, :].astype(np.float64)
        if np.isnan(a).any():
            continue
        n_sum += a
        n_sumsq += a * a
        n_count += 1
    zmean = n_sum / n_count
    zstd = np.sqrt(np.maximum(n_sumsq / n_count - zmean ** 2, 0.0))
    zstd[zstd < 1e-6] = 1.0
    print(f"population n={n_count}")
else:
    write_norm = mm["write_norm"].astype(np.float64)  # [NL, D_MLP]
    print(f"write_norm loaded: shape={write_norm.shape}")


# ── EN-known fact list (mirrors localize_topk_mask.py pass2 filter) ────────
FACT_KEYS, SUBJECTS, OBJECTS = [], [], []
for fp in tqdm(fact_files, desc="EN-known facts"):
    with np.load(fp, allow_pickle=True) as z:
        if args.lang not in z.files:
            continue
        meta = json.loads(z["meta"].item())
        rec = meta["per_lang"].get(args.lang)
        if rec is None or not rec.get("known"):
            continue
    FACT_KEYS.append(f"{meta['index']}__{meta['relation']}")
    SUBJECTS.append(rec.get("subject") or "")
    OBJECTS.append(rec.get("object") or "")
print(f"EN-known facts: {len(FACT_KEYS)}")

print(f"Loading KLAR cache: {args.cache_dir}")
cache = load_cache(args.cache_dir)


def cache_rec(fact_key):
    idx_str, relation = fact_key.split("__", 1)
    recs = cache.get((int(idx_str), relation))
    return None if recs is None else recs.get(args.lang)


usable = [i for i, k in enumerate(FACT_KEYS) if cache_rec(k) is not None]
print(f"{len(usable)}/{len(FACT_KEYS)} resolve in cache")
if len(usable) < args.n_facts + args.n_control + 1:
    sys.exit("Not enough usable facts.")

# Same draw order as run_ablation.py: pick targets, then per-target controls.
target_pos = RNG.choice(usable, size=args.n_facts, replace=False)
ctrl_pos_by_target = {}
for pos in target_pos:
    pool = [i for i in usable if i != pos]
    ctrl_pos_by_target[pos] = RNG.choice(pool, size=args.n_control, replace=False)


# ── Model ───────────────────────────────────────────────────────────────
print(f"\nLoading model: {args.model_name}")
tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name, trust_remote_code=True, device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

dec_layers = get_decoder_layers(model)
down_projs = {}
for li in LAYERS:
    mod, _ = get_mlp_down_proj(dec_layers[li])
    down_projs[li] = mod


def make_zero_hook(neuron_idx):
    idx_t = torch.as_tensor(neuron_idx, dtype=torch.long)

    def hook(module, inp):
        x = inp[0].clone()
        x[..., idx_t.to(x.device)] = 0.0
        return (x,) + inp[1:]

    return hook


def install_mask(mask_row):
    handles = []
    for li_pos, li in enumerate(LAYERS):
        h = down_projs[li].register_forward_pre_hook(make_zero_hook(mask_row[li_pos]))
        handles.append(h)
    return handles


def remove(handles):
    for h in handles:
        h.remove()


@torch.no_grad()
def greedy_predict(prompt):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(enc.input_ids, max_new_tokens=args.max_new_tokens,
                          do_sample=False, pad_token_id=tok.eos_token_id)
    gen = out[0][enc.input_ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).split("\n")[0].strip()


def is_correct(pred, obj):
    return match_fixed(pred, " " + obj, {})


_baseline_cache = {}


def baseline(fact_key):
    if fact_key not in _baseline_cache:
        rec = cache_rec(fact_key)
        pred = greedy_predict(rec["few_shot_prompt"])
        _baseline_cache[fact_key] = (pred, is_correct(pred, rec["object"]))
    return _baseline_cache[fact_key]


# ── Full per-layer z-scored neuron ranking for each target fact ───────────
print("\nRanking neurons (full, per target fact) ...")
full_rank = {}  # pos -> [NL, D_MLP] int32, descending score per layer
for pos in tqdm(target_pos, desc="rank"):
    fp = os.path.join(args.maps_dir, FACT_KEYS[pos] + ".npz")
    with np.load(fp, allow_pickle=True) as z:
        a = z[args.lang][:, POS_IX, :].astype(np.float64)
    if args.rank_metric == "zscore":
        score = (a - zmean) / zstd
    else:
        score = a * write_norm
    full_rank[pos] = np.argsort(-score, axis=1).astype(np.int32)  # [NL, D_MLP]


# ── Baseline predictions for every fact we'll ever touch ──────────────────
print("\nBaseline predictions (unablated) ...")
all_facts_needed = set(FACT_KEYS[p] for p in target_pos)
for pos in target_pos:
    for cp in ctrl_pos_by_target[pos]:
        all_facts_needed.add(FACT_KEYS[cp])
for fk in tqdm(sorted(all_facts_needed), desc="baseline"):
    baseline(fk)


# ── One ablation pass at a given K, over all targets ───────────────────────
def run_k(k):
    detail = []
    for pos in target_pos:
        t_key = FACT_KEYS[pos]
        t_rec = cache_rec(t_key)
        t_base_pred, t_base_ok = baseline(t_key)
        mask_row = full_rank[pos][:, :k]  # [NL, k]

        handles = install_mask(mask_row)
        try:
            t_abl_pred = greedy_predict(t_rec["few_shot_prompt"])
            t_abl_ok = is_correct(t_abl_pred, t_rec["object"])

            ctrl_rows = []
            for cp in ctrl_pos_by_target[pos]:
                c_key = FACT_KEYS[cp]
                c_rec = cache_rec(c_key)
                c_base_pred, c_base_ok = baseline(c_key)
                c_abl_pred = greedy_predict(c_rec["few_shot_prompt"])
                c_abl_ok = is_correct(c_abl_pred, c_rec["object"])
                ctrl_rows.append((c_key, c_base_ok, c_abl_ok))
        finally:
            remove(handles)

        ctrl_evaluable = sum(1 for _, b, _ in ctrl_rows if b)
        ctrl_survived = sum(1 for _, b, a_ok in ctrl_rows if b and a_ok)

        detail.append(dict(
            k=k, target_key=t_key,
            baseline_ok=t_base_ok, ablated_ok=t_abl_ok,
            broken=bool(t_base_ok and not t_abl_ok),
            control_survived=ctrl_survived, control_evaluable=ctrl_evaluable,
        ))
    return detail


def summarize(k, detail, phase):
    ev = [r for r in detail if r["baseline_ok"]]
    broken = [r for r in ev if r["broken"]]
    ctrl_ev = sum(r["control_evaluable"] for r in ev)
    ctrl_surv = sum(r["control_survived"] for r in ev)
    return dict(
        k=k, phase=phase,
        n_targets_evaluable=len(ev),
        target_broken=len(broken),
        target_break_rate=len(broken) / max(len(ev), 1),
        n_control_pairs_evaluable=ctrl_ev,
        control_survived_pairs=ctrl_surv,
        control_survival_rate=ctrl_surv / max(ctrl_ev, 1),
    )


# ── Doubling phase ─────────────────────────────────────────────────────────
all_detail = []
all_summary = []
k = args.k_start
k0_survival = None
last_good_k, first_bad_k = None, None

print(f"\n=== Doubling phase (k_start={args.k_start}, control_floor={args.control_floor}) ===")
while True:
    print(f"\n--- K={k} ---")
    detail = run_k(k)
    all_detail.extend(detail)
    summ = summarize(k, detail, "doubling")
    all_summary.append(summ)
    print(f"  break_rate={summ['target_break_rate']:.3f}  "
          f"control_survival={summ['control_survival_rate']:.3f}")

    if k0_survival is None:
        k0_survival = summ["control_survival_rate"]

    if summ["control_survival_rate"] < args.control_floor * k0_survival:
        first_bad_k = k
        break

    last_good_k = k
    if k * 2 > K_MAX:
        print(f"Reached K_MAX={K_MAX} without a significant control decline.")
        break
    k *= 2

# ── Bisection phase ─────────────────────────────────────────────────────────
if last_good_k is not None and first_bad_k is not None:
    print(f"\n=== Bisection phase between {last_good_k} (safe) and {first_bad_k} (unsafe) ===")
    lo, hi = last_good_k, first_bad_k
    for i in range(args.n_bisect):
        mid = (lo + hi) // 2
        if mid in (lo, hi):
            break
        print(f"\n--- bisect K={mid} ---")
        detail = run_k(mid)
        all_detail.extend(detail)
        summ = summarize(mid, detail, "bisection")
        all_summary.append(summ)
        print(f"  break_rate={summ['target_break_rate']:.3f}  "
              f"control_survival={summ['control_survival_rate']:.3f}")
        if summ["control_survival_rate"] < args.control_floor * k0_survival:
            hi = mid
        else:
            lo = mid
    print(f"\nKnee localized between K={lo} (safe) and K={hi} (unsafe).")


# ── Write outputs ───────────────────────────────────────────────────────────
summary_path = os.path.join(args.out_dir, f"{args.out_prefix}_summary.csv")
detail_path = os.path.join(args.out_dir, f"{args.out_prefix}_detail.csv")

all_summary.sort(key=lambda r: r["k"])
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(all_summary[0].keys()))
    w.writeheader()
    w.writerows(all_summary)

with open(detail_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(all_detail[0].keys()))
    w.writeheader()
    w.writerows(all_detail)

print(f"\nWrote {summary_path}")
print(f"Wrote {detail_path}")
print("\nSummary (K, break_rate, control_survival, phase):")
for r in all_summary:
    print(f"  K={r['k']:5d}  break={r['target_break_rate']:.3f}  "
          f"ctrl_surv={r['control_survival_rate']:.3f}  [{r['phase']}]")
