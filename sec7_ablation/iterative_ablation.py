"""
iterative_ablation.py
----------------------
Ablation stage, greedy iterative localization (GPU, single model load).
Llama-3.1-8B, English only.

Idea: a flat top-K (by z-score, static, all layers at once) needs a huge K
before it breaks most facts (see sweep_ablation.py) -- plausibly because the
model has redundant/backup neurons that only become "responsible" once the
primary ones are gone. Instead of a fixed K, grow the mask ONE small batch of
neurons at a time, always picking (globally across all captured layers) the
neurons that are currently most z-scored-active GIVEN the current partial
ablation -- so a rerouted backup path gets caught on the next iteration.

Per target fact:
  1. mask = {} (empty).
  2. Forward pass with current mask installed; capture each layer's
     (pre-zero, i.e. true) neuron activations at last_token, z-score them
     against the fixed EN population mean/std, mask out already-selected
     neurons (-inf), and check a cheap single-token proxy: does the argmax
     next-token still match the correct answer's first token?
       - if NOT (broken): stop, do one real multi-token greedy generate +
         match_fixed to confirm, record convergence.
       - if still correct: add the top `--batch_size` (layer, neuron) pairs
         (globally ranked) to the mask and repeat.
  3. If `--max_iters` is reached without breaking: record as unconverged
     (censored) with whatever mask was built.
  4. Apply the resulting mask (converged or capped) to the SAME fixed set of
     `--n_control` fresh-sampled control facts as run_ablation.py /
     sweep_ablation.py (single ablation pass each, no iteration needed there).

Outputs (under --out_dir, files <out_prefix>_summary.txt / _detail.csv):
  detail: one row per target fact -- converged, n_iterations, mask_size,
    n_layers_touched, layer_breakdown, control_survived/evaluable.

Usage:
  python -u iterative_ablation.py \
      --model_name meta-llama/Llama-3.1-8B \
      --cache_dir  /path/to/lab_dir/cached/meta-llama_Llama-3.1-8B_section6/examples \
      --maps_dir   /path/to/lab_dir/transfer_similarity/out/meta-llama_Llama-3.1-8B/maps \
      --out_dir    /path/to/lab_dir/transfer_similarity/results/meta-llama_Llama-3.1-8B/ablation \
      --n_facts 100 --n_control 3 --batch_size 5 --max_iters 100 --seed 42
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
p.add_argument("--n_facts", type=int, default=100)
p.add_argument("--n_control", type=int, default=3)
p.add_argument("--batch_size", type=int, default=5)
p.add_argument("--max_iters", type=int, default=100)
p.add_argument("--max_new_tokens", type=int, default=10)
p.add_argument("--lang", default="en")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out_prefix", default="iter")
args = p.parse_args()

RNG = np.random.default_rng(args.seed)
os.makedirs(args.out_dir, exist_ok=True)


# ── Population mean/std per (layer, neuron), EN, last_token ────────────────
mm = np.load(os.path.join(args.maps_dir, "model_meta.npz"), allow_pickle=True)
LAYERS = [int(x) for x in mm["layers"]]
POSITIONS = [str(x) for x in mm["positions"]]
D_MLP = int(mm["d_mlp"])
NL = len(LAYERS)
POS_IX = POSITIONS.index("last_token")
print(f"layers={LAYERS}  d_mlp={D_MLP}")

fact_files = sorted(glob.glob(os.path.join(args.maps_dir, "*.npz")))
fact_files = [f for f in fact_files if os.path.basename(f) != "model_meta.npz"]
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


# ── EN-known fact list ──────────────────────────────────────────────────────
FACT_KEYS = []
for fp in tqdm(fact_files, desc="EN-known facts"):
    with np.load(fp, allow_pickle=True) as z:
        if args.lang not in z.files:
            continue
        meta = json.loads(z["meta"].item())
        rec = meta["per_lang"].get(args.lang)
        if rec is None or not rec.get("known"):
            continue
    FACT_KEYS.append(f"{meta['index']}__{meta['relation']}")
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

target_pos = RNG.choice(usable, size=args.n_facts, replace=False)
ctrl_pos_by_target = {}
for pos in target_pos:
    pool = [i for i in usable if i != pos]
    ctrl_pos_by_target[pos] = RNG.choice(pool, size=args.n_control, replace=False)


# ── Model ────────────────────────────────────────────────────────────────
print(f"\nLoading model: {args.model_name}")
tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name, trust_remote_code=True, device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

dec_layers = get_decoder_layers(model)
down_projs = []
for li in LAYERS:
    mod, _ = get_mlp_down_proj(dec_layers[li])
    down_projs.append(mod)

_grab = {}  # layer index in LAYERS -> np.array [d_mlp] (pre-zero activation at last_token)


def make_hook(li_pos, neuron_idx):
    idx_t = torch.as_tensor(neuron_idx, dtype=torch.long) if len(neuron_idx) else None

    def hook(module, inp):
        x = inp[0]
        _grab[li_pos] = x[0, -1, :].detach().float().cpu().numpy()
        if idx_t is not None:
            x = x.clone()
            x[..., idx_t.to(x.device)] = 0.0
            return (x,) + inp[1:]
        return None

    return hook


def install(mask_by_layer):
    """mask_by_layer: dict li_pos -> list[int] neuron indices (this layer)."""
    handles = []
    for li_pos in range(NL):
        handles.append(down_projs[li_pos].register_forward_pre_hook(
            make_hook(li_pos, mask_by_layer.get(li_pos, []))))
    return handles


def remove(handles):
    for h in handles:
        h.remove()


@torch.no_grad()
def forward_check(prompt, mask_by_layer, correct_first_id):
    """One forward pass with mask installed. Returns (top1_matches, enc)."""
    enc = tok(prompt, return_tensors="pt").to(model.device)
    handles = install(mask_by_layer)
    try:
        out = model(enc.input_ids, use_cache=False)
        top1 = int(out.logits[0, -1, :].argmax())
    finally:
        remove(handles)
    return top1 == correct_first_id, enc


@torch.no_grad()
def greedy_predict(prompt, mask_by_layer):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    handles = install(mask_by_layer)
    try:
        out = model.generate(enc.input_ids, max_new_tokens=args.max_new_tokens,
                              do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        remove(handles)
    gen = out[0][enc.input_ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).split("\n")[0].strip()


def is_correct(pred, obj):
    return match_fixed(pred, " " + obj, {})


_baseline_cache = {}


def baseline(fact_key):
    if fact_key not in _baseline_cache:
        rec = cache_rec(fact_key)
        pred = greedy_predict(rec["few_shot_prompt"], {})
        _baseline_cache[fact_key] = (pred, is_correct(pred, rec["object"]))
    return _baseline_cache[fact_key]


def first_token_id(text):
    ids = tok(" " + text, add_special_tokens=False).input_ids
    return ids[0] if ids else None


# ── Iterative localization for one target fact ─────────────────────────────
def localize(fact_key):
    rec = cache_rec(fact_key)
    prompt = rec["few_shot_prompt"]
    correct_first_id = first_token_id(rec["object"])

    mask_by_layer = {}   # li_pos -> list[int]
    masked_flat = np.zeros((NL, D_MLP), dtype=bool)
    n_iter = 0
    converged = False

    for n_iter in range(1, args.max_iters + 1):
        still_ok, _ = forward_check(prompt, mask_by_layer, correct_first_id)
        if not still_ok:
            converged = True
            break

        # rank ALL (layer, neuron) by current z-score, excluding already-masked
        acts = np.stack([_grab[li] for li in range(NL)], axis=0).astype(np.float64)
        z = (acts - zmean) / zstd
        z[masked_flat] = -np.inf
        flat_idx = np.argsort(-z, axis=None)[:args.batch_size]
        li_idx, neuron_idx = np.unravel_index(flat_idx, z.shape)
        for li, ni in zip(li_idx, neuron_idx):
            mask_by_layer.setdefault(int(li), []).append(int(ni))
            masked_flat[li, ni] = True

    mask_size = int(masked_flat.sum())
    layer_breakdown = "|".join(f"{LAYERS[li]}:{len(ns)}" for li, ns in
                                sorted(mask_by_layer.items()))

    final_pred = greedy_predict(prompt, mask_by_layer)
    final_ok = is_correct(final_pred, rec["object"])

    return dict(
        target_key=fact_key, converged=converged, n_iterations=n_iter,
        mask_size=mask_size, n_layers_touched=len(mask_by_layer),
        layer_breakdown=layer_breakdown,
        final_pred=final_pred, final_broken=bool(not final_ok),
        mask_by_layer=mask_by_layer,
    )


# ── Main loop ────────────────────────────────────────────────────────────
rows = []
for pos in tqdm(target_pos, desc="targets"):
    t_key = FACT_KEYS[pos]
    t_base_pred, t_base_ok = baseline(t_key)
    if not t_base_ok:
        continue

    loc = localize(t_key)

    ctrl_keys = [FACT_KEYS[cp] for cp in ctrl_pos_by_target[pos]]
    ctrl_survived, ctrl_evaluable = 0, 0
    for ck in ctrl_keys:
        c_rec = cache_rec(ck)
        c_base_pred, c_base_ok = baseline(ck)
        if not c_base_ok:
            continue
        ctrl_evaluable += 1
        c_abl_pred = greedy_predict(c_rec["few_shot_prompt"], loc["mask_by_layer"])
        if is_correct(c_abl_pred, c_rec["object"]):
            ctrl_survived += 1

    rows.append(dict(
        target_key=t_key, baseline_ok=t_base_ok,
        converged=loc["converged"], n_iterations=loc["n_iterations"],
        mask_size=loc["mask_size"], n_layers_touched=loc["n_layers_touched"],
        layer_breakdown=loc["layer_breakdown"],
        final_broken=loc["final_broken"],
        control_survived=ctrl_survived, control_evaluable=ctrl_evaluable,
    ))


# ── Write outputs ───────────────────────────────────────────────────────────
detail_path = os.path.join(args.out_dir, f"{args.out_prefix}_detail.csv")
with open(detail_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

converged_rows = [r for r in rows if r["converged"]]
mask_sizes = [r["mask_size"] for r in converged_rows]
ctrl_ratios = [r["control_survived"] / max(r["control_evaluable"], 1)
               for r in rows if r["control_evaluable"] > 0]

summary_path = os.path.join(args.out_dir, f"{args.out_prefix}_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    def w(line):
        print(line)
        f.write(line + "\n")

    w(f"targets evaluated: {len(rows)}")
    w(f"converged (broke within {args.max_iters} iters / "
      f"{args.max_iters*args.batch_size} neurons): "
      f"{len(converged_rows)}/{len(rows)} "
      f"({100*len(converged_rows)/max(len(rows),1):.1f}%)")
    if mask_sizes:
        w(f"mask size at convergence: mean={np.mean(mask_sizes):.1f}  "
          f"median={np.median(mask_sizes):.0f}  "
          f"min={min(mask_sizes)}  max={max(mask_sizes)}")
        n_layers = [r["n_layers_touched"] for r in converged_rows]
        w(f"n_layers touched: mean={np.mean(n_layers):.1f}  "
          f"median={np.median(n_layers):.0f}  (of {NL} total)")
    if ctrl_ratios:
        w(f"mean control survival (converged-mask applied to fresh controls): "
          f"{np.mean(ctrl_ratios):.3f}  over {len(ctrl_ratios)} targets")
        n_perfect = sum(1 for x in ctrl_ratios if x == 1.0)
        w(f"targets with 100% control survival: {n_perfect}/{len(ctrl_ratios)}")

print(f"\nWrote {detail_path}")
print(f"Wrote {summary_path}")
