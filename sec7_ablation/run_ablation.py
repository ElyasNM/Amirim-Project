"""
run_ablation.py
----------------
Ablation stage, step 2 (GPU). Llama-3.1-8B, English only.

For a sample of English facts the model knows (per localize_topk_mask.py's
masks), zero out that fact's top-K z-scored MLP neurons in every captured
layer (single combined mask, all layers at once) and re-generate greedy.
Checks:
  - does the TARGET fact break (was right, now wrong)?
  - under the SAME mask, does a freshly-sampled set of `n_control` other
    known facts (excluding the target) still answer right? Reported as
    X/n_control per target.

Both target and controls are re-generated WITHOUT ablation first too (fresh
baseline), so results don't depend on a possibly-stale cached prediction.

Usage:
  python -u run_ablation.py \
      --model_name meta-llama/Llama-3.1-8B \
      --cache_dir  cache/meta-llama_Llama-3.1-8B_section6/examples \
      --masks_path results/meta-llama_Llama-3.1-8B/ablation/topk_masks_en.npz \
      --out_csv    results/meta-llama_Llama-3.1-8B/ablation/ablation_results.csv \
      --n_facts 20 --n_control 3 --seed 42    2>&1
"""

import os
import sys
import json
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
p.add_argument("--cache_dir", required=True, help="KLAR cacher.py examples dir")
p.add_argument("--masks_path", required=True, help="output of localize_topk_mask.py")
p.add_argument("--out_csv", required=True)
p.add_argument("--n_facts", type=int, default=20, help="number of target facts")
p.add_argument("--n_control", type=int, default=3)
p.add_argument("--max_new_tokens", type=int, default=10, help="matches klar/cacher.py")
p.add_argument("--lang", default="en")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

RNG = np.random.default_rng(args.seed)


# ── Load masks + KLAR cache ─────────────────────────────────────────────────
print(f"Loading masks: {args.masks_path}")
mz = np.load(args.masks_path, allow_pickle=True)
FACT_KEYS = [str(x) for x in mz["fact_keys"]]
MASKS = mz["masks"]                       # [n_facts, NL, K] neuron indices
LAYERS = [int(x) for x in mz["layers"]]   # actual decoder layer indices
NL = len(LAYERS)
print(f"  {len(FACT_KEYS)} EN-known facts with masks, layers={LAYERS}, "
      f"topk={int(mz['topk'])}")

print(f"Loading KLAR cache: {args.cache_dir}")
cache = load_cache(args.cache_dir)


def cache_rec(fact_key):
    idx_str, relation = fact_key.split("__", 1)
    key = (int(idx_str), relation)
    recs = cache.get(key)
    if recs is None:
        return None
    return recs.get(args.lang)


# keep only mask facts that resolve in the cache and have a prompt
usable = [i for i, k in enumerate(FACT_KEYS) if cache_rec(k) is not None]
print(f"  {len(usable)}/{len(FACT_KEYS)} masked facts resolve in cache")
if len(usable) < args.n_facts + args.n_control + 1:
    sys.exit("Not enough usable facts for the requested n_facts/n_control.")

target_pos = RNG.choice(usable, size=args.n_facts, replace=False)


# ── Model ────────────────────────────────────────────────────────────────
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
    """mask_row: [NL, K] neuron indices, one row per LAYERS entry. Returns handles."""
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


# ── Baseline-prediction cache (avoid redundant forward passes) ─────────────
_baseline_cache = {}


def baseline(fact_key):
    if fact_key not in _baseline_cache:
        rec = cache_rec(fact_key)
        pred = greedy_predict(rec["few_shot_prompt"])
        _baseline_cache[fact_key] = (pred, is_correct(pred, rec["object"]))
    return _baseline_cache[fact_key]


# ── Main loop ────────────────────────────────────────────────────────────
rows = []
for pos in tqdm(target_pos, desc="targets"):
    t_key = FACT_KEYS[pos]
    t_rec = cache_rec(t_key)
    t_mask = MASKS[pos]  # [NL, K]

    t_base_pred, t_base_ok = baseline(t_key)

    control_pool = [i for i in usable if FACT_KEYS[i] != t_key]
    ctrl_pos = RNG.choice(control_pool, size=args.n_control, replace=False)
    ctrl_keys = [FACT_KEYS[i] for i in ctrl_pos]

    handles = install_mask(t_mask)
    try:
        t_abl_pred = greedy_predict(t_rec["few_shot_prompt"])
        t_abl_ok = is_correct(t_abl_pred, t_rec["object"])

        ctrl_results = []
        for ck in ctrl_keys:
            c_rec = cache_rec(ck)
            c_base_pred, c_base_ok = baseline(ck)
            c_abl_pred = greedy_predict(c_rec["few_shot_prompt"])
            c_abl_ok = is_correct(c_abl_pred, c_rec["object"])
            ctrl_results.append(dict(
                key=ck, base_ok=c_base_ok, abl_ok=c_abl_ok,
                base_pred=c_base_pred, abl_pred=c_abl_pred,
            ))
    finally:
        remove(handles)

    ctrl_survived = sum(1 for c in ctrl_results if c["base_ok"] and c["abl_ok"])
    ctrl_evaluable = sum(1 for c in ctrl_results if c["base_ok"])

    rows.append(dict(
        target_key=t_key,
        subject=t_rec.get("subject", ""),
        object=t_rec.get("object", ""),
        baseline_pred=t_base_pred, baseline_ok=t_base_ok,
        ablated_pred=t_abl_pred, ablated_ok=t_abl_ok,
        broken=bool(t_base_ok and not t_abl_ok),
        control_keys="|".join(ctrl_keys),
        control_survived=ctrl_survived,
        control_evaluable=ctrl_evaluable,
        control_n=args.n_control,
    ))

# ── Write compact CSV ───────────────────────────────────────────────────────
import csv
os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

n_eval = sum(1 for r in rows if r["baseline_ok"])
n_broken = sum(1 for r in rows if r["broken"])
print(f"\nTargets: {len(rows)}  (baseline still correct: {n_eval})")
print(f"Broken by ablation: {n_broken}/{n_eval}")
print(f"Mean control survival: "
      f"{np.mean([r['control_survived']/max(r['control_evaluable'],1) for r in rows if r['control_evaluable']>0]):.2f}")
print(f"Wrote {args.out_csv}")
