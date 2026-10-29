"""
localize_topk_mask.py
----------------------
Ablation stage, step 1 (CPU, no model needed).

For every English fact the model answered correctly (per the cached
`meta.per_lang.en.known` flag written by cache_mlp_activation_maps.py), rank
the MLP neurons in each captured layer by their z-scored activation at the
`last_token` position (same z-scoring as `analyze_mlp_similarity.py`'s
cosine_z: per-neuron mean/std computed over the WHOLE English fact population,
so an always-on neuron doesn't dominate) and keep the top-K per layer.

Output: one compact .npz with, for every EN-known fact, its top-K neuron
indices per layer -- the mask `run_ablation.py` will zero out.

Usage:
  python -u localize_topk_mask.py \
      --maps_dir out/meta-llama_Llama-3.1-8B/maps \
      --topk 40 \
      --out_path results/meta-llama_Llama-3.1-8B/ablation/topk_masks_en.npz
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--maps_dir", required=True,
               help="Phase-1 maps dir, e.g. out/<TAG>/maps")
p.add_argument("--lang", default="en")
p.add_argument("--topk", type=int, default=40)
p.add_argument("--out_path", required=True)
args = p.parse_args()

mm = np.load(os.path.join(args.maps_dir, "model_meta.npz"), allow_pickle=True)
LAYERS = [int(x) for x in mm["layers"]]
POSITIONS = [str(x) for x in mm["positions"]]
D_MLP = int(mm["d_mlp"])
NL = len(LAYERS)
if "last_token" not in POSITIONS:
    sys.exit(f"'last_token' not among captured positions: {POSITIONS}")
POS_IX = POSITIONS.index("last_token")
print(f"layers={LAYERS}  d_mlp={D_MLP}  last_token position index={POS_IX}")

fact_files = sorted(glob.glob(os.path.join(args.maps_dir, "*.npz")))
fact_files = [f for f in fact_files if os.path.basename(f) != "model_meta.npz"]
print(f"{len(fact_files)} fact files")


# ── Pass 1: population mean/std per (layer, neuron), over ALL en facts ─────
n_sum = np.zeros((NL, D_MLP), dtype=np.float64)
n_sumsq = np.zeros((NL, D_MLP), dtype=np.float64)
n_count = 0

for fp in tqdm(fact_files, desc="pass1 (mean/std)"):
    with np.load(fp, allow_pickle=True) as z:
        if args.lang not in z.files:
            continue
        a = z[args.lang][:, POS_IX, :].astype(np.float64)  # [NL, D_MLP]
    if np.isnan(a).any():
        continue
    n_sum += a
    n_sumsq += a * a
    n_count += 1

if n_count == 0:
    sys.exit(f"No usable '{args.lang}' activations found.")

zmean = n_sum / n_count
zstd = np.sqrt(np.maximum(n_sumsq / n_count - zmean ** 2, 0.0))
zstd[zstd < 1e-6] = 1.0
print(f"population n={n_count}")


# ── Pass 2: per EN-known fact, top-K z-scored neurons per layer ────────────
fact_keys, masks, subjects, objects, predictions = [], [], [], [], []

for fp in tqdm(fact_files, desc="pass2 (topk masks)"):
    with np.load(fp, allow_pickle=True) as z:
        if args.lang not in z.files:
            continue
        meta = json.loads(z["meta"].item())
        rec = meta["per_lang"].get(args.lang)
        if rec is None or not rec.get("known"):
            continue
        a = z[args.lang][:, POS_IX, :].astype(np.float64)  # [NL, D_MLP]
    if np.isnan(a).any():
        continue

    zscored = (a - zmean) / zstd  # [NL, D_MLP]
    topk_idx = np.argpartition(-zscored, args.topk - 1, axis=1)[:, :args.topk]
    # sort each layer's K indices by descending z for readability
    row_z = np.take_along_axis(zscored, topk_idx, axis=1)
    order = np.argsort(-row_z, axis=1)
    topk_idx = np.take_along_axis(topk_idx, order, axis=1)  # [NL, K]

    key = f"{meta['index']}__{meta['relation']}"
    fact_keys.append(key)
    masks.append(topk_idx.astype(np.int32))
    subjects.append(rec.get("subject") or "")
    objects.append(rec.get("object") or "")
    predictions.append(rec.get("prediction") or "")

print(f"EN-known facts with a mask: {len(fact_keys)}")

masks = np.stack(masks, axis=0)  # [n_facts, NL, K]
os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
np.savez_compressed(
    args.out_path,
    fact_keys=np.array(fact_keys),
    masks=masks,
    layers=np.array(LAYERS, dtype=np.int32),
    topk=np.int32(args.topk),
    subjects=np.array(subjects),
    objects=np.array(objects),
    predictions=np.array(predictions),
    lang=np.array(args.lang),
)
print(f"Wrote {args.out_path}  masks shape={masks.shape}")
