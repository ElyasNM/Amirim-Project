"""
cache_mlp_activation_maps.py
----------------------------
Phase 1 (GPU) of the cross-lingual "transfer via shared MLP paths" experiment
-- SIMILARITY STAGE ONLY, no ablation.

For every (fact, language) pair in an existing cacher.py cache, run ONE forward
pass over the exact few-shot prompt the model was scored on and record the
per-neuron MLP activation vector (the input to each layer's down-projection --
width = intermediate_size) at two token positions:

    last_subject : last token of the subject's last occurrence  ("enrichment" site)
    last_token   : final prompt token, i.e. the position that predicts the answer

These vectors are the "activation maps" whose cross-language similarity Phase 2
turns into a transfer score.  Nothing is ablated or modified here.

Output
------
  <out_dir>/maps/<index>__<relation>.npz          one file per fact
      key "<lang>"        float16 [n_sel_layers, n_positions, d_mlp]
                          (NaN-filled slice if that position was unresolved)
      key "meta"          json string: per-lang subj_pos / prompt_len / object /
                          prediction / known, plus layer + position indexing
  <out_dir>/maps/model_meta.npz
      write_norm          float32 [n_sel_layers, d_mlp]  (L2 norm of each neuron's
                          down_proj column -- optional contribution weighting in Phase 2)
      layers, positions, d_mlp, n_layers_total, model_name
  <out_dir>/cache_metadata.json                   run manifest

Resumable: a fact file is skipped once it contains every requested language.

Usage
-----
  python -u cache_mlp_activation_maps.py \
      --model_name meta-llama/Llama-3.1-8B \
      --cache_dir  cache/meta-llama_Llama-3.1-8B_section6/examples \
      --out_dir    transfer_similarity/out/meta-llama_Llama-3.1-8B \
      --layers all --capture neurons     2>&1
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)  # unbuffered for cluster logs

from common import (
    load_cache, fact_stem, is_known, parse_layers,
    get_decoder_layers, get_mlp_down_proj, find_subject_last_token,
)


# ── Args ───────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model_name", required=True)
p.add_argument("--cache_dir", required=True,
               help="cacher.py output dir, e.g. cache/<tag>_section6/examples")
p.add_argument("--out_dir", required=True,
               help="destination root; maps go to <out_dir>/maps/")
p.add_argument("--layers", default="all",
               help="'all' | '20-31' | '0,4,8' | '15-'  (default: all)")
p.add_argument("--positions", default="last_subject,last_token",
               help="comma list from {last_subject,last_token} (default: both)")
p.add_argument("--capture", default="neurons",
               choices=["neurons", "output", "both"],
               help="neurons = down_proj input (per-neuron map, default); "
                    "output = down_proj output (residual-space MLP contribution); "
                    "both = concatenate [neurons | output-padded] -- usually keep 'neurons'")
p.add_argument("--langs", default=None,
               help="restrict to these langs (comma list); default = all in cache")
p.add_argument("--max_facts", type=int, default=None, help="cap facts (testing)")
p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

STORE_DTYPE = np.float16 if args.dtype == "float16" else np.float32
POSITIONS = [x.strip() for x in args.positions.split(",") if x.strip()]
for pos in POSITIONS:
    if pos not in ("last_subject", "last_token"):
        sys.exit(f"Unknown position '{pos}'")

torch.manual_seed(args.seed)
np.random.seed(args.seed)


# ── Load cache, choose (fact, lang) work list ──────────────────────────────
print(f"Loading cache: {args.cache_dir}")
facts = load_cache(args.cache_dir)
print(f"  {len(facts)} facts")

all_langs = sorted({l for recs in facts.values() for l in recs})
langs = ([l.strip() for l in args.langs.split(",")] if args.langs else all_langs)
langs = [l for l in langs if l in all_langs]
print(f"  languages: {langs}")

fact_keys = sorted(facts.keys())
if args.max_facts:
    fact_keys = fact_keys[:args.max_facts]

maps_dir = os.path.join(args.out_dir, "maps")
os.makedirs(maps_dir, exist_ok=True)


def already_done(key):
    fp = os.path.join(maps_dir, fact_stem(*key) + ".npz")
    if not os.path.exists(fp):
        return False
    try:
        with np.load(fp, allow_pickle=False) as z:
            have = set(z.files)
    except Exception:
        return False
    need = {l for l in langs if l in facts[key]} | {"meta"}
    return need.issubset(have)


pending = [k for k in fact_keys if not already_done(k)]
print(f"  facts pending this run: {len(pending)} / {len(fact_keys)}")
if not pending:
    print("Nothing to do.")
    sys.exit(0)


# ── Model ─────────────────────────────────────────────────────────────────
print(f"\nLoading model: {args.model_name}")
tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name, trust_remote_code=True, device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

dec_layers = get_decoder_layers(model)
n_layers_total = len(dec_layers)
SEL_LAYERS = parse_layers(args.layers, n_layers_total)
print(f"  {n_layers_total} layers total; capturing {len(SEL_LAYERS)}: {SEL_LAYERS}")

down_projs = []
for li in SEL_LAYERS:
    mod, nm = get_mlp_down_proj(dec_layers[li])
    down_projs.append(mod)
# nn.Linear weight is [out_features, in_features]; in_features == #MLP neurons.
d_out, d_mlp = down_projs[0].weight.shape
d_out, d_mlp = int(d_out), int(d_mlp)
print(f"  down_proj '{nm}':  in(neurons)={d_mlp}  out(model)={d_out}")

if args.capture == "neurons":
    FEAT = d_mlp
elif args.capture == "output":
    FEAT = d_out
else:
    FEAT = d_mlp + d_out
print(f"  feature width stored per (layer,position): {FEAT}")

# write_norm[j] = ||W_down[:, j]||  -- magnitude with which neuron j writes to residual
write_norm = np.zeros((len(SEL_LAYERS), d_mlp), dtype=np.float32)
for i, mod in enumerate(down_projs):
    W = mod.weight.detach().float().cpu().numpy()      # [d_out, d_mlp]
    write_norm[i] = np.linalg.norm(W, axis=0)
np.savez(os.path.join(maps_dir, "model_meta.npz"),
         write_norm=write_norm,
         layers=np.array(SEL_LAYERS, dtype=np.int32),
         positions=np.array(POSITIONS),
         d_mlp=np.int64(d_mlp), d_out=np.int64(d_out),
         n_layers_total=np.int64(n_layers_total),
         capture=np.array(args.capture),
         model_name=np.array(args.model_name))


# ── Activation capture hooks ──────────────────────────────────────────────
_grab = {}          # layer_slot -> {"in": tensor[seq,d_mlp], "out": tensor[seq,d_out]}


def _mk_prehook(slot):
    def _hook(module, inp):
        _grab.setdefault(slot, {})["in"] = inp[0][0].detach()
    return _hook


def _mk_fwdhook(slot):
    def _hook(module, inp, out):
        _grab.setdefault(slot, {})["out"] = out[0].detach()
    return _hook


handles = []
for slot, mod in enumerate(down_projs):
    if args.capture in ("neurons", "both"):
        handles.append(mod.register_forward_pre_hook(_mk_prehook(slot)))
    if args.capture in ("output", "both"):
        handles.append(mod.register_forward_hook(_mk_fwdhook(slot)))


@torch.no_grad()
def run_prompt(prompt, subject):
    """One forward pass. Returns (feat[n_sel_layers, n_pos, FEAT] float array,
    subj_pos or None, prompt_len)."""
    enc = tok(prompt, return_tensors="pt")
    ids = enc.input_ids
    ids_list = ids[0].tolist()
    prompt_len = len(ids_list)
    ids = ids.to(model.device)

    subj_pos = find_subject_last_token(ids_list, subject, tok)
    pos_idx = {
        "last_token": prompt_len - 1,
        "last_subject": subj_pos if subj_pos is not None else None,
    }

    _grab.clear()
    model(ids, use_cache=False)

    feat = np.full((len(SEL_LAYERS), len(POSITIONS), FEAT), np.nan, dtype=np.float32)
    for si in range(len(SEL_LAYERS)):
        g = _grab.get(si, {})
        for pi, pname in enumerate(POSITIONS):
            tpos = pos_idx[pname]
            if tpos is None:
                continue
            parts = []
            if args.capture in ("neurons", "both"):
                parts.append(g["in"][tpos].float().cpu().numpy())
            if args.capture in ("output", "both"):
                parts.append(g["out"][tpos].float().cpu().numpy())
            feat[si, pi] = np.concatenate(parts) if len(parts) > 1 else parts[0]
    return feat, subj_pos, prompt_len


# ── Main loop ─────────────────────────────────────────────────────────────
print(f"\nCapturing maps for {len(pending)} facts x up to {len(langs)} langs ...")
n_pairs = 0
n_subj_missing = 0

for key in tqdm(pending, desc="facts", file=sys.stderr):
    fp = os.path.join(maps_dir, fact_stem(*key) + ".npz")
    store, meta_lang = {}, {}
    if os.path.exists(fp):
        try:
            with np.load(fp, allow_pickle=True) as z:
                for k in z.files:
                    if k == "meta":
                        meta_lang = json.loads(z["meta"].item()).get("per_lang", {})
                    else:
                        store[k] = z[k]
        except Exception:
            store, meta_lang = {}, {}

    for lang in langs:
        rec = facts[key].get(lang)
        if rec is None or "few_shot_prompt" not in rec:
            continue
        if lang in store and lang in meta_lang:
            continue  # resume

        feat, subj_pos, plen = run_prompt(rec["few_shot_prompt"], rec.get("subject", ""))
        if subj_pos is None:
            n_subj_missing += 1
        store[lang] = feat.astype(STORE_DTYPE)
        meta_lang[lang] = {
            "subj_pos": subj_pos,
            "prompt_len": plen,
            "subject": rec.get("subject"),
            "object": rec.get("object"),
            "prediction": rec.get("prediction"),
            "known": bool(is_known(rec)),
        }
        n_pairs += 1

    meta = {
        "index": key[0], "relation": key[1],
        "layers": SEL_LAYERS, "positions": POSITIONS,
        "capture": args.capture, "feat_width": FEAT,
        "d_mlp": d_mlp, "d_out": d_out,
        "per_lang": meta_lang,
    }
    tmp = fp + ".tmp.npz"   # keep .npz suffix so np.savez_compressed doesn't rename
    np.savez_compressed(tmp, meta=json.dumps(meta, ensure_ascii=False), **store)
    os.replace(tmp, fp)

for h in handles:
    h.remove()

manifest = {
    "model_name": args.model_name,
    "cache_dir": args.cache_dir,
    "out_dir": args.out_dir,
    "langs": langs,
    "layers": SEL_LAYERS,
    "positions": POSITIONS,
    "capture": args.capture,
    "feat_width": FEAT,
    "d_mlp": d_mlp, "d_out": d_out,
    "n_layers_total": n_layers_total,
    "facts_total": len(fact_keys),
    "pairs_written_this_run": n_pairs,
    "subject_pos_missing": n_subj_missing,
    "store_dtype": args.dtype,
    "seed": args.seed,
}
with open(os.path.join(args.out_dir, "cache_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"\nDone. (fact,lang) maps written this run: {n_pairs}")
print(f"  subject position unresolved: {n_subj_missing}")
print(f"  maps dir : {maps_dir}")
print(f"  manifest : {os.path.join(args.out_dir, 'cache_metadata.json')}")
