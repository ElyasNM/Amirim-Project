"""
cacher_subject_activation_patching.py
-------------------------------------
Cross-language subject activation patching + per-layer logit lens on the
subject token, for the fail_en_ok regime.

Hypothesis under test
---------------------
KLAR claims target-language failures are driven by a "translation failure"
in the final layers. Our prior findings (target-lang token at rank 0-5 in
83% of failures, EN-answer rise in mid-layers) suggest the failure is more
likely a narrow competition loss with intact entity recognition, not a
translation collapse. This script runs the cleanest causal test:

For each (fact, lang_L) where EN succeeded but L failed:
  1. EN forward pass → extract last-subject-token hidden state per layer (E_l).
  2. L  forward pass → extract last-subject-token hidden state per layer (S_l).
  3. (B) Logit-lens analysis: project E_l and S_l through the unembedding,
     score top-K tokens and the rank/probability of:
        - EN subject first token, L subject first token
        - EN object  first token, L object  first token
     This characterises decode-ability of the entity from each representation
     without any patchscope-prompt confound.
  4. (C) Causal intervention: for each layer l, re-run the L forward with
     S_l replaced by E_l at the L subject-token position; record the next-
     token argmax and whether it now matches EN object or L object. If
     patching the EN subject rescues the failure, the failure was an entity-
     encoding issue.

(B) and (C) share the EN/L extractions, so combined cost == cost of (C).

Output: one JSON per (index, relation), one record per non-EN language.
Resumable: per-language records inside a fact file are skipped if present.

Usage:
    python cacher_subject_activation_patching.py \
        --model_name meta-llama/Llama-3.1-8B \
        --existing_cache /path/to/Llama-3.1-8B_section6/examples \
        --split fail_en_ok
"""

import os
import sys
import json
import glob
import random
import argparse
import unicodedata
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import CACHE_ROOT

# Force unbuffered output so prints appear immediately on cluster.
sys.stdout.reconfigure(line_buffering=True)


# ── Args ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--existing_cache", type=str, required=True,
                    help="Path to cacher.py output, e.g. "
                         "cache/meta-llama_Llama-3.1-8B_section6/examples")
parser.add_argument("--split", type=str, default="fail_en_ok",
                    choices=["fail_en_ok", "success", "both"],
                    help="Which (fact, lang) pairs to process. "
                         "Only EN-correct facts are ever included (we patch "
                         "FROM the EN representation).")
parser.add_argument("--max_examples", type=int, default=None,
                    help="Cap total (fact, lang) pairs (for quick testing).")
parser.add_argument("--layer_batch_size", type=int, default=8,
                    help="How many per-layer patches to batch into one "
                         "forward pass. Reduce if OOM.")
parser.add_argument("--top_k_lens", type=int, default=5,
                    help="Top-K tokens to record from logit-lens decode.")
parser.add_argument("--top_k_patch", type=int, default=5,
                    help="Top-K tokens to record from each patched forward.")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# ── Matching ────────────────────────────────────────────────────────────────
def _norm(s):
    return unicodedata.normalize("NFC", s.lower().strip())

def _pfx(a, b):
    a, b = _norm(a), _norm(b)
    return len(a) > 0 and b.startswith(a)

def match_fixed(pred, target, rec):
    """Same conservative match used in deep_failure_investigation/common.py.
       rec is the original cache record (provides logit_lens token ids)."""
    if _pfx(pred, target) or _pfx(target, pred):
        return True
    p, t = _norm(pred), _norm(target)
    ll = rec.get("logit_lens", {}) if isinstance(rec, dict) else {}
    ptid = ll.get("predicted_token_id")
    ctid = ll.get("correct_object_token_id")
    if len(p) > 4 and t.endswith(p) and p != t:
        return True
    if len(p) > 2 and t.endswith(p) and p != t and ptid is not None and ptid == ctid:
        return True
    if 0 < len(p) <= 2 and ptid is not None and ptid == ctid:
        return True
    return False

def match_simple(pred, target):
    """Bidirectional prefix match for argmax tokens from our patched forwards.
       We don't have a fresh logit_lens record, so the simpler form suffices."""
    return _pfx(pred, target) or _pfx(target, pred)


# ── Token helpers ───────────────────────────────────────────────────────────
def find_subject_last_token(input_ids_list, subject, tokenizer):
    """Position of the last token of the subject's last occurrence."""
    for prefix in (" ", ""):
        sub_ids = tokenizer.encode(prefix + subject, add_special_tokens=False)
        if not sub_ids:
            continue
        for start in range(len(input_ids_list) - len(sub_ids), -1, -1):
            if input_ids_list[start:start + len(sub_ids)] == sub_ids:
                return start + len(sub_ids) - 1
    last_word = subject.split()[-1] if subject.split() else subject
    for prefix in (" ", ""):
        sub_ids = tokenizer.encode(prefix + last_word, add_special_tokens=False)
        if not sub_ids:
            continue
        for start in range(len(input_ids_list) - len(sub_ids), -1, -1):
            if input_ids_list[start:start + len(sub_ids)] == sub_ids:
                return start + len(sub_ids) - 1
    return None

def first_token_id(text, tokenizer):
    """First token id of ' '+text (preferred) or text alone. None if empty."""
    if not text:
        return None
    for prefix in (" ", ""):
        ids = tokenizer.encode(prefix + text, add_special_tokens=False)
        if ids:
            return ids[0]
    return None


# ── Module finders ──────────────────────────────────────────────────────────
def _resolve(model, dotted):
    obj = model
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj

def get_decoder_layers(model):
    candidates = ["model.layers", "transformer.h",
                  "gpt_neox.layers", "transformer.layers"]
    for attr in candidates:
        try:
            return _resolve(model, attr)
        except AttributeError:
            continue
    raise ValueError(f"Cannot find decoder layers. Tried: {candidates}")

def get_final_norm(model):
    candidates = ["model.norm", "model.final_layernorm", "model.layer_norm",
                  "transformer.ln_f", "transformer.norm",
                  "gpt_neox.final_layer_norm"]
    for attr in candidates:
        try:
            return _resolve(model, attr)
        except AttributeError:
            continue
    raise ValueError(f"Cannot find final layer norm. Tried: {candidates}")

def get_lm_head(model):
    for attr in ("lm_head", "embed_out", "output"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise ValueError("Cannot find lm_head.")


# ── Load existing cache ─────────────────────────────────────────────────────
print(f"Loading existing cache from {args.existing_cache} ...")
existing = {}
for path in sorted(glob.glob(os.path.join(args.existing_cache, "*.json"))):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    existing[(data["index"], data["relation"])] = data
print(f"  Loaded {len(existing)} fact files.")

all_langs = set()
for d in existing.values():
    for k, v in d.items():
        if k not in ("index", "relation") and isinstance(v, dict):
            all_langs.add(k)
non_en = sorted(l for l in all_langs if l != "en")
print(f"  Non-EN languages: {non_en}")


# ── Select (fact, lang) pairs ───────────────────────────────────────────────
to_process = []
for key, data in existing.items():
    en_rec = data.get("en")
    if not isinstance(en_rec, dict):
        continue
    if "few_shot_prompt" not in en_rec:
        continue
    en_ok = match_fixed(en_rec.get("prediction", ""),
                        " " + en_rec.get("object", ""), en_rec)
    if not en_ok:
        # We patch FROM the EN subject representation, so we require EN to be correct.
        continue

    for lang in non_en:
        rec = data.get(lang)
        if not isinstance(rec, dict) or "few_shot_prompt" not in rec:
            continue
        tgt_ok = match_fixed(rec.get("prediction", ""),
                             " " + rec.get("object", ""), rec)
        status = "success" if tgt_ok else "fail_en_ok"

        if args.split == "fail_en_ok" and status != "fail_en_ok":
            continue
        if args.split == "success" and status != "success":
            continue
        # split == "both": keep all

        to_process.append((key, lang, en_rec, rec, status))

if args.max_examples:
    random.shuffle(to_process)
    to_process = to_process[:args.max_examples]

status_counts = defaultdict(int)
for _, _, _, _, s in to_process:
    status_counts[s] += 1
print(f"\n(fact, lang) pairs to process: {len(to_process)}")
for s, c in sorted(status_counts.items()):
    print(f"  {s}: {c}")


# ── Output paths ────────────────────────────────────────────────────────────
model_tag = args.model_name.replace("/", "_")
out_root  = os.path.join(str(CACHE_ROOT), f"{model_tag}_section6_subj_act_patch")
out_dir   = os.path.join(out_root, "examples")
os.makedirs(out_dir, exist_ok=True)

def out_path(index, relation):
    return os.path.join(out_dir, f"{index}__{relation}.json")


# ── Model ───────────────────────────────────────────────────────────────────
print(f"\nLoading model: {args.model_name}")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name, trust_remote_code=True, device_map="auto"
)
model.eval()

decoder_layers = get_decoder_layers(model)
final_norm     = get_final_norm(model)
lm_head        = get_lm_head(model)
num_layers     = len(decoder_layers)
print(f"  num_hidden_layers = {num_layers}")


# ── Subject hidden-state extraction ─────────────────────────────────────────
# We capture the pre-final-norm hidden state via a forward pre-hook because
# HF's output_hidden_states[-1] is post-norm in current Transformers; for
# patching we want the layer's actual block output (pre-norm).
_pre_norm_capture = {}
def _pre_norm_pre_hook(module, args_tuple):
    _pre_norm_capture["h"] = args_tuple[0]
    return args_tuple

@torch.no_grad()
def extract_subject_hidden(input_ids, subj_pos):
    """
    Forward pass over input_ids; return one cloned hidden vector at subj_pos
    per transformer block (index i = output of decoder block i).
    """
    _pre_norm_capture.clear()
    handle = final_norm.register_forward_pre_hook(_pre_norm_pre_hook)
    try:
        out = model(input_ids, output_hidden_states=True, use_cache=False)
    finally:
        handle.remove()

    hs = out.hidden_states
    per_layer = []
    for i in range(num_layers):
        # hs[0] is the embedding output; hs[i+1] is the output of decoder block i
        # for i < num_layers-1. The last entry hs[num_layers] may be POST-norm
        # depending on Transformers version, so we use the captured pre-norm
        # state for the final block instead.
        if i < num_layers - 1:
            v = hs[i + 1][0, subj_pos, :].clone()
        else:
            v = _pre_norm_capture["h"][0, subj_pos, :].clone()
        per_layer.append(v)

    del out
    _pre_norm_capture.clear()
    return per_layer


# ── Logit-lens scoring (B) ──────────────────────────────────────────────────
@torch.no_grad()
def project_to_logits(h):
    """[hidden] → [vocab]. Standard logit-lens: apply final norm + lm_head
       to mid-layer hidden states. Approximate but the conventional choice."""
    h = h.unsqueeze(0)              # [1, hidden]
    h = final_norm(h)
    return lm_head(h).squeeze(0)    # [vocab]

@torch.no_grad()
def lens_record(h, tokens_of_interest):
    """
    Top-K decode + per-token rank/prob from a single hidden vector.
    tokens_of_interest: dict {name: token_id_or_None}.
    """
    logits = project_to_logits(h)
    probs  = torch.softmax(logits, dim=-1)

    top_p, top_idx = probs.topk(args.top_k_lens)
    top = [
        {"token": tokenizer.decode([int(i)]), "id": int(i), "prob": float(p)}
        for p, i in zip(top_p, top_idx)
    ]

    # Rank lookup: rank_lookup[tid] = position of tid in descending-logit order.
    sorted_tids = logits.argsort(descending=True)
    rank_lookup = torch.empty_like(sorted_tids)
    rank_lookup[sorted_tids] = torch.arange(len(sorted_tids), device=sorted_tids.device)

    ranks, probs_out = {}, {}
    for name, tid in tokens_of_interest.items():
        if tid is None:
            ranks[name] = None
            probs_out[name] = None
        else:
            ranks[name] = int(rank_lookup[tid].item())
            probs_out[name] = float(probs[tid].item())

    return {"top": top, "ranks": ranks, "probs": probs_out}


# ── Causal patching (C) ─────────────────────────────────────────────────────
class _PatchAtBatchIdx:
    """
    Forward hook on a decoder block. On its output, overrides
    hidden[batch_idx, target_pos, :] with replacement_vec. Llama/Qwen/OLMo
    blocks return a tuple (hidden, ...); GPT-style return a bare tensor.
    """
    __slots__ = ("batch_idx", "target_pos", "replacement_vec")

    def __init__(self, batch_idx, target_pos, replacement_vec):
        self.batch_idx       = batch_idx
        self.target_pos      = target_pos
        self.replacement_vec = replacement_vec

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        # In-place edit is safe under no_grad and avoids cloning whole layer
        # outputs every forward.
        hs[self.batch_idx, self.target_pos, :] = self.replacement_vec
        if is_tuple:
            return (hs,) + output[1:]
        return hs


@torch.no_grad()
def run_patched_batch(tgt_input_ids, tgt_subj_pos,
                      en_hidden_layers, layer_indices):
    """
    Batched patched forward. Each batch row b corresponds to layer
    layer_indices[b]: at decoder_layers[layer_indices[b]] the subject token
    output is replaced by en_hidden_layers[layer_indices[b]].

    Returns:
        last_logits: [B, vocab] next-token logits for each patched run.
    """
    B = len(layer_indices)
    batched = tgt_input_ids.expand(B, -1).contiguous()

    handles = []
    for batch_idx, layer_idx in enumerate(layer_indices):
        hook = _PatchAtBatchIdx(batch_idx, tgt_subj_pos,
                                en_hidden_layers[layer_idx])
        handles.append(decoder_layers[layer_idx].register_forward_hook(hook))
    try:
        out = model(batched, use_cache=False)
        last_logits = out.logits[:, -1, :].clone()
    finally:
        for h in handles:
            h.remove()
        del out

    return last_logits


@torch.no_grad()
def patch_all_layers(tgt_input_ids, tgt_subj_pos, en_hidden_layers,
                     en_object, tgt_object, en_obj_first, tgt_obj_first):
    """
    Run a patched forward for every layer in 1..N. Returns per-layer records
    with argmax token, top-K, rank/prob of EN/L object first tokens, and
    rescue flags (does the argmax now match EN object or L object?).
    """
    records = [None] * num_layers
    for batch_start in range(0, num_layers, args.layer_batch_size):
        batch_end     = min(batch_start + args.layer_batch_size, num_layers)
        layer_indices = list(range(batch_start, batch_end))

        last_logits = run_patched_batch(
            tgt_input_ids, tgt_subj_pos, en_hidden_layers, layer_indices
        )                                                  # [B, vocab]
        probs = torch.softmax(last_logits, dim=-1)

        for b, l in enumerate(layer_indices):
            top_p, top_idx = probs[b].topk(args.top_k_patch)
            top = [
                {"token": tokenizer.decode([int(i)]),
                 "id": int(i), "prob": float(p)}
                for p, i in zip(top_p, top_idx)
            ]
            argmax_tid = int(top_idx[0].item())
            argmax_str = tokenizer.decode([argmax_tid])

            sorted_tids = last_logits[b].argsort(descending=True)
            rank_lookup = torch.empty_like(sorted_tids)
            rank_lookup[sorted_tids] = torch.arange(
                len(sorted_tids), device=sorted_tids.device
            )

            match_en  = match_simple(argmax_str, " " + en_object) if en_object else False
            match_tgt = match_simple(argmax_str, " " + tgt_object) if tgt_object else False

            records[l] = {
                "layer":         l + 1,             # 1-indexed for plotting
                "argmax_token":  argmax_str,
                "argmax_id":     argmax_tid,
                "top":           top,
                "rank_obj_en":   int(rank_lookup[en_obj_first].item())  if en_obj_first  is not None else None,
                "rank_obj_tgt":  int(rank_lookup[tgt_obj_first].item()) if tgt_obj_first is not None else None,
                "prob_obj_en":   float(probs[b, en_obj_first].item())   if en_obj_first  is not None else None,
                "prob_obj_tgt":  float(probs[b, tgt_obj_first].item())  if tgt_obj_first is not None else None,
                "match_en":      match_en,
                "match_tgt":     match_tgt,
                "rescue":        match_en or match_tgt,
            }
    return records


# ── Main loop ───────────────────────────────────────────────────────────────
print(f"\nProcessing {len(to_process)} (fact, lang) pairs ...")

# Group by fact so EN extraction is shared across that fact's target languages.
grouped = defaultdict(list)
for key, lang, en_rec, rec, status in to_process:
    grouped[key].append((lang, en_rec, rec, status))

n_subj_not_found_en  = 0
n_subj_not_found_tgt = 0
n_records_written    = 0

for key, items in tqdm(grouped.items(), desc="Facts", file=sys.stderr):
    index, relation = key
    opath = out_path(index, relation)

    if os.path.exists(opath):
        with open(opath, "r", encoding="utf-8") as f:
            file_data = json.load(f)
    else:
        file_data = {"index": index, "relation": relation}

    # Skip if every requested lang for this fact is already present.
    pending = [t for t in items
               if not (t[0] in file_data and isinstance(file_data.get(t[0]), dict))]
    if not pending:
        continue

    # ── EN extraction (once per fact, reused across all target langs) ──
    en_rec     = items[0][1]
    en_prompt  = en_rec["few_shot_prompt"]
    en_subject = en_rec["subject"]
    en_object  = en_rec["object"]

    en_ids      = tokenizer(en_prompt, return_tensors="pt").input_ids
    en_ids_list = en_ids[0].tolist()
    en_ids      = en_ids.to(model.device)

    en_subj_pos = find_subject_last_token(en_ids_list, en_subject, tokenizer)
    if en_subj_pos is None:
        n_subj_not_found_en += 1
        continue

    en_hidden = extract_subject_hidden(en_ids, en_subj_pos)
    en_subj_first = first_token_id(en_subject, tokenizer)
    en_obj_first  = first_token_id(en_object,  tokenizer)

    changed = False

    for lang, _, rec, status in pending:
        tgt_prompt  = rec["few_shot_prompt"]
        tgt_subject = rec["subject"]
        tgt_object  = rec["object"]

        tgt_ids      = tokenizer(tgt_prompt, return_tensors="pt").input_ids
        tgt_ids_list = tgt_ids[0].tolist()
        tgt_ids      = tgt_ids.to(model.device)

        tgt_subj_pos = find_subject_last_token(tgt_ids_list, tgt_subject, tokenizer)
        if tgt_subj_pos is None:
            n_subj_not_found_tgt += 1
            file_data[lang] = {
                "status":         status,
                "skipped_reason": "subject_not_found_in_target",
            }
            changed = True
            continue

        tgt_subj_first = first_token_id(tgt_subject, tokenizer)
        tgt_obj_first  = first_token_id(tgt_object,  tokenizer)

        # (B): extract TGT subject hidden states + logit-lens BOTH sides.
        tgt_hidden = extract_subject_hidden(tgt_ids, tgt_subj_pos)

        tokens_of_interest = {
            "subj_en":  en_subj_first,
            "subj_tgt": tgt_subj_first,
            "obj_en":   en_obj_first,
            "obj_tgt":  tgt_obj_first,
        }
        lens_en  = [lens_record(h, tokens_of_interest) for h in en_hidden]
        lens_tgt = [lens_record(h, tokens_of_interest) for h in tgt_hidden]

        # (C): patch EN subject hidden state into TGT forward, layer by layer.
        patch_records = patch_all_layers(
            tgt_ids, tgt_subj_pos, en_hidden,
            en_object, tgt_object, en_obj_first, tgt_obj_first
        )

        rescue_layers_en  = [r["layer"] for r in patch_records if r["match_en"]]
        rescue_layers_tgt = [r["layer"] for r in patch_records if r["match_tgt"]]
        any_rescue_en     = bool(rescue_layers_en)
        any_rescue_tgt    = bool(rescue_layers_tgt)
        first_rescue_layer = (
            min(rescue_layers_en + rescue_layers_tgt)
            if (rescue_layers_en or rescue_layers_tgt) else None
        )

        file_data[lang] = {
            "status":              status,
            "subject_en":          en_subject,
            "subject_tgt":         tgt_subject,
            "object_en":           en_object,
            "object_tgt":          tgt_object,
            "prediction_unpatched": rec.get("prediction"),
            "subject_pos_en":      en_subj_pos,
            "subject_pos_tgt":     tgt_subj_pos,
            "tokens_of_interest": {
                "subj_en_first":  en_subj_first,
                "subj_tgt_first": tgt_subj_first,
                "obj_en_first":   en_obj_first,
                "obj_tgt_first":  tgt_obj_first,
            },
            "lens_en_subject":  lens_en,         # (B) decode-ability of EN subj hidden
            "lens_tgt_subject": lens_tgt,        # (B) decode-ability of TGT subj hidden
            "patch_per_layer":  patch_records,   # (C) intervention results
            "rescue_summary": {
                "any_rescue_en":      any_rescue_en,
                "any_rescue_tgt":     any_rescue_tgt,
                "any_rescue":         any_rescue_en or any_rescue_tgt,
                "rescue_layers_en":   rescue_layers_en,
                "rescue_layers_tgt":  rescue_layers_tgt,
                "first_rescue_layer": first_rescue_layer,
            },
        }
        changed = True
        n_records_written += 1

    if changed:
        tmp = opath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(file_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, opath)


# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\nDone. Records written this run: {n_records_written}")
print(f"  EN  subject not found: {n_subj_not_found_en}")
print(f"  TGT subject not found: {n_subj_not_found_tgt}")

meta = {
    "model_name":              args.model_name,
    "existing_cache":          args.existing_cache,
    "split":                   args.split,
    "num_layers":              num_layers,
    "total_pairs_requested":   len(to_process),
    "records_written_this_run": n_records_written,
    "subject_not_found_en":    n_subj_not_found_en,
    "subject_not_found_tgt":   n_subj_not_found_tgt,
    "status_counts":           dict(status_counts),
    "layer_batch_size":        args.layer_batch_size,
    "top_k_lens":              args.top_k_lens,
    "top_k_patch":             args.top_k_patch,
    "seed":                    args.seed,
}
meta_path = os.path.join(out_root, "metadata.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nOutput:   {out_dir}")
print(f"Metadata: {meta_path}")
