"""
cacher_patchscopes.py
---------------------
Identity patchscopes on the SUBJECT token for already-cached examples.

For each example in the requested split, loads the cached few-shot prompt,
runs a forward pass, extracts the hidden state at the last subject token
at every layer, patches it into "Repeat this:X,X,X,X,X,", and generates.

This tests: does the model recognize the SUBJECT at each layer?
If the model can't decode "Charles II" from the subject hidden state,
it has no chance of retrieving factual knowledge about him.

Reads the existing cache produced by cacher.py to:
  - Get few-shot prompts
  - Get subject names
  - Identify which examples to process (success/failure status)

Output: <model_tag>_section6_patchscopes/examples/
        One JSON per (index, relation), same structure as other cachers.

Usage:
    python cacher_patchscopes.py \
        --model_name meta-llama/Llama-3.1-8B \
        --existing_cache meta-llama_Llama-3.1-8B_section6/examples \
        --split fail_en_ok
"""

import os
import sys
import json
import glob
import random
import argparse
import unicodedata
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import CACHE_ROOT

# Force unbuffered output so prints appear immediately on cluster
sys.stdout.reconfigure(line_buffering=True)


# ── Args ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--existing_cache", type=str, required=True,
                    help="Path to existing cacher.py output, e.g. "
                         "cache/meta-llama_Llama-3.1-8B_section6/examples")
parser.add_argument("--split", type=str, default="fail_en_ok",
                    choices=["fail_en_ok", "fail_all", "success", "all"],
                    help="Which examples to process.")
parser.add_argument("--patchscope_prompt", type=str, default=None,
                    help="Patchscope inspection prompt. Auto-selected per model if not set.")
parser.add_argument("--replace_token", type=str, default=None,
                    help="Placeholder token in patchscope prompt. Auto-selected per model if not set.")
parser.add_argument("--max_new_tokens", type=int, default=10,
                    help="Max tokens to generate from patchscope prompt.")
parser.add_argument("--max_examples", type=int, default=None,
                    help="Cap total examples (for quick testing).")
parser.add_argument("--max_success_sample", type=int, default=500)
parser.add_argument("--layer_batch_size", type=int, default=32,
                    help="How many layers to batch in one generate call. "
                         "Reduce if OOM.")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ── Per-model patchscope config ─────────────────────────────────────────────
# Maps lowercase substring of model_name -> (prompt, placeholder).
# Llama-2 needs no leading space before X; all others do.
_MODEL_PS_CONFIGS = {
    "llama-2": ("Repeat this X X X", "X"),
}
_DEFAULT_PS_CONFIG = ("Repeat this X X X", " X")

if args.patchscope_prompt is None or args.replace_token is None:
    _cfg_prompt, _cfg_ph = _DEFAULT_PS_CONFIG
    for _key, (_p, _ph) in _MODEL_PS_CONFIGS.items():
        if _key in args.model_name.lower():
            _cfg_prompt, _cfg_ph = _p, _ph
            break
    if args.patchscope_prompt is None:
        args.patchscope_prompt = _cfg_prompt
    if args.replace_token is None:
        args.replace_token = _cfg_ph


# ── Seed ────────────────────────────────────────────────────────────────────
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# ── Matching (same as analysis scripts) ─────────────────────────────────────
def _norm(s):
    return unicodedata.normalize("NFC", s.lower().strip())

def _pfx(a, b):
    a, b = _norm(a), _norm(b)
    return len(a) > 0 and b.startswith(a)

def match_fixed(pred, target, rec):
    if _pfx(pred, target) or _pfx(target, pred):
        return True
    p, t = _norm(pred), _norm(target)
    ll = rec.get("logit_lens", {})
    ptid = ll.get("predicted_token_id")
    ctid = ll.get("correct_object_token_id")
    if len(p) > 4 and t.endswith(p) and p != t:
        return True
    if len(p) > 2 and t.endswith(p) and p != t and ptid is not None and ptid == ctid:
        return True
    if 0 < len(p) <= 2 and ptid is not None and ptid == ctid:
        return True
    return False


# ── Subject matching ────────────────────────────────────────────────────────
def subject_match(generated, subject):
    """Check if patchscope output indicates subject recognition."""
    gen = _norm(generated)
    sub = _norm(subject)
    if not gen or not sub:
        return False
    # Direct containment (either direction)
    if sub in gen or gen in sub:
        return len(gen) >= 2
    # First-word match for multi-word subjects
    first_part = gen.split(",")[0].strip()
    gen_first = first_part.split()[0] if first_part.split() else ""
    sub_first = sub.split()[0] if sub.split() else ""
    return len(gen_first) >= 3 and (gen_first == sub_first)


def find_subject_last_token(input_ids_list, subject, tokenizer):
    """Find position of the last token of the subject's last occurrence."""
    for prefix in [" ", ""]:
        sub_ids = tokenizer.encode(prefix + subject, add_special_tokens=False)
        if not sub_ids:
            continue
        for start in range(len(input_ids_list) - len(sub_ids), -1, -1):
            if input_ids_list[start:start + len(sub_ids)] == sub_ids:
                return start + len(sub_ids) - 1
    # Fallback: last word of subject
    last_word = subject.split()[-1] if subject.split() else subject
    for prefix in [" ", ""]:
        sub_ids = tokenizer.encode(prefix + last_word, add_special_tokens=False)
        if not sub_ids:
            continue
        for start in range(len(input_ids_list) - len(sub_ids), -1, -1):
            if input_ids_list[start:start + len(sub_ids)] == sub_ids:
                return start + len(sub_ids) - 1
    return None


# ── Load existing cache ────────────────────────────────────────────────────
print(f"Loading existing cache from {args.existing_cache} ...")
existing = {}

for path in sorted(glob.glob(os.path.join(args.existing_cache, "*.json"))):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    key = (data["index"], data["relation"])
    existing[key] = data

print(f"  Loaded {len(existing)} fact files.")

# ── Discover languages ─────────────────────────────────────────────────────
all_langs = set()
for data in existing.values():
    for k, v in data.items():
        if k not in ("index", "relation") and isinstance(v, dict):
            all_langs.add(k)
all_langs = sorted(all_langs)
non_en = [l for l in all_langs if l != "en"]
print(f"  Languages: {all_langs}")


# ── Select examples ────────────────────────────────────────────────────────
to_process = []

for key, data in existing.items():
    en_rec = data.get("en")
    if not en_rec or not isinstance(en_rec, dict):
        continue
    en_ok = match_fixed(en_rec.get("prediction", ""),
                        " " + en_rec.get("object", ""), en_rec)

    for lang in non_en:
        rec = data.get(lang)
        if not rec or not isinstance(rec, dict):
            continue
        if "few_shot_prompt" not in rec:
            continue

        tgt_ok = match_fixed(rec.get("prediction", ""),
                             " " + rec.get("object", ""), rec)

        if tgt_ok:
            status = "success"
        elif en_ok:
            status = "fail_en_ok"
        else:
            status = "fail_en_bad"

        if args.split == "fail_en_ok" and status != "fail_en_ok":
            continue
        if args.split == "fail_all" and status == "success":
            continue
        if args.split == "success" and status != "success":
            continue

        to_process.append((key, lang, rec, status))

if args.split == "all" and args.max_success_sample:
    successes = [x for x in to_process if x[3] == "success"]
    non_succ  = [x for x in to_process if x[3] != "success"]
    if len(successes) > args.max_success_sample:
        random.shuffle(successes)
        successes = successes[:args.max_success_sample]
    to_process = non_succ + successes

if args.max_examples:
    random.shuffle(to_process)
    to_process = to_process[:args.max_examples]

status_counts = defaultdict(int)
for _, _, _, s in to_process:
    status_counts[s] += 1
print(f"\nExamples to process: {len(to_process)}")
for s, c in sorted(status_counts.items()):
    print(f"  {s}: {c}")


# ── Output dir ─────────────────────────────────────────────────────────────
model_tag = args.model_name.replace("/", "_")
out_dir = os.path.join(str(CACHE_ROOT), f"{model_tag}_section6_patchscopes", "examples")
os.makedirs(out_dir, exist_ok=True)

def out_path(index, relation):
    return os.path.join(out_dir, f"{index}__{relation}.json")


# ── Model ──────────────────────────────────────────────────────────────────
print(f"\nLoading model: {args.model_name}")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name, trust_remote_code=True, device_map="auto"
)
model.eval()

num_layers = model.config.num_hidden_layers


# ── Patchscope prompt setup ────────────────────────────────────────────────
print(f"\nPatchscope prompt: '{args.patchscope_prompt}'")
print(f"Replace token:     '{args.replace_token}'")

ps_input_ids = tokenizer.encode(args.patchscope_prompt, add_special_tokens=False)
ps_tensor = torch.tensor(ps_input_ids, device=model.device)

# Find replacement positions: try multiple tokenizations of the replace token.
# E.g. Llama merges "X," into a single token, so "X" alone won't match.
replace_candidates = [
    args.replace_token,             # "X"
    args.replace_token + ",",       # "X,"  (often merged in BPE)
    " " + args.replace_token,       # " X"
    " " + args.replace_token + ",", # " X,"
]
replace_tids = set()
for cand in replace_candidates:
    ids = tokenizer.encode(cand, add_special_tokens=False)
    if len(ids) == 1:
        replace_tids.add(ids[0])

replace_mask = torch.zeros_like(ps_tensor, dtype=torch.bool)
for tid in replace_tids:
    replace_mask = replace_mask | (ps_tensor == tid)

# If still nothing found, fall back: find the most-repeated token in the prompt
# (the X slots are the repeated part)
if not replace_mask.any():
    from collections import Counter
    counts = Counter(ps_input_ids)
    most_common_tid, most_common_count = counts.most_common(1)[0]
    if most_common_count >= 3:
        replace_mask = (ps_tensor == most_common_tid)
        print(f"  Fallback: using most-repeated token {most_common_tid} "
              f"('{tokenizer.decode([most_common_tid])}', appears {most_common_count}x)")

n_replace = replace_mask.sum().item()
print(f"  Prompt token IDs: {ps_input_ids}")
print(f"  Replace positions: {replace_mask.nonzero().flatten().tolist()} ({n_replace} slots)")
if n_replace == 0:
    print("  WARNING: no replacement positions found! Check replace_token.")

ps_len = len(ps_input_ids)  # for stripping prompt from output


# ── Pre-norm hook for last layer ───────────────────────────────────────────
# hidden_states[-1] from HF is post-final-norm; we need pre-norm for patching.
pre_norm_state = {}

def pre_norm_hook(module, args_tuple):
    pre_norm_state["h"] = args_tuple[0].clone()
    return args_tuple

# Identify final norm module
def get_final_norm(model):
    candidates = [
        "model.norm", "model.final_layernorm", "model.layer_norm",
        "transformer.ln_f", "transformer.norm", "gpt_neox.final_layer_norm",
    ]
    for attr in candidates:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"Cannot find final LayerNorm. Tried: {candidates}")

norm_module = get_final_norm(model)


# ── Patchscope runner ──────────────────────────────────────────────────────
embedding_layer = model.get_input_embeddings()
with torch.no_grad():
    base_ps_embeds = embedding_layer(ps_tensor).unsqueeze(0)  # [1, ps_len, hidden]

# Constant: positions in the patchscope prompt that get patched. Compute once.
replace_positions = replace_mask.nonzero(as_tuple=False).flatten()


@torch.no_grad()
def run_patchscopes(layer_hidden):
    """
    Patch each pre-extracted subject hidden state into the patchscope prompt
    and generate.

    Args:
        layer_hidden: list of 1D tensors [hidden_dim], one per transformer layer.
    Returns:
        List of generated strings, one per layer.
    """
    results = []
    total_layers = len(layer_hidden)

    for batch_start in range(0, total_layers, args.layer_batch_size):
        batch_end = min(batch_start + args.layer_batch_size, total_layers)
        batch_h = layer_hidden[batch_start:batch_end]
        bsz = len(batch_h)

        batched_embeds = base_ps_embeds.repeat(bsz, 1, 1)  # [bsz, ps_len, hidden]
        # Vectorized patch: assign batch_h[b] to every replace_positions index for each b.
        # batch_h_stacked: [bsz, 1, hidden] broadcasts over n_replace positions.
        batch_h_stacked = torch.stack(batch_h, dim=0).unsqueeze(1)
        batched_embeds[:, replace_positions, :] = batch_h_stacked

        outputs = model.generate(
            inputs_embeds=batched_embeds,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        for output in outputs:
            gen_tokens = output[ps_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            results.append(gen_text)

    return results


# ── Main loop ──────────────────────────────────────────────────────────────
print(f"\nProcessing {len(to_process)} examples ...")

grouped = defaultdict(list)
for key, lang, rec, status in to_process:
    grouped[key].append((lang, rec, status))

n_subject_not_found = 0

for key, items in tqdm(grouped.items(), desc="Facts", file=sys.stderr):
    index, relation = key
    opath = out_path(index, relation)

    if os.path.exists(opath):
        with open(opath, "r", encoding="utf-8") as f:
            file_data = json.load(f)
    else:
        file_data = {"index": index, "relation": relation}

    changed = False

    for lang, rec, status in items:
        if lang in file_data and isinstance(file_data.get(lang), dict):
            continue

        few_shot_prompt = rec["few_shot_prompt"]
        subject = rec.get("subject", "")

        input_ids = tokenizer(few_shot_prompt, return_tensors="pt").input_ids
        input_ids_list = input_ids[0].tolist()
        input_ids = input_ids.to(model.device)

        # Find last subject token
        subj_pos = find_subject_last_token(input_ids_list, subject, tokenizer)
        if subj_pos is None:
            n_subject_not_found += 1
            continue

        # Forward pass with hidden states
        hook_handle = norm_module.register_forward_pre_hook(pre_norm_hook)
        with torch.no_grad():
            fwd = model(input_ids, output_hidden_states=True)
        hook_handle.remove()

        # Extract subject hidden state per layer. .clone() detaches each slice
        # from the full activation tensor so the rest can be freed before generate.
        # Layers 1..N-1 use post-block hidden_states[i]; layer N uses pre-norm.
        hs = fwd.hidden_states
        n_total = len(hs)  # embedding + N transformer layers
        layer_hidden = []
        for i in range(1, n_total):
            if i < n_total - 1:
                h = hs[i][0, subj_pos, :].clone()
            else:
                h = pre_norm_state["h"][0, subj_pos, :].clone()
            layer_hidden.append(h)

        # Free the full forward-pass activations (~num_layers * seq_len * hidden floats)
        # before running the generate batch.
        del fwd, hs
        pre_norm_state.clear()

        # Run patchscopes
        per_layer_text = run_patchscopes(layer_hidden)
        per_layer_match = [subject_match(t, subject) for t in per_layer_text]

        first_success = None
        last_success = None
        for i, m in enumerate(per_layer_match):
            if m:
                if first_success is None:
                    first_success = i + 1   # 1-indexed layer
                last_success = i + 1

        record = {
            "status": status,
            "subject": subject,
            "object": rec.get("object"),
            "prediction": rec.get("prediction"),
            "subject_token_pos": subj_pos,
            "patchscopes": {
                "per_layer_text": per_layer_text,
                "per_layer_match": per_layer_match,
                "first_success_layer": first_success,
                "last_success_layer": last_success,
                "patchscopes_success": any(per_layer_match),
            }
        }

        file_data[lang] = record
        changed = True

    if changed:
        with open(opath, "w", encoding="utf-8") as f:
            json.dump(file_data, f, ensure_ascii=False, indent=2)


# ── Summary ────────────────────────────────────────────────────────────────
print(f"\nSubject token not found: {n_subject_not_found}")

# ── Metadata ───────────────────────────────────────────────────────────────
meta = {
    "model_name": args.model_name,
    "existing_cache": args.existing_cache,
    "split": args.split,
    "patchscope_prompt": args.patchscope_prompt,
    "replace_token": args.replace_token,
    "max_new_tokens": args.max_new_tokens,
    "num_layers": num_layers,
    "total_processed": len(to_process),
    "subject_not_found": n_subject_not_found,
    "status_counts": dict(status_counts),
    "seed": args.seed,
}
meta_path = os.path.join(str(CACHE_ROOT), f"{model_tag}_section6_patchscopes", "metadata.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nDone. Output: {out_dir}")
print(f"Metadata: {meta_path}")
