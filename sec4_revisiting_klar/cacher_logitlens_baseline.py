"""
cacher_logitlens_baseline.py
----------------------------
Baseline test for the "model thinks in English" hypothesis.

The original cacher tracks the correct-EN token through logit lens and
shows it rising in middle layers.  But is this specific to the *correct*
English answer, or do *all* English tokens rise?  This script adds
baseline tokens to answer that question.

For each example in the requested split, it runs a forward pass and
tracks logit-lens ranks for:

  1. correct_object_token  — correct answer in the target language
  2. correct_en_token      — correct answer in English
  3. random_en_answers     — correct EN tokens from OTHER facts (same relation)
                             Tests: does any EN answer token rise, or only the right one?
  4. random_vocab           — random vocabulary tokens
                             Tests: is the rise specific to English at all?

The script reads the existing cache produced by cacher.py to:
  - Get the few-shot prompts (avoids seed reconstruction)
  - Identify which examples to process (success/failure status)
  - Collect the pool of EN answer tokens for baseline sampling

It does NOT hardcode model names, language lists, or failure indices.
Everything is inferred from the existing cache and CLI args.

Output: <model_tag>_section6_baseline/examples/
        One JSON per (index, relation), same structure as original cache.

Usage:
    python cacher_logitlens_baseline.py \\
        --model_name meta-llama/Llama-3.1-8B \\
        --existing_cache meta-llama_Llama-3.1-8B_section6/examples \\
        --split fail_en_ok \\
        --n_random_en 10 \\
        --n_random_vocab 10 \\
        --max_success_sample 500
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
                    help="Which examples to process. "
                         "fail_en_ok = non-EN wrong + EN correct (the key cases). "
                         "fail_all = all non-EN failures. "
                         "success = non-EN successes (for comparison). "
                         "all = everything.")
parser.add_argument("--n_random_en", type=int, default=10,
                    help="Number of random EN answer tokens to track per example.")
parser.add_argument("--n_random_vocab", type=int, default=10,
                    help="Number of random vocabulary tokens to track per example.")
parser.add_argument("--max_examples", type=int, default=None,
                    help="Cap total examples to process (for quick testing).")
parser.add_argument("--max_success_sample", type=int, default=500,
                    help="When split=all, cap how many successes to include.")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()


# ── Seed ────────────────────────────────────────────────────────────────────
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# ── Matching (same as analysis scripts — needed to identify failures) ──────
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


# ── Load existing cache ────────────────────────────────────────────────────
print(f"Loading existing cache from {args.existing_cache} ...")
existing = {}          # (index, rel) -> full file dict
en_answer_tids = {}    # (index, rel) -> EN correct_object_token_id
all_en_tids = set()    # pool of all EN answer token IDs (for random sampling)
per_rel_en_tids = defaultdict(set)  # relation -> set of EN answer tids

for path in sorted(glob.glob(os.path.join(args.existing_cache, "*.json"))):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    key = (data["index"], data["relation"])
    existing[key] = data

    en_rec = data.get("en")
    if en_rec and isinstance(en_rec, dict):
        tid = en_rec.get("logit_lens", {}).get("correct_object_token_id")
        if tid is not None:
            en_answer_tids[key] = tid
            all_en_tids.add(tid)
            per_rel_en_tids[data["relation"]].add(tid)

print(f"  Loaded {len(existing)} fact files.")
print(f"  Unique EN answer token IDs: {len(all_en_tids)}")

# ── Discover languages from existing cache ──────────────────────────────────
all_langs = set()
for data in existing.values():
    for k, v in data.items():
        if k not in ("index", "relation") and isinstance(v, dict):
            all_langs.add(k)
all_langs = sorted(all_langs)
non_en = [l for l in all_langs if l != "en"]
print(f"  Languages: {all_langs}")


# ── Select examples to process ──────────────────────────────────────────────
to_process = []  # list of (key, lang, rec, status)

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
        # Need the stored few-shot prompt
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

        # Filter by requested split
        if args.split == "fail_en_ok" and status != "fail_en_ok":
            continue
        if args.split == "fail_all" and status == "success":
            continue
        if args.split == "success" and status != "success":
            continue

        to_process.append((key, lang, rec, status))

# For split=all, cap successes to avoid overwhelming
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


# ── Output dir ──────────────────────────────────────────────────────────────
model_tag = args.model_name.replace("/", "_")
out_dir = os.path.join(str(CACHE_ROOT), f"{model_tag}_section6_baseline", "examples")
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


# ── Logit Lens helpers ──────────────────────────────────────────────────────
def get_final_norm(model, model_name):
    name = model_name.lower()
    if "llama" in name:
        return model.model.norm
    elif "bloom" in name:
        return model.transformer.ln_f
    for attr in ["model.norm", "transformer.ln_f"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError(f"Cannot find final LayerNorm for {model_name}.")

final_norm = get_final_norm(model, args.model_name)
lm_head = model.lm_head
num_layers = model.config.num_hidden_layers
vocab_size = model.config.vocab_size


@torch.no_grad()
def logit_lens_ranks(hidden_states, token_ids):
    """
    For each layer's hidden state (skipping layer 0 = raw embedding),
    project through final_norm + lm_head and return the rank of each
    requested token ID.

    Returns {tid: [rank_layer1, ..., rank_layerN]} for each tid in token_ids.
    """
    ranks = {tid: [] for tid in token_ids}
    for layer_h in hidden_states[1:]:
        last = layer_h[0, -1, :].unsqueeze(0)
        logits = lm_head(final_norm(last)).squeeze(0)
        order = torch.argsort(logits, descending=True)
        rank_map = torch.empty_like(order)
        rank_map[order] = torch.arange(len(order), device=order.device)
        for tid in token_ids:
            ranks[tid].append(rank_map[tid].item())
    return ranks


# ── Build random vocab token pool ───────────────────────────────────────────
# Select tokens that decode to reasonable ASCII strings (likely English words)
print("Building random vocab pool ...")
vocab_pool = []
for tid in range(vocab_size):
    decoded = tokenizer.decode([tid])
    stripped = decoded.strip()
    # Keep tokens that are ASCII, alpha, length 3+
    if len(stripped) >= 3 and stripped.isascii() and stripped.isalpha():
        vocab_pool.append(tid)
# Remove any that are actual EN answer tokens (to keep baseline clean)
vocab_pool = [t for t in vocab_pool if t not in all_en_tids]
random.shuffle(vocab_pool)
print(f"  Vocab pool size: {len(vocab_pool)} tokens")


# ── Main loop ───────────────────────────────────────────────────────────────
print(f"\nProcessing {len(to_process)} examples ...")

# Group by (index, relation) for file-level output
grouped = defaultdict(list)
for key, lang, rec, status in to_process:
    grouped[key].append((lang, rec, status))

for key, items in tqdm(grouped.items(), desc="Facts", file=sys.stderr):
    index, relation = key
    opath = out_path(index, relation)

    # Load existing output if resuming
    if os.path.exists(opath):
        with open(opath, "r", encoding="utf-8") as f:
            file_data = json.load(f)
    else:
        file_data = {"index": index, "relation": relation}

    changed = False

    for lang, rec, status in items:
        # Skip if already cached
        if lang in file_data and isinstance(file_data.get(lang), dict):
            continue

        few_shot_prompt = rec["few_shot_prompt"]
        input_ids = tokenizer(few_shot_prompt, return_tensors="pt").input_ids.to(model.device)

        # ── Forward pass ────────────────────────────────────────────────
        with torch.no_grad():
            fwd = model(input_ids, output_hidden_states=True)
        hidden_states = fwd.hidden_states

        # ── Collect token IDs to track ──────────────────────────────────
        ll_orig = rec.get("logit_lens", {})
        correct_obj_tid = ll_orig.get("correct_object_token_id")
        correct_en_tid = ll_orig.get("correct_en_token_id")

        # Random EN answer tokens: from same relation, excluding current fact's EN token
        rel_pool = list(per_rel_en_tids[relation] - {correct_en_tid})
        if len(rel_pool) < args.n_random_en:
            # Fall back to all EN answer tids
            rel_pool = list(all_en_tids - {correct_en_tid})
        random.shuffle(rel_pool)
        random_en_tids = rel_pool[:args.n_random_en]

        # Random vocab tokens
        random.shuffle(vocab_pool)
        random_vocab_tids = vocab_pool[:args.n_random_vocab]

        # All unique tids to track in one pass
        all_tids = set()
        if correct_obj_tid is not None:
            all_tids.add(correct_obj_tid)
        if correct_en_tid is not None:
            all_tids.add(correct_en_tid)
        all_tids.update(random_en_tids)
        all_tids.update(random_vocab_tids)

        ranks = logit_lens_ranks(hidden_states, all_tids)

        # ── Build output record ─────────────────────────────────────────
        record = {
            "status": status,
            "subject": rec.get("subject"),
            "object": rec.get("object"),
            "prediction": rec.get("prediction"),

            "correct_object_token_id": correct_obj_tid,
            "correct_object_token": ll_orig.get("correct_object_token"),
            "correct_object_ranks": ranks.get(correct_obj_tid, []),

            "correct_en_token_id": correct_en_tid,
            "correct_en_token": ll_orig.get("correct_en_token"),
            "correct_en_ranks": ranks.get(correct_en_tid, []),

            "random_en_answer_token_ids": random_en_tids,
            "random_en_answer_tokens": [tokenizer.decode([t]) for t in random_en_tids],
            "random_en_answer_ranks": [ranks.get(t, []) for t in random_en_tids],

            "random_vocab_token_ids": random_vocab_tids,
            "random_vocab_tokens": [tokenizer.decode([t]) for t in random_vocab_tids],
            "random_vocab_ranks": [ranks.get(t, []) for t in random_vocab_tids],
        }

        file_data[lang] = record
        changed = True

    if changed:
        with open(opath, "w", encoding="utf-8") as f:
            json.dump(file_data, f, ensure_ascii=False, indent=2)


# ── Save metadata ───────────────────────────────────────────────────────────
meta = {
    "model_name": args.model_name,
    "existing_cache": args.existing_cache,
    "split": args.split,
    "n_random_en": args.n_random_en,
    "n_random_vocab": args.n_random_vocab,
    "num_layers": num_layers,
    "vocab_size": vocab_size,
    "total_processed": len(to_process),
    "status_counts": dict(status_counts),
    "seed": args.seed,
}
meta_path = os.path.join(str(CACHE_ROOT), f"{model_tag}_section6_baseline", "metadata.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nDone. Output: {out_dir}")
print(f"Metadata: {meta_path}")
