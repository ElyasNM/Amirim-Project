"""
cacher.py
-----------------
Phase 1+2 combined: for every example in the dataset, run generation +
Logit Lens and save a detailed JSON to disk.

Resumable: if the JSON for an example already exists it is skipped, so you
can interrupt and restart freely.

After all examples are processed, a second pass fills in the `en_match`
field (whether the English version of the same fact was answered correctly),
which requires knowing all results first.

Output directory: <model_tag>_section6/examples/
One file per fact: <index>__<relation>.json  (all languages inside)
"""

import os
import json
import glob
import random
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
from tqdm import tqdm
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import CACHE_ROOT, KLAR_DIR

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_name",  type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--seed",        type=int, default=12345)
parser.add_argument("--n_shot",      type=int, default=3)
parser.add_argument("--max_samples", type=int, default=None,
                    help="Cap dataset size for quick testing (e.g. 200)")
parser.add_argument("--data_dir",    type=str, default=str(KLAR_DIR))
parser.add_argument("--languages",   type=str, default=None,
                    help="Comma-separated language codes (e.g. 'en,fr,de'). "
                         "Auto-detected from data_dir if omitted.")
parser.add_argument("--skip_patchscopes", action="store_true",
                    help="Skip patchscopes (only run generation + logit lens).")
parser.add_argument("--patchscope_prompt", type=str,
                    default="Repeat this:X,X,X,X,X,")
parser.add_argument("--replace_token", type=str, default="X")
parser.add_argument("--ps_max_new_tokens", type=int, default=10)
args = parser.parse_args()

model_name = args.model_name

# ── Seed ──────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# ── Output dir ────────────────────────────────────────────────────────────────
model_tag  = model_name.replace("/", "_")
cache_dir  = os.path.join(str(CACHE_ROOT), f"{model_tag}_section6", "examples")
os.makedirs(cache_dir, exist_ok=True)

def example_path(index, relation):
    safe = str(index).replace("/", "_")
    return os.path.join(cache_dir, f"{safe}__{relation}.json")

# ── Data loading ─────────────────────────────────────────────────────────────
relations = [
    "applies_to_jurisdiction", "capital", "capital_of", "continent",
    "country_of_citizenship", "developer", "field_of_work", "headquarters_location",
    "instrument", "language_of_work_or_name", "languages_spoken", "location_of_formation",
    "manufacturer", "native_language", "occupation", "official_language",
    "owned_by", "place_of_birth", "place_of_death", "religion"
]

# Auto-detect languages from data directory, or use --languages override
if args.languages:
    valid_langs = set(args.languages.split(","))
else:
    valid_langs = {d for d in os.listdir(args.data_dir)
                   if os.path.isdir(os.path.join(args.data_dir, d))}
    if not valid_langs:
        raise ValueError(f"No language directories found in {args.data_dir}")
print(f"Languages: {sorted(valid_langs)}")

valid_rels  = set(relations)

json_paths = glob.glob(os.path.join(args.data_dir, "*", "*.json"))
path_map   = defaultdict(dict)
for path in json_paths:
    lang = os.path.basename(os.path.dirname(path))
    rel  = os.path.splitext(os.path.basename(path))[0]
    if lang in valid_langs and rel in valid_rels:
        path_map[rel][lang] = path

samples = []
for rel, lang_paths in path_map.items():
    if not all(lang in lang_paths for lang in valid_langs):
        continue
    for lang in valid_langs:
        with open(lang_paths[lang], "r", encoding="utf-8") as f:
            content  = json.load(f)
            template = content["prompt_templates"][0]
            for sample in content["samples"]:
                prompt = template.replace("<subject>", sample["subject"]).replace("<mask>", "").strip()
                samples.append({
                    "subject":  sample["subject"],
                    "object":   sample["object"],
                    "language": lang,
                    "relation": rel,
                    "template": template,
                    "input":    prompt,
                    "target":   " " + sample["object"],
                    "index":    sample["index"],
                })

if args.max_samples is not None:
    samples = samples[:args.max_samples]

print(f"Total examples: {len(samples)}")

# ── Model ─────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
model.eval()

# ── Logit Lens helpers ────────────────────────────────────────────────────────
def get_final_norm(model, model_name):
    """Find the final layer norm. Supports Llama, Qwen, Qwen2, OLMo, Bloom, etc."""
    # Try known paths in priority order
    candidates = [
        "model.norm",           # Llama, Qwen2, OLMo2, Gemma
        "model.final_layernorm",# OLMo1
        "model.layer_norm",     # some OLMo variants
        "transformer.ln_f",    # Bloom, GPT-2, GPT-Neo, older Qwen
        "transformer.norm",    # some Qwen variants
        "gpt_neox.final_layer_norm",  # GPT-NeoX, Pythia
    ]
    for attr in candidates:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError(
        f"Cannot find final LayerNorm for {model_name}. "
        f"Tried: {candidates}. Check your model's architecture."
    )

final_norm = get_final_norm(model, model_name)
lm_head    = model.lm_head

@torch.no_grad()
def logit_lens_ranks(hidden_states, token_ids_of_interest):
    """
    Returns {token_id: [rank_layer1, rank_layer2, ...]} for each requested token.
    Rank 0 = most probable. Index 0 of hidden_states is the raw embedding; we skip it.
    """
    ranks = {tid: [] for tid in token_ids_of_interest}
    for layer_h in hidden_states[1:]:
        last    = layer_h[0, -1, :].unsqueeze(0)              # (1, hidden_dim)
        logits  = lm_head(final_norm(last)).squeeze(0)        # (vocab_size,)
        order   = torch.argsort(logits, descending=True)
        rank_map = torch.empty_like(order)
        rank_map[order] = torch.arange(len(order), device=order.device)
        for tid in token_ids_of_interest:
            ranks[tid].append(rank_map[tid].item())
    return ranks

# ── Matching ─────────────────────────────────────────────────────────────────
def is_nontrivial_prefix(pred, target):
    pred, target = pred.lower().strip(), target.lower().strip()
    return len(pred) > 0 and target.startswith(pred)

def is_match(prediction, target):
    return is_nontrivial_prefix(prediction, target) or is_nontrivial_prefix(target, prediction)

# ── Patchscopes helpers ──────────────────────────────────────────────────────
import unicodedata

def _norm_ps(s):
    return unicodedata.normalize("NFC", s.lower().strip())

def subject_match(generated, subject):
    gen = _norm_ps(generated)
    sub = _norm_ps(subject)
    if not gen or not sub:
        return False
    if sub in gen or gen in sub:
        return len(gen) >= 2
    first_part = gen.split(",")[0].strip()
    gen_first = first_part.split()[0] if first_part.split() else ""
    sub_first = sub.split()[0] if sub.split() else ""
    return len(gen_first) >= 3 and gen_first == sub_first

def find_subject_last_token(input_ids_list, subject, tokenizer):
    for prefix in [" ", ""]:
        sub_ids = tokenizer.encode(prefix + subject, add_special_tokens=False)
        if not sub_ids:
            continue
        for start in range(len(input_ids_list) - len(sub_ids), -1, -1):
            if input_ids_list[start:start + len(sub_ids)] == sub_ids:
                return start + len(sub_ids) - 1
    last_word = subject.split()[-1] if subject.split() else subject
    for prefix in [" ", ""]:
        sub_ids = tokenizer.encode(prefix + last_word, add_special_tokens=False)
        if not sub_ids:
            continue
        for start in range(len(input_ids_list) - len(sub_ids), -1, -1):
            if input_ids_list[start:start + len(sub_ids)] == sub_ids:
                return start + len(sub_ids) - 1
    return None

# ── Patchscope setup ─────────────────────────────────────────────────────────
if not args.skip_patchscopes:
    ps_input_ids = tokenizer.encode(args.patchscope_prompt, add_special_tokens=False)
    ps_tensor = torch.tensor(ps_input_ids, device=model.device)
    replace_candidates = [
        args.replace_token, args.replace_token + ",",
        " " + args.replace_token, " " + args.replace_token + ",",
    ]
    replace_tids = set()
    for cand in replace_candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            replace_tids.add(ids[0])
    ps_replace_mask = torch.zeros_like(ps_tensor, dtype=torch.bool)
    for tid in replace_tids:
        ps_replace_mask = ps_replace_mask | (ps_tensor == tid)
    if not ps_replace_mask.any():
        from collections import Counter
        counts = Counter(ps_input_ids)
        most_common_tid, mc = counts.most_common(1)[0]
        if mc >= 3:
            ps_replace_mask = (ps_tensor == most_common_tid)
            print(f"  Fallback: using most-repeated token {most_common_tid} "
                  f"('{tokenizer.decode([most_common_tid])}', appears {mc}x)")
    ps_len = len(ps_input_ids)
    ps_base_embeds = model.get_input_embeddings()(ps_tensor).unsqueeze(0)
    ps_replace_positions = ps_replace_mask.nonzero(as_tuple=False).flatten()

    # Pre-norm hook for last layer
    ps_pre_norm_state = {}
    def ps_pre_norm_hook(module, args_tuple):
        ps_pre_norm_state["h"] = args_tuple[0].clone()
        return args_tuple

    print(f"Patchscopes enabled: prompt='{args.patchscope_prompt}', "
          f"{ps_replace_mask.sum().item()} replace slots")

# ── Main loop ─────────────────────────────────────────────────────────────────
# Group by (index, relation) for demo sampling
grouped = defaultdict(dict)
for ex in samples:
    grouped[(ex["index"], ex["relation"])][ex["language"]] = ex

print(f"\nProcessing {len(samples)} examples (resumable — skipping existing files)...\n")

for ex in tqdm(samples):
    lang, relation, index = ex["language"], ex["relation"], ex["index"]
    out_path = example_path(index, relation)

    # ── Load existing file (other languages may already be cached) ────────────
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            file_data = json.load(f)
    else:
        file_data = {"index": index, "relation": relation}

    # ── Resume: skip if this language already cached ───────────────────────────
    if lang in file_data:
        continue

    # ── Build few-shot prompt (saved so Phase 3 is consistent) ────────────────
    candidates = [c for c in samples
                  if c["index"] != index and c["language"] == lang and c["relation"] == relation]
    demos      = random.sample(candidates, min(args.n_shot, len(candidates)))
    few_shot_prompt = "".join(f"{d['input']}{d['target']}\n" for d in demos) + ex["input"]

    input_ids = tokenizer(few_shot_prompt, return_tensors="pt").input_ids.to(model.device)

    # ── Generation ────────────────────────────────────────────────────────────
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=10)

    # FIX: It was `decoded[len(few_shot_prompt):]` which caused truncated
    # predictions because tokenizer.decode(tokenizer.encode(text)) can differ
    # in length from the original text (Unicode normalization, whitespace).
    # So I fixed to decode only the generated tokens using the token-level
    # boundary (input_ids.shape[1]), which is exact regardless of string length.
    generated  = output[0][input_ids.shape[1]:]
    prediction = tokenizer.decode(generated, skip_special_tokens=True).split('\n')[0].strip()
    match      = is_match(prediction, ex["target"])

    # ── Logit Lens ────────────────────────────────────────────────────────────
    with torch.no_grad():
        fwd = model(input_ids, output_hidden_states=True)

    hidden_states = fwd.hidden_states  # (num_layers+1) x (1, seq_len, hidden)

    # Derive predicted_token_id from the same forward pass used for logit lens,
    # so that predicted_token_ranks[-1] is guaranteed to be rank 0 by construction.
    pred_token_id = int(fwd.logits[0, -1, :].argmax())

    # Token ids to track
    correct_obj_ids = tokenizer(" " + ex["object"],    add_special_tokens=False).input_ids
    correct_obj_tid = correct_obj_ids[0] if correct_obj_ids else None

    # English correct answer token (look up from grouped)
    en_ex           = grouped[(index, relation)].get("en")
    correct_en_ids  = tokenizer(" " + en_ex["object"], add_special_tokens=False).input_ids \
                      if en_ex else []
    correct_en_tid  = correct_en_ids[0] if correct_en_ids else None

    # Collect unique, valid token ids
    tids_to_track = {tid for tid in [correct_obj_tid, correct_en_tid, pred_token_id]
                     if tid is not None}

    ranks = logit_lens_ranks(hidden_states, tids_to_track)

    # ── Patchscopes ──────────────────────────────────────────────────────────
    ps_record = None
    if not args.skip_patchscopes:
        input_ids_list = input_ids[0].tolist()
        subj_pos = find_subject_last_token(input_ids_list, ex["subject"], tokenizer)
        if subj_pos is not None:
            # Re-run forward pass with pre-norm hook
            hook_handle = final_norm.register_forward_pre_hook(ps_pre_norm_hook)
            with torch.no_grad():
                fwd_ps = model(input_ids, output_hidden_states=True)
            pre_norm_h = ps_pre_norm_state["h"].clone()
            hook_handle.remove()

            hs = fwd_ps.hidden_states
            layer_hidden = []
            for i, layer_h in enumerate(hs[1:]):
                if i < len(hs) - 2:
                    layer_hidden.append(layer_h[0, subj_pos, :])
                else:
                    layer_hidden.append(pre_norm_h[0, subj_pos, :])

            batched_embeds = ps_base_embeds.repeat(len(layer_hidden), 1, 1)
            for b in range(len(layer_hidden)):
                for pos in ps_replace_positions:
                    batched_embeds[b, pos] = layer_hidden[b]

            with torch.no_grad():
                ps_out = model.generate(
                    inputs_embeds=batched_embeds,
                    max_new_tokens=args.ps_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            per_layer_text = []
            for out_seq in ps_out:
                gen_text = tokenizer.decode(out_seq[ps_len:], skip_special_tokens=True).strip()
                per_layer_text.append(gen_text)

            per_layer_match = [subject_match(t, ex["subject"]) for t in per_layer_text]
            first_succ = next((i+1 for i, m in enumerate(per_layer_match) if m), None)
            last_succ  = next((len(per_layer_match)-i for i, m in enumerate(reversed(per_layer_match)) if m), None)

            ps_record = {
                "subject_token_pos": subj_pos,
                "per_layer_text": per_layer_text,
                "per_layer_match": per_layer_match,
                "first_success_layer": first_succ,
                "last_success_layer": last_succ,
                "patchscopes_success": any(per_layer_match),
            }

    # ── Build output record ───────────────────────────────────────────────────
    record = {
        # ---- identity ----
        "subject":          ex["subject"],
        "object":           ex["object"],
        # ---- prompt ----
        "prompt":           ex["input"],
        "few_shot_prompt":  few_shot_prompt,
        # ---- prediction ----
        "prediction":       prediction,
        "match":            bool(match),
        "en_match":         None,          # filled in post-processing pass below
        # ---- logit lens ----
        "logit_lens": {
            "correct_object_token_id":   correct_obj_tid,
            "correct_object_token":      tokenizer.decode([correct_obj_tid]) if correct_obj_tid is not None else None,
            "correct_object_ranks":      ranks.get(correct_obj_tid, []),

            "correct_en_token_id":       correct_en_tid,
            "correct_en_token":          tokenizer.decode([correct_en_tid]) if correct_en_tid is not None else None,
            "correct_en_ranks":          ranks.get(correct_en_tid, []),

            "predicted_token_id":        pred_token_id,
            "predicted_token":           tokenizer.decode([pred_token_id]) if pred_token_id is not None else None,
            "predicted_token_ranks":     ranks.get(pred_token_id, []),
        },
        # ---- patchscopes ----
        "patchscopes": ps_record,
    }

    file_data[lang] = record
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(file_data, f, ensure_ascii=False, indent=2)

# ── Post-processing: fill in en_match ─────────────────────────────────────────
print("\nFilling in en_match field...")

updated = 0
for path in glob.glob(os.path.join(cache_dir, "*.json")):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    en_match = data.get("en", {}).get("match", None)
    changed  = False
    for lang, rec in data.items():
        if lang in ("index", "relation") or not isinstance(rec, dict):
            continue
        if rec.get("en_match") != en_match:
            rec["en_match"] = en_match
            changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        updated += 1

print(f"Updated en_match in {updated} files.")

# ── Save metadata ─────────────────────────────────────────────────────────────
all_files = glob.glob(os.path.join(cache_dir, "*.json"))
meta = {
    "model_name":   model_name,
    "n_shot":       args.n_shot,
    "seed":         args.seed,
    "num_layers":   model.config.num_hidden_layers,
    "vocab_size":   model.config.vocab_size,
    "total_cached": len(all_files),
}
meta_path = os.path.join(str(CACHE_ROOT), f"{model_tag}_section6", "metadata.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nDone. Cache: {cache_dir}")
print(f"Metadata:   {meta_path}")