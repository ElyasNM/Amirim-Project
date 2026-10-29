"""
compute_binding.py
-------------------
For each unique entity-name string in binding_triplets.json, run Llama-3.1-8B
on the string alone, extract per-layer hidden states, mean-pool over the
entity's non-BOS tokens -> one vector per layer per string.

Then for every triplet compute, per layer:
  cos_translation = cosine(subject_en, subject_lang)          # same entity, cross-language
  cos_baseline    = cosine(subject_en, baseline_subject_en)   # different entity, same language (English)

Output: binding_results.csv with columns:
  relation, lang, index, layer, cos_translation, cos_baseline
"""
import sys
import os
import json
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import BINDING_DIR
parser.add_argument("--triplets_path", type=str, default=str(BINDING_DIR / "binding_triplets.json"))
parser.add_argument("--out_path", type=str, default=str(BINDING_DIR / "binding_results.csv"))
args = parser.parse_args()

print(f"Loading {args.model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16
)
model.eval()
num_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer
print(f"Model loaded. num_hidden_layers+1 = {num_layers}")

with open(args.triplets_path, encoding="utf-8") as f:
    triplets = json.load(f)
print(f"Loaded {len(triplets)} triplets")

# collect unique strings that need a forward pass
unique_strings = set()
for t in triplets:
    unique_strings.add(t["subject_en"])
    unique_strings.add(t["subject_lang"])
    unique_strings.add(t["baseline_subject_en"])
unique_strings = sorted(unique_strings)
print(f"Unique strings to encode: {len(unique_strings)}")

# vec_cache[string] -> np.ndarray [num_layers, hidden_dim], float32, mean-pooled over entity tokens
vec_cache = {}

with torch.no_grad():
    for s in tqdm(unique_strings, desc="Encoding entities"):
        input_ids = tokenizer(s, return_tensors="pt").input_ids.to(model.device)
        if input_ids.shape[1] < 2:
            # degenerate: only BOS token, nothing to pool over; fall back to BOS itself
            token_slice = slice(0, 1)
        else:
            token_slice = slice(1, None)  # drop BOS
        out = model(input_ids, output_hidden_states=True)
        # hidden_states: tuple of (num_layers) tensors [1, seq_len, hidden]
        layer_vecs = []
        for h in out.hidden_states:
            pooled = h[0, token_slice, :].mean(dim=0)  # [hidden]
            layer_vecs.append(pooled.float().cpu().numpy())
        vec_cache[s] = np.stack(layer_vecs, axis=0)  # [num_layers, hidden]

print("Encoding done. Computing cosine similarities ...")


def cosine(a, b):
    a_n = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return (a_n * b_n).sum(axis=-1)  # [num_layers]


rows = []
for t in tqdm(triplets, desc="Scoring triplets"):
    v_en = vec_cache[t["subject_en"]]
    v_lang = vec_cache[t["subject_lang"]]
    v_base = vec_cache[t["baseline_subject_en"]]

    cos_translation = cosine(v_en, v_lang)  # [num_layers]
    cos_baseline = cosine(v_en, v_base)  # [num_layers]

    for layer in range(v_en.shape[0]):
        rows.append((
            t["relation"], t["lang"], t["index"], layer,
            float(cos_translation[layer]), float(cos_baseline[layer]),
        ))

import csv
with open(args.out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["relation", "lang", "index", "layer", "cos_translation", "cos_baseline"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {args.out_path}")
