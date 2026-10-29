"""Runs on the cluster (GPU). Evaluates a base LM on eval_subset/ with 3-shot
prompting (matching KLAR's n_shot=3 default), Q:/A: format, resumable per model.

Prompt: 3 demos of "Q: <question>\nA: <answer>\n\n" sampled from the same
(language, category) excluding the eval item itself, then "Q: <question>\nA:".
"""
import argparse
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import NUMERIC_ROOT

N_SHOT = 3
MAX_NEW_TOKENS = 15
SEED = 42


def log(msg):
    print(msg, flush=True)


def load_eval_subset(root):
    # lang -> category -> list of samples
    data = {}
    for lang_dir in sorted(root.iterdir()):
        if not lang_dir.is_dir():
            continue
        data[lang_dir.name] = {}
        for f in lang_dir.glob("*.json"):
            data[lang_dir.name][f.stem] = json.loads(f.read_text(encoding="utf-8"))["samples"]
    return data


def build_prompt(rng, samples, target_idx):
    others = [s for i, s in enumerate(samples) if i != target_idx]
    demos = rng.sample(others, min(N_SHOT, len(others)))
    parts = [f"Q: {d['question']}\nA: {d['answer']}\n\n" for d in demos]
    parts.append(f"Q: {samples[target_idx]['question']}\nA:")
    return "".join(parts)


NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def extract_number(text):
    m = NUM_RE.search(text.replace(",", "") if text.count(",") > 1 else text)
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def score(prediction, answer, aliases):
    pred_norm = prediction.strip().lower()
    candidates = [answer] + (aliases or [])
    for cand in candidates:
        if cand.strip().lower() in pred_norm:
            return True
    pred_num = extract_number(prediction)
    gold_num = extract_number(answer)
    if pred_num is not None and gold_num is not None and abs(pred_num - gold_num) < 1e-6:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-tag", required=True, help="short name for output files")
    ap.add_argument("--eval-subset", default=str(NUMERIC_ROOT / "eval_subset"))
    ap.add_argument("--output", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    out_path = Path(args.output or NUMERIC_ROOT / "results" / f"{args.model_tag}_results.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done_keys.add((r["lang"], r["category"], r["source_id"]))
        log(f"Resuming: {len(done_keys)} already done")

    root = Path(args.eval_subset)
    data = load_eval_subset(root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    out_f = open(out_path, "a", encoding="utf-8")
    rng = random.Random(SEED)

    n_correct = 0
    n_total = 0
    for lang, categories in data.items():
        for category, samples in categories.items():
            pending = [
                (i, s) for i, s in enumerate(samples)
                if (lang, category, s["source_id"]) not in done_keys
            ]
            if not pending:
                continue
            log(f"{args.model_tag} | {lang} | {category}: {len(pending)} items")
            for batch_start in range(0, len(pending), args.batch_size):
                batch = pending[batch_start:batch_start + args.batch_size]
                prompts = [build_prompt(rng, samples, i) for i, _ in batch]
                enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
                with torch.no_grad():
                    generated = model.generate(
                        **enc, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False, pad_token_id=tokenizer.pad_token_id,
                    )
                gen_only = generated[:, enc["input_ids"].shape[1]:]
                decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

                for (i, s), pred in zip(batch, decoded):
                    pred_answer = pred.split("\n")[0].strip()
                    correct = score(pred_answer, s["answer"], s.get("answer_aliases", []))
                    n_total += 1
                    n_correct += int(correct)
                    out_f.write(json.dumps({
                        "lang": lang, "category": category, "source_id": s["source_id"],
                        "source": s["source"], "question": s["question"], "gold_answer": s["answer"],
                        "prediction_raw": pred, "prediction_answer": pred_answer, "correct": correct,
                    }, ensure_ascii=False) + "\n")
                out_f.flush()

    out_f.close()
    log(f"\n{args.model_tag}: {n_correct}/{n_total} = {n_correct/max(n_total,1):.3f} accuracy (this run's new items)")


if __name__ == "__main__":
    main()
