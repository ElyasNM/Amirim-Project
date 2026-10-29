"""Runs on the cluster (GPU). Translates translation_jobs/to_translate.jsonl
using NLLB-200, grouped by target language for batching efficiency.
Writes translation_jobs/translated.jsonl (resumable: skips langs already
fully written to the output on restart, at whole-language granularity)."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import NUMERIC_ROOT

MKQA_TO_NLLB = {
    "ar": "arb_Arab", "da": "dan_Latn", "de": "deu_Latn", "en": "eng_Latn",
    "es": "spa_Latn", "fi": "fin_Latn", "fr": "fra_Latn", "he": "heb_Hebr",
    "hu": "hun_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "km": "khm_Khmr", "ms": "zsm_Latn", "nl": "nld_Latn", "no": "nob_Latn",
    "pl": "pol_Latn", "pt": "por_Latn", "ru": "rus_Cyrl", "sv": "swe_Latn",
    "th": "tha_Thai", "tr": "tur_Latn", "vi": "vie_Latn", "zh_cn": "zho_Hans",
    "zh_hk": "zho_Hant", "zh_tw": "zho_Hant",
}


def log(msg):
    print(msg, flush=True)


def load_done_langs(out_path):
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("_lang_complete"):
                    done.add(rec["target_lang"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--input", default=str(NUMERIC_ROOT / "translation_jobs" / "to_translate.jsonl"))
    ap.add_argument("--output", default=str(NUMERIC_ROOT / "translation_jobs" / "translated.jsonl"))
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    by_lang = defaultdict(list)
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_lang[rec["target_lang"]].append(rec)

    done_langs = load_done_langs(out_path)
    log(f"Languages already complete (skipping): {sorted(done_langs)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()

    out_f = open(out_path, "a", encoding="utf-8")

    for lang, items in by_lang.items():
        if lang in done_langs:
            continue
        if lang not in MKQA_TO_NLLB:
            log(f"WARNING: no NLLB code for {lang}, skipping {len(items)} items")
            continue
        nllb_code = MKQA_TO_NLLB[lang]
        tokenizer.src_lang = "eng_Latn"
        log(f"Translating {len(items)} items -> {lang} ({nllb_code})...")

        for i in range(0, len(items), args.batch_size):
            batch = items[i:i + args.batch_size]
            texts = [it["text"] for it in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **enc,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(nllb_code),
                    max_length=256,
                )
            translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for it, translated in zip(batch, translations):
                out_f.write(json.dumps({
                    "source": it["source"],
                    "source_id": it["source_id"],
                    "category": it["category"],
                    "target_lang": lang,
                    "translated_question": translated,
                }, ensure_ascii=False) + "\n")
            out_f.flush()
            if (i // args.batch_size) % 20 == 0:
                log(f"  {lang}: {i + len(batch)}/{len(items)}")

        out_f.write(json.dumps({"target_lang": lang, "_lang_complete": True}) + "\n")
        out_f.flush()
        log(f"Done: {lang}")

    out_f.close()
    log("All languages complete.")


if __name__ == "__main__":
    main()
