"""Scan filtered/numeric_qa_raw.jsonl for Mintaka items missing any of
MKQA's 26 languages, and emit one row per (item, missing_lang) to translate.

Only questions are translated - numeric answers are language-invariant for
Mintaka (already true in filtered/numeric_qa_raw.jsonl).

Output: translation_jobs/to_translate.jsonl
  {"source": str, "source_id": str, "category": str, "target_lang": str,
   "text": str}   # text = the English question (canonical source text)
"""
import json
from pathlib import Path

from lang_codes import MKQA_26_LANGS

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import NUMERIC_ROOT
ROOT = NUMERIC_ROOT
IN_PATH = ROOT / "filtered" / "numeric_qa_raw.jsonl"
OUT_DIR = ROOT / "translation_jobs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "to_translate.jsonl"

TARGET_SOURCES = {"mintaka"}


def log(msg):
    print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def main():
    mkqa_langs = set(MKQA_26_LANGS)
    n_jobs = 0
    n_items = 0
    with open(IN_PATH, encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            if rec["source"] not in TARGET_SOURCES:
                continue
            have_langs = set(rec["items"].keys())
            missing = mkqa_langs - have_langs
            if not missing:
                continue
            en_text = rec["items"].get("en", {}).get("question")
            if not en_text:
                # fall back to any available language's question if no English
                en_text = next(iter(rec["items"].values()))["question"]
            n_items += 1
            for lang in sorted(missing):
                fout.write(json.dumps({
                    "source": rec["source"],
                    "source_id": rec["source_id"],
                    "category": rec["category"],
                    "target_lang": lang,
                    "text": en_text,
                }, ensure_ascii=False) + "\n")
                n_jobs += 1

    log(f"{n_items} items need translation, {n_jobs} total (item, lang) translation jobs")
    log(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
