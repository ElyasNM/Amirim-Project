"""Filter each source down to numeric-answer QA only and normalize into one
intermediate schema, per item:

{
  "source": "mkqa" | "mintaka",
  "source_id": str,
  "category": str,          # topical bucket (used later as KLAR relation name)
  "answer_type": str,       # date | number | number_with_unit | numerical
  "items": {                # per-language question/answer, only langs available natively
      "<lang>": {"question": str, "answer": str, "answer_aliases": [str]}
  }
}

Written to filtered/numeric_qa_raw.jsonl (one JSON object per line).
"""
import json
from pathlib import Path

from datasets import load_from_disk

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import NUMERIC_ROOT
ROOT = NUMERIC_ROOT
RAW = ROOT / "raw"
OUT_DIR = ROOT / "filtered"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "numeric_qa_raw.jsonl"

# MKQA ClassLabel index -> name (from mkqa.py loading script)
MKQA_TYPES = [
    "entity", "long_answer", "unanswerable", "date",
    "number", "number_with_unit", "short_phrase", "binary",
]
MKQA_NUMERIC_TYPES = {"date", "number", "number_with_unit"}

MINTAKA_NUMERIC_TYPES = {"numerical", "date"}


def log(msg):
    print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def filter_mkqa(fh):
    ds = load_from_disk(str(RAW / "mkqa"))
    kept = 0
    for row in ds:
        en_answers = row["answers"].get("en", [])
        if not en_answers:
            continue
        type_idx = en_answers[0]["type"]
        type_name = MKQA_TYPES[type_idx]
        if type_name not in MKQA_NUMERIC_TYPES:
            continue
        items = {}
        for lang, ans_list in row["answers"].items():
            if not ans_list:
                continue
            a = ans_list[0]
            q = row["queries"].get(lang)
            if not q or not a["text"]:
                continue
            items[lang] = {
                "question": q,
                "answer": a["text"],
                "answer_aliases": a["aliases"],
            }
        if not items:
            continue
        record = {
            "source": "mkqa",
            "source_id": row["example_id"],
            "category": "open_domain",
            "answer_type": type_name,
            "items": items,
        }
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        kept += 1
    log(f"MKQA: kept {kept} numeric items")


def filter_mintaka(fh):
    kept = 0
    for split in ("train", "dev", "test"):
        path = RAW / "mintaka_json" / f"mintaka_{split}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data:
            answer = row["answer"]
            atype = answer["answerType"]
            if atype not in MINTAKA_NUMERIC_TYPES:
                continue
            ans_entries = answer.get("answer")
            if not ans_entries:
                continue
            # numerical/date answers are plain values, e.g. [{"name": "1976", ...}] or raw string
            first = ans_entries[0] if isinstance(ans_entries, list) else ans_entries
            if isinstance(first, dict):
                answer_text = first.get("name") or first.get("label")
            else:
                answer_text = first
            if answer_text is None:
                continue
            items = {"en": {"question": row["question"], "answer": str(answer_text), "answer_aliases": []}}
            for lang, translated_q in row.get("translations", {}).items():
                if translated_q:
                    items[lang] = {
                        "question": translated_q,
                        "answer": str(answer_text),  # numeric value is language-invariant
                        "answer_aliases": [],
                    }
            record = {
                "source": "mintaka",
                "source_id": row["id"],
                "category": row["category"],
                "answer_type": atype,
                "items": items,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    log(f"Mintaka: kept {kept} numeric items")


if __name__ == "__main__":
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        filter_mkqa(fh)
        filter_mintaka(fh)
    log(f"\nWritten to {OUT_PATH}")
