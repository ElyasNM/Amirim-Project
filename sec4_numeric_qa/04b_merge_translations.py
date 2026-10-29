"""Merge translation_jobs/translated.jsonl back into numeric_qa/<lang>/<category>.json,
adding a "translated": true / "translation_model" marker on injected items."""
import json
from collections import defaultdict
from pathlib import Path

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import NUMERIC_ROOT
ROOT = NUMERIC_ROOT
FILTERED_PATH = ROOT / "filtered" / "numeric_qa_raw.jsonl"
TRANSLATED_PATH = ROOT / "translation_jobs" / "translated.jsonl"
OUT_ROOT = ROOT / "numeric_qa"
MODEL_NAME = "facebook/nllb-200-distilled-600M"


def log(msg):
    print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def main():
    # (source, source_id) -> {answer, answer_aliases, answer_type}
    answer_lookup = {}
    with open(FILTERED_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            any_item = next(iter(rec["items"].values()))
            answer_lookup[(rec["source"], rec["source_id"])] = {
                "answer": any_item["answer"],
                "answer_aliases": any_item["answer_aliases"],
                "answer_type": rec["answer_type"],
            }

    # existing max index per (lang, category), so appended items don't collide
    next_index = defaultdict(int)
    existing_data = {}
    for lang_dir in OUT_ROOT.iterdir():
        if not lang_dir.is_dir():
            continue
        for f in lang_dir.glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            key = (lang_dir.name, f.stem)
            existing_data[key] = data
            if data["samples"]:
                next_index[key] = max(s["index"] for s in data["samples"]) + 1

    n_added = 0
    n_skipped_no_answer = 0
    with open(TRANSLATED_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("_lang_complete"):
                continue
            lookup_key = (rec["source"], rec["source_id"])
            ans = answer_lookup.get(lookup_key)
            if ans is None:
                n_skipped_no_answer += 1
                continue

            key = (rec["target_lang"], rec["category"])
            if key not in existing_data:
                existing_data[key] = {"name": rec["category"], "samples": []}

            idx = next_index[key]
            next_index[key] += 1
            existing_data[key]["samples"].append({
                "question": rec["translated_question"],
                "answer": ans["answer"],
                "answer_aliases": ans["answer_aliases"],
                "source": rec["source"],
                "source_id": rec["source_id"],
                "answer_type": ans["answer_type"],
                "index": idx,
                "translated": True,
                "translation_model": MODEL_NAME,
            })
            n_added += 1

    for (lang, category), data in existing_data.items():
        lang_dir = OUT_ROOT / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        out_path = lang_dir / f"{category}.json"
        out_path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")

    log(f"Added {n_added} translated samples across {len(existing_data)} (lang, category) files")
    if n_skipped_no_answer:
        log(f"Skipped {n_skipped_no_answer} translated rows with no matching answer (unexpected)")


if __name__ == "__main__":
    main()
