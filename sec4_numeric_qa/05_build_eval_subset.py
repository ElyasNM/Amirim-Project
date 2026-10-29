"""Build a filtered eval subset for the first model-eval run, sized for two
7B models on a single L40S in a few hours rather than a full day.

Sampling plan (fixed seed for reproducibility):
  - open_domain (MKQA): 150 random items
  - each of Mintaka's 8 topic categories: 60 random items restricted to
    complexityType == "count" (simple direct factual lookups; harder types
    like multihop/comparative/superlative are saved for a later, harder run
    since they're likely near-floor for 7B models)

Selection happens once on (source, source_id) pairs, then applied identically
to every language so the subset stays fully parallel across all 26 languages.

Output: eval_subset/<lang>/<category>.json (same schema as numeric_qa/).
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import NUMERIC_ROOT
ROOT = NUMERIC_ROOT
NUMERIC_QA = ROOT / "numeric_qa"
RAW_MINTAKA = ROOT / "raw" / "mintaka_json"
OUT_ROOT = ROOT / "eval_subset"

SEED = 42
MKQA_26 = [
    "ar", "da", "de", "en", "es", "fi", "fr", "he", "hu", "it", "ja", "ko",
    "km", "ms", "nl", "no", "pl", "pt", "ru", "sv", "th", "tr", "vi",
    "zh_cn", "zh_hk", "zh_tw",
]
N_OPEN_DOMAIN = 150
N_MINTAKA_PER_CATEGORY = 60
MINTAKA_CATEGORIES = [
    "books", "geography", "music", "sports", "movies", "politics", "videogames", "history",
]


def log(msg):
    print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def load_mintaka_complexity():
    complexity = {}
    for split in ("train", "dev", "test"):
        data = json.loads((RAW_MINTAKA / f"mintaka_{split}.json").read_text(encoding="utf-8"))
        for row in data:
            complexity[row["id"]] = row["complexityType"]
    return complexity


def main():
    rng = random.Random(SEED)
    complexity = load_mintaka_complexity()

    # Use the "en" files as the canonical index of available (source, source_id) per category
    en_dir = NUMERIC_QA / "en"
    selected = {}  # category -> set of source_id

    # open_domain: random N
    od_data = json.loads((en_dir / "open_domain.json").read_text(encoding="utf-8"))
    od_ids = [s["source_id"] for s in od_data["samples"]]
    rng.shuffle(od_ids)
    selected["open_domain"] = set(od_ids[:N_OPEN_DOMAIN])
    log(f"open_domain: {len(selected['open_domain'])} items (of {len(od_ids)})")

    # mintaka categories: count-type only, random N
    for cat in MINTAKA_CATEGORIES:
        cat_data = json.loads((en_dir / f"{cat}.json").read_text(encoding="utf-8"))
        count_ids = [s["source_id"] for s in cat_data["samples"] if complexity.get(s["source_id"]) == "count"]
        rng.shuffle(count_ids)
        selected[cat] = set(count_ids[:N_MINTAKA_PER_CATEGORY])
        log(f"{cat}: {len(selected[cat])} items (of {len(count_ids)} count-type, {len(cat_data['samples'])} total)")

    total_items = sum(len(v) for v in selected.values())
    log(f"\nTotal unique items selected: {total_items}")

    # Apply the same selection to every language
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    n_files = 0
    n_samples = 0
    for lang in MKQA_26:
        lang_dir = NUMERIC_QA / lang
        out_lang_dir = OUT_ROOT / lang
        out_lang_dir.mkdir(parents=True, exist_ok=True)
        for category, ids in selected.items():
            data = json.loads((lang_dir / f"{category}.json").read_text(encoding="utf-8"))
            filtered_samples = [s for s in data["samples"] if s["source_id"] in ids]
            out_path = out_lang_dir / f"{category}.json"
            out_path.write_text(
                json.dumps({"name": category, "samples": filtered_samples}, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )
            n_files += 1
            n_samples += len(filtered_samples)

    log(f"\nWrote {n_files} files, {n_samples} total samples across {len(MKQA_26)} languages under {OUT_ROOT}")
    log(f"Expected: {total_items} items x {len(MKQA_26)} langs = {total_items * len(MKQA_26)}")


if __name__ == "__main__":
    main()
