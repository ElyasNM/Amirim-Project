"""Convert filtered/numeric_qa_raw.jsonl into KLAR-style per-language files:

numeric_qa/<lang>/<category>.json
{
  "name": "<category>",
  "samples": [
    {"question": str, "answer": str, "answer_aliases": [str],
     "source": "mkqa"|"mintaka", "source_id": str, "answer_type": str, "index": int}
  ]
}

This mirrors klar/klar/<lang>/<relation>.json (per-language folders, one JSON per
topical bucket) but swaps the relation's subject/object/prompt_templates for
question/answer, since these are free-form QA pairs rather than subject-relation
fill-in-the-blank triples.
"""
import json
from collections import defaultdict
from pathlib import Path

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import NUMERIC_ROOT
ROOT = NUMERIC_ROOT
IN_PATH = ROOT / "filtered" / "numeric_qa_raw.jsonl"
OUT_ROOT = ROOT / "numeric_qa"


def log(msg):
    print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def main():
    # buckets[lang][category] = list of sample dicts
    buckets = defaultdict(lambda: defaultdict(list))
    running_index = defaultdict(lambda: defaultdict(int))

    with open(IN_PATH, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            category = rec["category"]
            for lang, qa in rec["items"].items():
                idx = running_index[lang][category]
                buckets[lang][category].append({
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "answer_aliases": qa["answer_aliases"],
                    "source": rec["source"],
                    "source_id": rec["source_id"],
                    "answer_type": rec["answer_type"],
                    "index": idx,
                })
                running_index[lang][category] += 1

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    n_files = 0
    n_langs = len(buckets)
    for lang, categories in buckets.items():
        lang_dir = OUT_ROOT / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        for category, samples in categories.items():
            out_path = lang_dir / f"{category}.json"
            payload = {"name": category, "samples": samples}
            out_path.write_text(
                json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8"
            )
            n_files += 1

    log(f"Wrote {n_files} files across {n_langs} languages under {OUT_ROOT}")
    for lang in sorted(buckets):
        total = sum(len(v) for v in buckets[lang].values())
        log(f"  {lang}: {total} samples across {len(buckets[lang])} categories")


if __name__ == "__main__":
    main()
