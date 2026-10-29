"""Download Mintaka's original JSON (train/dev/test) which retains the answerType field
that the HF parquet conversion dropped."""
import json
import urllib.request
from pathlib import Path

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import NUMERIC_ROOT
RAW_DIR = NUMERIC_ROOT / "raw" / "mintaka_json"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "dev", "test"]
BASE = "https://raw.githubusercontent.com/amazon-science/mintaka/main/data/mintaka_{split}.json"


def log(msg):
    print(msg, flush=True)


def main():
    for split in SPLITS:
        url = BASE.format(split=split)
        log(f"Fetching {url}")
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.load(resp)
        out_path = RAW_DIR / f"mintaka_{split}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        log(f"  saved {len(data)} rows -> {out_path}")

    sample = json.loads((RAW_DIR / "mintaka_train.json").read_text(encoding="utf-8"))[0]
    schema_path = RAW_DIR / "sample_schema.json"
    schema_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Sample schema written to {schema_path}")


if __name__ == "__main__":
    main()
