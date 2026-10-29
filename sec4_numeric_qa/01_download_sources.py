"""Download MKQA and Mintaka into raw/ and print schema samples for inspection."""
import json
import sys
from pathlib import Path

from datasets import load_dataset

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import NUMERIC_ROOT
RAW_DIR = NUMERIC_ROOT / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def dump_sample(name, example):
    text = json.dumps(example, indent=2, ensure_ascii=False, default=str)
    out_path = RAW_DIR / f"{name.lower()}_sample.json"
    out_path.write_text(text, encoding="utf-8")
    log(f"\n=== {name} sample written to {out_path} ===")


def download_mkqa():
    log("Downloading MKQA (apple/mkqa, parquet convert branch)...")
    ds = load_dataset(
        "parquet",
        data_files=(
            "https://huggingface.co/datasets/apple/mkqa/resolve/"
            "refs%2Fconvert%2Fparquet/mkqa/train/0000.parquet"
        ),
        split="train",
    )
    ds.save_to_disk(str(RAW_DIR / "mkqa"))
    dump_sample("MKQA", ds[0])
    log(f"MKQA rows: {len(ds)}")


def download_mintaka():
    log("Downloading Mintaka (AmazonScience/mintaka, parquet convert branch)...")
    ds = load_dataset(
        "parquet",
        data_files=(
            "https://huggingface.co/datasets/AmazonScience/mintaka/resolve/"
            "refs%2Fconvert%2Fparquet/all/train/0000.parquet"
        ),
        split="train",
    )
    ds.save_to_disk(str(RAW_DIR / "mintaka"))
    dump_sample("Mintaka", ds[0])
    log(f"Mintaka rows: {len(ds)}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "mkqa"):
        download_mkqa()
    if which in ("all", "mintaka"):
        download_mintaka()
    log("\nDone.")
