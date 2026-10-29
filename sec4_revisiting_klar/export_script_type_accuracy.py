"""
export_script_type_accuracy.py
-------------------------------
Runs script-type accuracy breakdown for all available model caches
and writes results to results/script_type_accuracy/<model>.csv
plus a combined all_models.csv.

Usage:
    python scripts/export_script_type_accuracy.py
"""

import os, sys, json, glob, unicodedata, csv
from collections import defaultdict

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import DATA_ROOT
BASE_DIR = str(DATA_ROOT)
CACHE_ROOT = os.path.join(BASE_DIR, "cache")
OUT_DIR = os.path.join(BASE_DIR, "results", "script_type_accuracy")
os.makedirs(OUT_DIR, exist_ok=True)

LANGS = ["en", "ca", "es", "fr", "hu", "ja", "ko", "nl", "ru", "uk", "vi", "zh"]
STYPES = ["latin", "numeric", "mixed", "native"]

MODELS = [
    "meta-llama_Llama-3.1-8B_section6",
    "meta-llama_Llama-2-7b-hf_section6",
    "Qwen_Qwen2.5-7B_section6",
    "allenai_OLMo-2-1124-7B_section6",
]


def _is_latin(c):
    cp = ord(c)
    return (
        cp <= 0x024F or
        0x0300 <= cp <= 0x036F or
        0x1E00 <= cp <= 0x1EFF or
        0x2C60 <= cp <= 0x2C7F or
        0xA720 <= cp <= 0xA7FF or
        0xAB30 <= cp <= 0xAB6F
    )


def script_type(s):
    s = s.strip()
    if not s:
        return "empty"
    if all(c.isdigit() or c in " .,/-" for c in s):
        return "numeric"
    if all(_is_latin(c) for c in s):
        return "latin"
    has_latin = any(_is_latin(c) and c.isalpha() for c in s)
    has_native = any(not _is_latin(c) and ord(c) >= 128 for c in s)
    if has_latin and has_native:
        return "mixed"
    return "native"


def norm(s):
    return unicodedata.normalize("NFC", s.lower().strip())


def _pfx(a, b):
    a, b = norm(a), norm(b)
    return len(a) > 0 and b.startswith(a)


def match_fixed(pred, target, rec):
    if _pfx(pred, target) or _pfx(target, pred):
        return True
    p, t = norm(pred), norm(target)
    ll = rec.get("logit_lens", {})
    ptid = ll.get("predicted_token_id")
    ctid = ll.get("correct_object_token_id")
    if len(p) > 4 and t.endswith(p) and p != t:
        return True
    if len(p) > 2 and t.endswith(p) and p != t and ptid is not None and ptid == ctid:
        return True
    if 0 < len(p) <= 2 and ptid is not None and ptid == ctid:
        return True
    return False


def compute(cache_dir_name):
    examples_dir = os.path.join(CACHE_ROOT, cache_dir_name, "examples")
    acc = defaultdict(lambda: defaultdict(lambda: {"ok": 0, "n": 0}))
    for path in glob.glob(os.path.join(examples_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for lang in LANGS:
            rec = data.get(lang)
            if not isinstance(rec, dict):
                continue
            obj = rec.get("object", "")
            pred = rec.get("prediction", "")
            stype = script_type(obj)
            ok = match_fixed(pred, " " + obj, rec)
            acc[lang][stype]["n"] += 1
            acc[lang][stype]["ok"] += int(ok)
    return acc


def rows_for_model(model_name, acc):
    rows = []
    for lang in LANGS:
        if lang not in acc:
            continue
        row = {"model": model_name, "lang": lang}
        total_ok = total_n = 0
        for st in STYPES:
            d = acc[lang][st]
            n, ok = d["n"], d["ok"]
            total_ok += ok
            total_n += n
            row[f"{st}_acc"] = round(ok / n * 100, 1) if n else None
            row[f"{st}_n"] = n if n else None
        row["overall_acc"] = round(total_ok / total_n * 100, 1) if total_n else None
        row["overall_n"] = total_n
        rows.append(row)
    return rows


FIELDS = (
    ["model", "lang"]
    + [f for st in STYPES for f in (f"{st}_acc", f"{st}_n")]
    + ["overall_acc", "overall_n"]
)

all_rows = []

for model in MODELS:
    cache_path = os.path.join(CACHE_ROOT, model, "examples")
    if not os.path.isdir(cache_path):
        print(f"  [skip] {model} — examples dir not found")
        continue

    print(f"Processing {model} ...", flush=True)
    acc = compute(model)
    rows = rows_for_model(model, acc)
    all_rows.extend(rows)

    # short model name for filename
    short = model.replace("_section6", "").replace("/", "_")
    out_path = os.path.join(OUT_DIR, f"{short}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {os.path.relpath(out_path, BASE_DIR)}")

# combined
combined_path = os.path.join(OUT_DIR, "all_models.csv")
with open(combined_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(all_rows)

print(f"\nDone. Combined -> {os.path.relpath(combined_path, BASE_DIR)}")
print(f"Output dir: results/script_type_accuracy/  ({len(MODELS)} model files + all_models.csv)")
