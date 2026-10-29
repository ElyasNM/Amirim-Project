"""
load_cache_and_match.py
-----------------------
Shared constants, data loading, and matching functions for all analysis scripts.
"""

import os
import json
import glob
import unicodedata
from collections import defaultdict

from paths import DATA_ROOT

BASE_DIR = str(DATA_ROOT)
CACHE_DIR = os.path.join(BASE_DIR, "cache", "meta-llama_Llama-3.1-8B_section6", "examples")
MODEL_TAG = os.path.basename(os.path.dirname(CACHE_DIR))  # e.g. Qwen_Qwen2.5-7B_section6
SHORT_NAME = MODEL_TAG.split("_section")[0].split("_", 1)[-1]  # e.g. Qwen2.5-7B
OUT_DIR = os.path.join(BASE_DIR, "results", MODEL_TAG)
os.makedirs(OUT_DIR, exist_ok=True)

LANGS = ["ca", "en", "es", "fr", "hu", "ja", "ko", "nl", "ru", "uk", "vi", "zh"]
NON_EN = [l for l in LANGS if l != "en"]

_meta_path = os.path.join(BASE_DIR, "cache", MODEL_TAG, "metadata.json")
with open(_meta_path, "r", encoding="utf-8") as _f:
    _meta = json.load(_f)
NUM_LAYERS = _meta["num_layers"]

GEO_RELS = {
    "place_of_birth", "place_of_death",
    "headquarters_location", "location_of_formation",
    "capital", "capital_of",
}


# ── Matching ────────────────────────────────────────────────────────────────

def norm(s):
    return unicodedata.normalize("NFC", s.lower().strip())


def _pfx(a, b):
    a, b = norm(a), norm(b)
    return len(a) > 0 and b.startswith(a)


def match_original(pred, target):
    """Original paper matching: bidirectional prefix."""
    return _pfx(pred, target) or _pfx(target, pred)


def match_fixed(pred, target, rec):
    """
    Conservative truncation-aware matching.

    The original `decoded[len(few_shot_prompt):]` extraction can chop the
    beginning off the prediction due to tokenizer encode/decode round-trip
    length changes.  This function catches those truncated predictions:

    1. Bidirectional prefix (same as original).
    2. Suffix match (len > 4) — long suffixes are unlikely coincidental.
    3. Suffix match (len > 2) confirmed by first-token-ID equality.
    4. Very short pred (len <= 2) confirmed by first-token-ID equality.
    """
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


# ── Data loading ────────────────────────────────────────────────────────────

def load_facts():
    """
    Returns dict  (index, relation) -> {lang: record_dict}
    """
    facts = {}
    for path in glob.glob(os.path.join(CACHE_DIR, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        key = (data["index"], data["relation"])
        facts[key] = {}
        for lang in (k for k in data if k not in ("index", "relation") and isinstance(data[k], dict)):
            facts[key][lang] = data[lang]
    return facts
