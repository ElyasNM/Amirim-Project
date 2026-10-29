"""
common.py
---------
Shared helpers for the cross-lingual MLP-activation-map similarity experiment
(Def-2 "transfer via shared paths", similarity stage only -- no ablation).

Used by:
    cache_mlp_activation_maps.py   (GPU phase 1: capture the maps)
    analyze_mlp_similarity.py      (CPU phase 2: aligned vs control similarity)
"""

import os
import re
import json
import glob
import unicodedata


# ── Text matching (mirrors analysis/load_cache_and_match.py) ─────────────────
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
    Conservative truncation-aware matching, identical to the one used across
    analysis/*.py so knowledge-state buckets here line up with earlier work.
    `rec` is the per-language cache record (supplies logit_lens token ids).
    """
    if _pfx(pred, target) or _pfx(target, pred):
        return True
    p, t = norm(pred), norm(target)
    ll = rec.get("logit_lens", {}) if isinstance(rec, dict) else {}
    ptid = ll.get("predicted_token_id")
    ctid = ll.get("correct_object_token_id")
    if len(p) > 4 and t.endswith(p) and p != t:
        return True
    if len(p) > 2 and t.endswith(p) and p != t and ptid is not None and ptid == ctid:
        return True
    if 0 < len(p) <= 2 and ptid is not None and ptid == ctid:
        return True
    return False


def is_known(rec):
    """Did the model answer this (fact, lang) record correctly?"""
    if not isinstance(rec, dict) or "prediction" not in rec:
        return False
    return match_fixed(rec.get("prediction", ""), " " + rec.get("object", ""), rec)


# ── Cache loading ───────────────────────────────────────────────────────────
def load_cache(cache_examples_dir):
    """
    Returns  dict  (index:int, relation:str) -> {lang: record_dict}
    from a cacher.py output directory (…/examples/*.json).
    """
    facts = {}
    for path in sorted(glob.glob(os.path.join(cache_examples_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        key = (data["index"], data["relation"])
        facts[key] = {
            lang: data[lang]
            for lang in data
            if lang not in ("index", "relation") and isinstance(data[lang], dict)
        }
    return facts


def fact_stem(index, relation):
    return f"{index}__{relation}"


# ── Layer / module resolution (Llama / Qwen / OLMo / GPT-NeoX / GPT-2) ──────
def _resolve(model, dotted):
    obj = model
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def get_decoder_layers(model):
    for attr in ("model.layers", "transformer.h", "gpt_neox.layers",
                 "transformer.layers", "model.decoder.layers"):
        try:
            return _resolve(model, attr)
        except AttributeError:
            continue
    raise ValueError("Cannot locate decoder layer list on this model.")


# Candidate attribute names for the MLP down-projection (output) linear layer.
# Its INPUT is the post-gating per-neuron activation vector of width
# intermediate_size -- that is our "activation map".
_DOWN_PROJ_NAMES = ("down_proj", "c_proj", "wo", "fc2", "dense_4h_to_h", "w2")


def get_mlp_down_proj(layer):
    mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
    if mlp is None:
        raise ValueError("Decoder layer has no .mlp / .feed_forward submodule.")
    for name in _DOWN_PROJ_NAMES:
        if hasattr(mlp, name):
            return getattr(mlp, name), name
    raise ValueError(f"Cannot find MLP down-projection; tried {_DOWN_PROJ_NAMES}.")


# ── Subject token localisation (mirrors klar cacher) ────────────────────────
def find_subject_last_token(input_ids_list, subject, tokenizer):
    """Index of the last token of the subject's last occurrence, else None."""
    def _search(text):
        for prefix in (" ", ""):
            sub_ids = tokenizer.encode(prefix + text, add_special_tokens=False)
            if not sub_ids:
                continue
            for start in range(len(input_ids_list) - len(sub_ids), -1, -1):
                if input_ids_list[start:start + len(sub_ids)] == sub_ids:
                    return start + len(sub_ids) - 1
        return None

    pos = _search(subject)
    if pos is not None:
        return pos
    words = subject.split()
    if words:
        return _search(words[-1])
    return None


# ── Layer spec parsing ─────────────────────────────────────────────────────
def parse_layers(spec, num_layers):
    """'all' | '20-31' | '0,4,8' | '10-' | '-5'  ->  sorted list of ints."""
    if spec is None or spec.strip().lower() == "all":
        return list(range(num_layers))
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d*)-(\d*)", part)
        if m:
            lo = int(m.group(1)) if m.group(1) else 0
            hi = int(m.group(2)) if m.group(2) else num_layers - 1
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(i for i in out if 0 <= i < num_layers)
