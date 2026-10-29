"""
accuracy_by_script_type.py
---------------------------
For each language, breaks down model accuracy by gold-answer script type:
  ascii_latin / numeric / mixed / native

Usage:
    python scripts/accuracy_by_script_type.py <cache_dir_name>
    e.g.:  python scripts/accuracy_by_script_type.py meta-llama_Llama-3.1-8B_section6
"""

import os, sys, json, glob, unicodedata
from collections import defaultdict

if len(sys.argv) < 2:
    print("Usage: python accuracy_by_script_type.py <cache_dir_name>")
    sys.exit(1)

CACHE_DIR_NAME = sys.argv[1]
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import DATA_ROOT
BASE_DIR = str(DATA_ROOT)
CACHE_DIR = os.path.join(BASE_DIR, "cache", CACHE_DIR_NAME, "examples")

LANGS = ["en", "ca", "es", "fr", "hu", "ja", "ko", "nl", "ru", "uk", "vi", "zh"]

def _is_latin(c):
    cp = ord(c)
    return (
        cp <= 0x024F or             # Basic Latin + Latin-1 Supplement + Extended A+B
        0x0300 <= cp <= 0x036F or   # Combining Diacritical Marks
        0x1E00 <= cp <= 0x1EFF or   # Latin Extended Additional (Vietnamese)
        0x2C60 <= cp <= 0x2C7F or   # Latin Extended-C
        0xA720 <= cp <= 0xA7FF or   # Latin Extended-D
        0xAB30 <= cp <= 0xAB6F      # Latin Extended-E
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

# acc[lang][script_type] = {"ok": int, "n": int}
acc = defaultdict(lambda: defaultdict(lambda: {"ok": 0, "n": 0}))

for path in glob.glob(os.path.join(CACHE_DIR, "*.json")):
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

STYPES = ["latin", "numeric", "mixed", "native"]

print(f"\n{'Lang':<6}", end="")
for st in STYPES:
    print(f"  {st:>12}", end="")
print(f"  {'overall':>8}")
print("-" * 75)

for lang in LANGS:
    if lang not in acc:
        continue
    print(f"{lang:<6}", end="")
    total_ok = total_n = 0
    for st in STYPES:
        d = acc[lang][st]
        n, ok = d["n"], d["ok"]
        total_ok += ok
        total_n += n
        if n == 0:
            print(f"  {'N/A':>12}", end="")
        else:
            print(f"  {ok/n*100:>7.1f}% ({n:>4})", end="")
    overall = total_ok / total_n * 100 if total_n else 0
    print(f"  {overall:>7.1f}%")

print("\nColumns: accuracy% (n facts) for each gold-answer script type")
print("latin   = answer uses any Latin script (Basic, Extended A-E, incl. diacritics)")
print("numeric = year / number gold answer")
print("native  = non-Latin native script gold answer")
