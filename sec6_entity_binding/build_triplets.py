"""
Build translation-binding triplets from the KLAR dataset.

Triplet = (subject_en, subject_lang, baseline_subject_en)
  - subject_en / subject_lang: same real-world entity, English name / target-language name
    (a translation pair -- aligned via the shared `index` field across KLAR language files)
  - baseline_subject_en: a DIFFERENT entity of the SAME relation type, in English
    (same-language, different-entity control)

Output: binding_triplets.json, a flat list of:
  {
    "relation": str, "lang": str,
    "index": int, "subject_en": str, "subject_lang": str,
    "baseline_index": int, "baseline_subject_en": str
  }
"""
import json
import os
import random

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import KLAR_DIR, BINDING_DIR
BASE = str(KLAR_DIR)
os.makedirs(BINDING_DIR, exist_ok=True)
LANGS = ["he", "ar", "ru", "ja", "zh", "ko", "es", "fr"]
MAX_PER_RELATION_LANG = 150
SEED = 0


def load_relation(lang, relation_file):
    path = os.path.join(BASE, lang, relation_file)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    rng = random.Random(SEED)
    relation_files = sorted(os.listdir(os.path.join(BASE, "en")))

    triplets = []
    stats = {}

    for relation_file in relation_files:
        relation = relation_file.replace(".json", "")
        en_data = load_relation("en", relation_file)
        en_by_index = {s["index"]: s["subject"] for s in en_data["samples"]}
        all_indices = list(en_by_index.keys())

        for lang in LANGS:
            path = os.path.join(BASE, lang, relation_file)
            if not os.path.exists(path):
                continue
            lang_data = load_relation(lang, relation_file)
            samples = lang_data["samples"]

            # sanity: subject_en in the lang file should match en file at same index
            usable = [s for s in samples if "subject_en" in s and s["index"] in en_by_index]
            rng.shuffle(usable)
            usable = usable[:MAX_PER_RELATION_LANG]

            for s in usable:
                idx = s["index"]
                subj_en = s["subject_en"]
                subj_lang = s["subject"]

                # pick a baseline: different entity, same relation, English
                candidates = [i for i in all_indices if i != idx and en_by_index[i].strip().lower() != subj_en.strip().lower()]
                if not candidates:
                    continue
                baseline_idx = rng.choice(candidates)
                baseline_subj_en = en_by_index[baseline_idx]

                triplets.append({
                    "relation": relation,
                    "lang": lang,
                    "index": idx,
                    "subject_en": subj_en,
                    "subject_lang": subj_lang,
                    "baseline_index": baseline_idx,
                    "baseline_subject_en": baseline_subj_en,
                })

            stats[(relation, lang)] = len(usable)

    out_path = os.path.join(str(BINDING_DIR), "binding_triplets.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)

    print(f"Total triplets: {len(triplets)}")
    print(f"Relations: {len(relation_files)}, Languages: {LANGS}")
    print(f"Wrote {out_path}")

    by_lang = {}
    for t in triplets:
        by_lang[t["lang"]] = by_lang.get(t["lang"], 0) + 1
    for lang, n in sorted(by_lang.items()):
        print(f"  {lang}: {n}")


if __name__ == "__main__":
    main()
