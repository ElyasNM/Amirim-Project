"""Mapping from MKQA's 26 language codes (also used as our numeric_qa/<lang>/
folder names) to NLLB-200 / FLORES-200 codes."""

MKQA_TO_NLLB = {
    "ar": "arb_Arab",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "he": "heb_Hebr",
    "hu": "hun_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "km": "khm_Khmr",
    "ms": "zsm_Latn",
    "nl": "nld_Latn",
    "no": "nob_Latn",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "sv": "swe_Latn",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "zh_cn": "zho_Hans",
    "zh_hk": "zho_Hant",
    "zh_tw": "zho_Hant",
}

MKQA_26_LANGS = list(MKQA_TO_NLLB.keys())
