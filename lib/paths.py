"""Where the scripts read and write data.

Everything that is not code lives under DATA_ROOT (default: <repo>/data, override with the
DATA_ROOT environment variable). See the README for what goes where.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", REPO_ROOT / "data"))

KLAR_DIR = DATA_ROOT / "klar"                      # KLAR dataset: <lang>/<relation>.json
CACHE_ROOT = DATA_ROOT / "cache"                   # <model>_section6[_baseline|_subj_act_patch|_patchscopes]/
RESULTS_ROOT = DATA_ROOT / "results"               # figures and tables written by the analysis scripts
NUMERIC_ROOT = DATA_ROOT / "numeric_qa_project"    # raw/, filtered/, numeric_qa/, eval_subset/, results/
BINDING_DIR = DATA_ROOT / "binding_analysis"       # triplets, per-layer cosines, summaries
MLP_ROOT = DATA_ROOT / "transfer_similarity"       # out/<tag>/maps (activation maps), results/<tag>/
