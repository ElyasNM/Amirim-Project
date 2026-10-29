"""
Run cacher_patchscopes.py for every model, one by one.
Stops if any model errors. Edit MODELS to change/skip.
"""
import subprocess
import sys
import os

MODELS = [
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen2.5-7B",
    "allenai/OLMo-2-1124-7B",
    "meta-llama/Llama-2-7b-hf",
]

HERE = os.path.dirname(os.path.abspath(__file__))
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from paths import CACHE_ROOT
ROOT = str(CACHE_ROOT)

for m in MODELS:
    tag = m.replace("/", "_")
    cache = os.path.join(ROOT, f"{tag}_section6", "examples")
    print("=" * 50, flush=True)
    print(f"Patchscopes: {m}", flush=True)
    print("=" * 50, flush=True)
    cmd = [
        sys.executable, os.path.join(HERE, "cacher_patchscopes.py"),
        "--model_name", m,
        "--existing_cache", cache,
    ]
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0:
        print(f"FAILED on {m} (exit {res.returncode})", flush=True)
        sys.exit(res.returncode)

print("All models done.", flush=True)
