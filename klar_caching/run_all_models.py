import os
import sys
import subprocess

models = [
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-2-7b-hf",
    "allenai/OLMo-2-1124-7B",
    "Qwen/Qwen2.5-7B",
]

for m in models:
    print(f"\n{'='*60}\nRunning: {m}\n{'='*60}")
    subprocess.run([
        sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cacher.py"),
        "--model_name", m,
        "--skip_patchscopes",
    ], check=True)
