The repository has the code and data behind every experiment in the paper:

1. Code: each folder matches one section of the paper (sec4_… to sec7_ablation, plus the appendix). Every script produces a number, table or figure that the paper reports.
2. Data: the KLAR dataset and cached model outputs are in data/archives/. Unpack them with python lib/unpack_data.py. The large MLP activation maps aren't included, and you regenerate them with cache_mlp_activation_maps.py.
3. Running it: install the requirements with pip install -r requirements.txt, then run the scripts from the repository root. The models (Llama-3.1-8B and others) come from Hugging Face, and the GPU runs used the included SLURM .sbatch files.
