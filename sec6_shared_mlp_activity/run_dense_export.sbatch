#!/bin/bash
#SBATCH --mem=16g
#SBATCH -c2
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/lab_dir/transfer_similarity/dense_export/log_%j.txt
#SBATCH --error=/path/to/lab_dir/transfer_similarity/dense_export/log_%j.txt
#SBATCH --job-name=dense_export

# CPU only. venv_nlp is built for Python 3.11, which the nodes no longer have (now 3.13),
# so use the system Python (has numpy 2.2), which is all this script needs.
cd /path/to/lab_dir/transfer_similarity
/usr/bin/python3 -u export_dense_maps.py \
  --out_dir /path/to/lab_dir/transfer_similarity/out/meta-llama_Llama-3.1-8B \
  --langs en,es,ru,zh,ja --n_stats 300 --n_cand 20 \
  --extra 6178__capital,6444__capital \
  --dest /path/to/lab_dir/transfer_similarity/dense_export
