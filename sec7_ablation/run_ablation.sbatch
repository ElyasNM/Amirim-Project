#!/bin/bash
#SBATCH --mail-user=you@example.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mem=48g
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=4:00:00
#SBATCH -c4
#SBATCH --output=/path/to/lab_dir/transfer_similarity/ts_ablation_%j.txt
#SBATCH --error=/path/to/lab_dir/transfer_similarity/ts_ablation_%j.txt
#SBATCH --job-name=ts_ablation

# Neuron ablation (top-40/layer, z-scored, English-only, Llama-3.1-8B).
# Step 1 (localize_topk_mask.py) is CPU-only, reads the existing last_token
# activation maps (layers 15-31). Step 2 (run_ablation.py) needs the GPU.

source /usr/local/lmod/8.7.56/init/bash
module load cuda
module load nvidia

export HF_HOME="/path/to/lab_dir/huggingface_cache"
export HF_DATASETS_CACHE="/path/to/lab_dir/huggingface_cache/datasets"
export HUGGINGFACE_HUB_CACHE="/path/to/lab_dir/huggingface_cache/hub"
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
export UV_HTTP_TIMEOUT=300

source /path/to/lab_dir/venv_nlp/bin/activate

cd /path/to/lab_dir/transfer_similarity

TAG=meta-llama_Llama-3.1-8B
MODEL=meta-llama/Llama-3.1-8B
CACHE=/path/to/lab_dir/cached/${TAG}_section6/examples
OUT=/path/to/lab_dir/transfer_similarity/out/${TAG}
RES=/path/to/lab_dir/transfer_similarity/results/${TAG}/ablation
N_FACTS=${N_FACTS:-20}
N_CONTROL=${N_CONTROL:-3}

python3 -u localize_topk_mask.py \
    --maps_dir  "${OUT}/maps" \
    --topk      40 \
    --out_path  "${RES}/topk_masks_en.npz"

python3 -u run_ablation.py \
    --model_name "$MODEL" \
    --cache_dir  "$CACHE" \
    --masks_path "${RES}/topk_masks_en.npz" \
    --out_csv    "${RES}/ablation_results.csv" \
    --n_facts    "$N_FACTS" \
    --n_control  "$N_CONTROL" \
    --seed       42

mail -s "ts_ablation ${SLURM_JOB_ID} done" you@example.com \
    < /path/to/lab_dir/transfer_similarity/ts_ablation_${SLURM_JOB_ID}.txt
