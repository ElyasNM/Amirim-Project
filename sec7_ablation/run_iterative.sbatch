#!/bin/bash
#SBATCH --mail-user=you@example.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mem=48g
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=4:00:00
#SBATCH -c4
#SBATCH --output=/path/to/lab_dir/transfer_similarity/ts_iter_%j.txt
#SBATCH --error=/path/to/lab_dir/transfer_similarity/ts_iter_%j.txt
#SBATCH --job-name=ts_iter

# Greedy iterative neuron localization: grow the mask one small batch at a
# time (globally ranked, re-scored each iteration) until the fact breaks.

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
MAPS=/path/to/lab_dir/transfer_similarity/out/${TAG}/maps
OUT=/path/to/lab_dir/transfer_similarity/results/${TAG}/ablation
N_FACTS=${N_FACTS:-100}
N_CONTROL=${N_CONTROL:-3}
BATCH_SIZE=${BATCH_SIZE:-5}
MAX_ITERS=${MAX_ITERS:-100}
OUT_PREFIX=${OUT_PREFIX:-iter}

python3 -u iterative_ablation.py \
    --model_name "$MODEL" \
    --cache_dir  "$CACHE" \
    --maps_dir   "$MAPS" \
    --out_dir    "$OUT" \
    --n_facts    "$N_FACTS" \
    --n_control  "$N_CONTROL" \
    --batch_size "$BATCH_SIZE" \
    --max_iters  "$MAX_ITERS" \
    --out_prefix "$OUT_PREFIX" \
    --seed 42

mail -s "ts_iter ${SLURM_JOB_ID} done" you@example.com \
    < /path/to/lab_dir/transfer_similarity/ts_iter_${SLURM_JOB_ID}.txt
