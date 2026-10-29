#!/bin/bash
#SBATCH --mail-user=you@example.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mem=48g
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=4:00:00
#SBATCH -c4
#SBATCH --output=/path/to/lab_dir/transfer_similarity/ts_sweep_%j.txt
#SBATCH --error=/path/to/lab_dir/transfer_similarity/ts_sweep_%j.txt
#SBATCH --job-name=ts_sweep

# K-sweep: top-K/layer z-scored neuron ablation, doubling K until control
# survival declines significantly, then bisecting to localize the knee.

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
N_FACTS=${N_FACTS:-200}
N_CONTROL=${N_CONTROL:-3}
K_START=${K_START:-40}
CONTROL_FLOOR=${CONTROL_FLOOR:-0.90}
RANK_METRIC=${RANK_METRIC:-zscore}
OUT_PREFIX=${OUT_PREFIX:-sweep}

python3 -u sweep_ablation.py \
    --model_name "$MODEL" \
    --cache_dir  "$CACHE" \
    --maps_dir   "$MAPS" \
    --out_dir    "$OUT" \
    --n_facts    "$N_FACTS" \
    --n_control  "$N_CONTROL" \
    --k_start    "$K_START" \
    --control_floor "$CONTROL_FLOOR" \
    --rank_metric "$RANK_METRIC" \
    --out_prefix "$OUT_PREFIX" \
    --seed 42

mail -s "ts_sweep ${SLURM_JOB_ID} done" you@example.com \
    < /path/to/lab_dir/transfer_similarity/ts_sweep_${SLURM_JOB_ID}.txt
