#!/bin/bash
#SBATCH --mail-user=you@example.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mem=48g
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH -c4
#SBATCH --output=/path/to/lab_dir/transfer_similarity/ts_llama31_%j.txt
#SBATCH --error=/path/to/lab_dir/transfer_similarity/ts_llama31_%j.txt
#SBATCH --job-name=ts_llama31

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
RES=/path/to/lab_dir/transfer_similarity/results/${TAG}

# Phase 1 (GPU): capture MLP activation maps. Layer band + single position to
# keep the footprint modest (shared lab FS is near full).
python3 -u cache_mlp_activation_maps.py \
    --model_name "$MODEL" --cache_dir "$CACHE" --out_dir "$OUT" \
    --layers 15-31 --positions last_token --capture neurons

# Phase 2 (CPU): aligned-vs-control similarity + transfer score.
python3 -u analyze_mlp_similarity.py \
    --out_dir "$OUT" --results_dir "$RES" \
    --topk 100 --n_control_perms 3 --neuron_scale write_norm

# Activation-map heatmaps.
python3 -u plot_activation_maps.py \
    --out_dir "$OUT" --langs en,es,fr,ru,zh,ja --position last_token \
    --sample 8 --bucket both --topk 40

mail -s "ts_llama31 ${SLURM_JOB_ID} done" you@example.com \
    < /path/to/lab_dir/transfer_similarity/ts_llama31_${SLURM_JOB_ID}.txt
