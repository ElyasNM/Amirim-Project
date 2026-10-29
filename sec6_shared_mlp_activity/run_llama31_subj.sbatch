#!/bin/bash
#SBATCH --mail-user=you@example.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mem=48g
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH -c4
#SBATCH --output=/path/to/lab_dir/transfer_similarity/ts_llama31_subj_%j.txt
#SBATCH --error=/path/to/lab_dir/transfer_similarity/ts_llama31_subj_%j.txt
#SBATCH --job-name=ts_llama31_subj

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
NEWTAG=meta-llama_Llama-3.1-8B_subj
MODEL=meta-llama/Llama-3.1-8B
CACHE=/path/to/lab_dir/cached/${TAG}_section6/examples
OUT=/path/to/lab_dir/transfer_similarity/out/${NEWTAG}
RES=/path/to/lab_dir/transfer_similarity/results/${NEWTAG}

# Phase 1 (GPU): capture per-neuron MLP activation maps at the SUBJECT last token.
# Layers 12-28: overlaps the last_token run's live band (15-28) for a clean
# subject-vs-last_token comparison, extended down to 12 for earlier subject
# enrichment. capture=neurons keeps it apples-to-apples with the last_token run
# and keeps the write-norm-weighted metric available. 12 langs. Maps kept.
python3 -u cache_mlp_activation_maps.py \
    --model_name "$MODEL" --cache_dir "$CACHE" --out_dir "$OUT" \
    --layers 12-28 --positions last_subject --capture neurons

# Phase 2 (CPU): aligned-vs-control similarity + transfer score.
python3 -u analyze_mlp_similarity.py \
    --out_dir "$OUT" --results_dir "$RES" --positions last_subject \
    --topk 100 --n_control_perms 3 --neuron_scale write_norm

# Activation-map heatmaps.
python3 -u plot_activation_maps.py \
    --out_dir "$OUT" --langs en,es,fr,ru,zh,ja --position last_subject \
    --sample 8 --bucket both --topk 40

mail -s "ts_llama31_subj ${SLURM_JOB_ID} done" you@example.com \
    < /path/to/lab_dir/transfer_similarity/ts_llama31_subj_${SLURM_JOB_ID}.txt
