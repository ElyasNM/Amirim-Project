#!/bin/bash
#SBATCH --mem=40g
#SBATCH -c4
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/lab_dir/transfer_similarity/relation_controls/log_%j.txt
#SBATCH --error=/path/to/lab_dir/transfer_similarity/relation_controls/log_%j.txt
#SBATCH --job-name=relctrl
cd /path/to/lab_dir/transfer_similarity
/usr/bin/python3 -u relation_controls.py --out_dir /path/to/lab_dir/transfer_similarity/out/meta-llama_Llama-3.1-8B --dest /path/to/lab_dir/transfer_similarity/relation_controls
