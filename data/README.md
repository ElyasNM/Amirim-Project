# Data

`python lib/unpack_data.py` unpacks everything in `archives/` into `DATA_ROOT` (default: this folder).
`python lib/unpack_data.py --list` shows the archives.

| Archive | Unpacks to | Used by |
|---|---|---|
| `klar.tar.gz` | `klar/<lang>/<relation>.json` (KLAR, 17 languages, 20 relations) | caching, entity binding |
| `cache_<model>_section6.tar.gz` (Llama-3.1-8B, Llama-2-7B, OLMo-2-7B, Qwen2.5-7B) | `cache/<model>_section6/` | §4 and everything downstream |
| `cache_meta-llama_Llama-3.1-8B_baseline_patchscopes.tar.gz` | `cache/…_section6_baseline/`, `cache/…_section6_patchscopes/` | §4 logit-lens baseline, appendix Patchscopes |
| `cache_<model>_section6_subj_act_patch*.tar.gz` (Llama-3.1-8B, Qwen2.5-7B; Qwen in two parts) | `cache/<model>_section6_subj_act_patch/` | §5 |
| `numeric_qa_project.tar.gz` | `numeric_qa_project/{eval_subset,results,translation_jobs}/` | §4.1 |
| `binding_analysis.tar.gz` | `binding_analysis/` (triplets, per-layer cosines, summaries) | §6.1 |
| `transfer_similarity.tar.gz` | `transfer_similarity/{results,relation_controls,relation_znorm,dense_export}/`, `results/meta-llama_Llama-3.1-8B/ablation/` | §6.2, §7 |

**Not included** (too large or re-creatable): the MLP activation maps (`transfer_similarity/out/<tag>/maps`,
tens of GB; regenerate with `sec6_shared_mlp_activity/cache_mlp_activation_maps.py`), the patching caches of
Llama-2 and OLMo-2 (not used in the paper), the raw MKQA / Mintaka downloads and the intermediate numeric-QA
files (rebuild with `sec4_numeric_qa/01_…05_*.py`), and model weights.

**Sources.** KLAR: Wang et al. (2025), CC-BY-4.0, from the authors' repository. Numeric QA items come from MKQA
and Mintaka (see their licenses). The caches, results and tables are outputs of the scripts in this repository.
In the numeric-QA files, the MGSM rows were removed to match the pipeline.
