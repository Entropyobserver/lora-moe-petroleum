#!/bin/bash -l
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -J eval_langid_router
#SBATCH -o /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe/experiments/expert_moe/logs/eval_langid_router-%j.out
#SBATCH -e /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe/experiments/expert_moe/logs/eval_langid_router-%j.err

# Experiment: LangID-Augmented Router Evaluation
# Each seed loads same-seed expert (seed=42 -> expert_seed=42, etc.)
# Input:  outputs/expert_moe/models/independent_experts/{lang}_seed{42,123,456}/final_model/
#         outputs/expert_moe/models/langid_router/seed{42,123,456}/best_router.pt
# Output: outputs/expert_moe/results/langid_router/seed{42,123,456}/results.json

source ~/miniconda3/bin/activate
conda activate /proj/uppmax2026-1-123/private/yaxj1/conda_envs/mt26

export HF_HOME=/crex/proj/uppmax2026-1-123/private/yaxj1/hf_cache
export TRANSFORMERS_CACHE=/crex/proj/uppmax2026-1-123/private/yaxj1/hf_cache

cd /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe
mkdir -p experiments/expert_moe/logs

echo "Eval LangID-augmented router | Job: $SLURM_JOB_ID | $(date)"
python experiments/expert_moe/f_eval_langid_router.py \
    --seeds 42 123 456 \
    --use_comet
echo "Done | Exit: $? | $(date)"