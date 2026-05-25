#!/bin/bash -l
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -J eval_moe_soft
#SBATCH -o /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe/experiments/expert_moe/logs/eval_moe_soft-%j.out
#SBATCH -e /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe/experiments/expert_moe/logs/eval_moe_soft-%j.err

# Experiment: MoE Soft Routing Evaluation
# Loads independent experts + trained router, evaluates with weighted adapter combination.
# Input:  outputs/expert_moe/models/independent_experts/{lang}_seed{42,123,456}/final_model/
#         outputs/expert_moe/models/moe_router/seed{42,123,456}/best_router/router.pt
# Output: outputs/expert_moe/results/moe_soft/seed{42,123,456}/results.json

source ~/miniconda3/bin/activate
conda activate /proj/uppmax2026-1-123/private/yaxj1/conda_envs/mt26

export HF_HOME=/crex/proj/uppmax2026-1-123/private/yaxj1/hf_cache
export TRANSFORMERS_CACHE=/crex/proj/uppmax2026-1-123/private/yaxj1/hf_cache

cd /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe
mkdir -p experiments/expert_moe/logs

echo "Eval MoE soft routing | Job: $SLURM_JOB_ID | $(date)"
python experiments/expert_moe/e_eval_moe_soft.py \
    --seeds 42 123 456 \
    --use_comet
echo "Done | Exit: $? | $(date)"