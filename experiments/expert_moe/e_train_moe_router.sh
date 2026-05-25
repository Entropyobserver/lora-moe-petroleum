#!/bin/bash -l
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -J train_moe_router
#SBATCH -o /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe/experiments/expert_moe/logs/train_moe_router-%j.out
#SBATCH -e /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe/experiments/expert_moe/logs/train_moe_router-%j.err

# Experiment: MoE Router Training
# Loads independent experts and trains the gated router on mixed multilingual data.
# Input:  outputs/expert_moe/models/independent_experts/{lang}_seed{42,123,456}/final_model/
# Output: outputs/expert_moe/models/moe_router/seed{42,123,456}/best_router/router.pt

source ~/miniconda3/bin/activate
conda activate /proj/uppmax2026-1-123/private/yaxj1/conda_envs/mt26

export HF_HOME=/crex/proj/uppmax2026-1-123/private/yaxj1/hf_cache
export TRANSFORMERS_CACHE=/crex/proj/uppmax2026-1-123/private/yaxj1/hf_cache

cd /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe
mkdir -p experiments/expert_moe/logs

echo "Train MoE router | Job: $SLURM_JOB_ID | $(date)"
python experiments/expert_moe/e_train_moe_router.py --seeds 42 123 456
echo "Done | Exit: $? | $(date)"