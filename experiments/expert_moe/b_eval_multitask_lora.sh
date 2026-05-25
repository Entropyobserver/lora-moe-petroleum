#!/bin/bash -l
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -J eval_multitask_lora
#SBATCH -o /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe/experiments/expert_moe/logs/eval_multitask_lora-%j.out
#SBATCH -e /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe/experiments/expert_moe/logs/eval_multitask_lora-%j.err

# Experiment: Multitask LoRA Evaluation
# Loads the shared LoRA adapter and evaluates on all language pairs.
# on the corresponding test set across 3 seeds.
# Input:  outputs/expert_moe/models/multitask_lora/seed{42,123,456}/final_model/
# Output: outputs/expert_moe/results/multitask_lora/seed{42,123,456}/results.json
#         outputs/expert_moe/results/multitask_lora/seed{42,123,456}/predictions.json

source ~/miniconda3/bin/activate
conda activate /proj/uppmax2026-1-123/private/yaxj1/conda_envs/mt26

export HF_HOME=/crex/proj/uppmax2026-1-123/private/yaxj1/hf_cache
export TRANSFORMERS_CACHE=/crex/proj/uppmax2026-1-123/private/yaxj1/hf_cache

cd /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe
mkdir -p experiments/expert_moe/logs

echo "Eval multitask LoRA | Job: $SLURM_JOB_ID | $(date)"
python experiments/expert_moe/b_eval_multitask_lora.py \
    --seeds 42 123 456 \
    --use_comet
echo "Done | Exit: $? | $(date)"