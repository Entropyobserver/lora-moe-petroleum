#!/bin/bash -l
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -J train_multitask
#SBATCH -o /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe/experiments/expert_moe/logs/train_multitask-%j.out
#SBATCH -e /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe/experiments/expert_moe/logs/train_multitask-%j.err

# Experiment: Multitask LoRA Training
# Trains one shared LoRA adapter on all 4 language pairs simultaneously
# using randomly mixed batches. Repeated for 3 seeds.
# Input:  data/final_splits_npd/train.json          (EN)
#         data/gpt_filtered/de_no/train.json         (DE)
#         data/gpt_filtered/fr_no/train.json         (FR)
#         data/gpt_filtered/nl_no/train.json         (NL)
# Output: outputs/expert_moe/models/multitask_lora/seed{42,123,456}/final_model/

source ~/miniconda3/bin/activate
conda activate /proj/uppmax2026-1-123/private/yaxj1/conda_envs/mt26

export HF_HOME=/crex/proj/uppmax2026-1-123/private/yaxj1/hf_cache
export TRANSFORMERS_CACHE=/crex/proj/uppmax2026-1-123/private/yaxj1/hf_cache

cd /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe
mkdir -p experiments/expert_moe/logs

echo "Train multitask LoRA | Job: $SLURM_JOB_ID | $(date)"
python experiments/expert_moe/b_train_multitask_lora.py \
    --seeds 42 123 456
echo "Done | Exit: $? | $(date)"