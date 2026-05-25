#!/bin/bash -l
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -J train_expert_de
#SBATCH -o /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe/experiments/expert_moe/logs/train_expert_de-%j.out
#SBATCH -e /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe/experiments/expert_moe/logs/train_expert_de-%j.err

# Experiment: Independent Expert Training, DE-NO
# Input:  data/gpt_filtered/de_no/train.json
# Output: outputs/expert_moe/models/independent_experts/de_seed{42,123,456}/final_model/

source ~/miniconda3/bin/activate
conda activate /proj/uppmax2026-1-123/private/yaxj1/conda_envs/mt26

cd /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe
mkdir -p experiments/expert_moe/logs

echo "Train independent expert DE | Job: $SLURM_JOB_ID | $(date)"
python experiments/expert_moe/a_train_independent_experts.py \
    --lang de \
    --seeds 42 123 456
echo "Done | Exit: $? | $(date)"