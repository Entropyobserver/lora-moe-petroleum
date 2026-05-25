#!/bin/bash -l
#SBATCH -A uppmax2026-1-123
#SBATCH -M pelle
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -J train_seq_forget
#SBATCH -o /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe/experiments/expert_moe/logs/train_seq_forget-%j.out
#SBATCH -e /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe/experiments/expert_moe/logs/train_seq_forget-%j.err
 
# Experiment: Sequential Forgetting Training (LoRA)
# Trains one shared LoRA adapter sequentially on EN→DE→NL→FR.
# Records EN BLEU after each step to track catastrophic forgetting.
# Output: outputs/expert_moe/models/sequential_forgetting/seed{42,123,456}/
 
source ~/miniconda3/bin/activate
conda activate /proj/uppmax2026-1-123/private/yaxj1/conda_envs/mt26
 
export HF_HOME=/crex/proj/uppmax2026-1-123/private/yaxj1/hf_cache
export TRANSFORMERS_CACHE=/crex/proj/uppmax2026-1-123/private/yaxj1/hf_cache
 
cd /crex/proj/uppmax2026-1-123/private/yaxj1/mt_oil_no_moe
mkdir -p experiments/expert_moe/logs
 
echo "Train sequential forgetting | Job: $SLURM_JOB_ID | $(date)"
python experiments/expert_moe/c_train_sequential_forgetting_lora.py --seeds 42 123 456
echo "Done | Exit: $? | $(date)"