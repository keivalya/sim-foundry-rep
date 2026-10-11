#!/bin/bash
#SBATCH --job-name=pi05-so101
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/pandya.kei/simfoundry-repro/vla/logs/so101_%j.log
set -x
export HF_HOME=/scratch/pandya.kei/hf_cache
export HF_TOKEN=$(cat /scratch/pandya.kei/hf_cache/token)
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
nvidia-smi
/scratch/pandya.kei/conda_envs/openpi/bin/python -I /scratch/pandya.kei/simfoundry-repro/vla/smoke_so101.py
echo "EXIT=$?"
