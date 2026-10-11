#!/bin/bash
#SBATCH --job-name=simfoundry-faiss-check
#SBATCH --output=/scratch/pandya.kei/logs/faiss_%j.out
#SBATCH --error=/scratch/pandya.kei/logs/faiss_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1
#
# Completes the faiss GPU check deferred by a CPU-node install
# (SIMFOUNDRY_SKIP_FAISS_GPU_VALIDATE=1). Minutes, not hours -- this is the only
# part of the install that genuinely needs a device, so it is the only part that
# should ever hold a GPU.
set -euo pipefail
export PATH=/scratch/pandya.kei/miniforge3/bin:$PATH
eval "$(mamba shell hook --shell bash)"
source /scratch/pandya.kei/repos/SimFoundry/scripts/installation/faiss_gpu.sh
unset SIMFOUNDRY_SKIP_FAISS_GPU_VALIDATE   # we DO want the real probe here

nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1
for env in simfoundry da3; do
    if [[ -x /scratch/pandya.kei/conda_envs/$env/bin/python ]]; then
        echo ">>> validating faiss GPU in '$env'"
        validate_faiss_gpu "$env"
    else
        echo ">>> env '$env' absent, skipping"
    fi
done
echo "FAISS GPU VALIDATION DONE"
