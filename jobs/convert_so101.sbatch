#!/bin/bash
#SBATCH --job-name=so101-usd
#SBATCH --output=/scratch/pandya.kei/logs/so101usd_%j.out
#SBATCH --error=/scratch/pandya.kei/logs/so101usd_%j.err
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=sharing
#SBATCH --gres=gpu:l40s:1
set -euo pipefail
ENVS=/scratch/pandya.kei/conda_envs
export HF_HOME=/scratch/pandya.kei/hf_cache
export PATH=/scratch/pandya.kei/miniforge3/bin:$PATH
export LD_LIBRARY_PATH=$ENVS/simfoundry/lib:${LD_LIBRARY_PATH:-}   # GLIBCXX_3.4.30 for libcarb
export OMNI_KIT_ACCEPT_EULA=YES
export CONDA_PREFIX="$ENVS/simfoundry"
export PATH="$CONDA_PREFIX/bin:$PATH"
cd /scratch/pandya.kei/simfoundry-repro/robot/so101_src
"$CONDA_PREFIX/bin/python" /scratch/pandya.kei/convert_so101_usd.py \
  --urdf /scratch/pandya.kei/simfoundry-repro/robot/so101_src/so101_new_calib.urdf \
  --dest /scratch/pandya.kei/simfoundry-repro/robot/so101_usd/so101.usd
echo "=== artifacts ==="; ls -la /scratch/pandya.kei/simfoundry-repro/robot/so101_usd/ 2>/dev/null
