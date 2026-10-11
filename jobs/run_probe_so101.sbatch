#!/bin/bash
#SBATCH --job-name=so101-probe
#SBATCH --output=/scratch/pandya.kei/logs/so101probe_%j.out
#SBATCH --error=/scratch/pandya.kei/logs/so101probe_%j.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=sharing
#SBATCH --gres=gpu:l40s:1
#
# Scripted pick-and-place in our reconstructed scene. Needs RT cores (Isaac Sim),
# so `sharing` -- which caps at 1h. Zero Gemini calls, so the free-tier quota is
# irrelevant here. Keeps the GPU busy throughout (Isaac renders every step), so
# the idle-GPU reaper that killed jobs 10773471/10801646 should not trigger.
set -euo pipefail
REPO=/scratch/pandya.kei/repos/SimFoundry
ENVS=/scratch/pandya.kei/conda_envs
export HF_HOME=/scratch/pandya.kei/hf_cache
export PATH=/scratch/pandya.kei/miniforge3/bin:$PATH
export LD_LIBRARY_PATH=$ENVS/simfoundry/lib:${LD_LIBRARY_PATH:-}   # GLIBCXX_3.4.30 for libcarb
export OMNI_KIT_ACCEPT_EULA=YES                                     # else Kit prompts and hangs
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
eval "$(mamba shell hook --shell bash)"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -1
cd "$REPO"
# Call the env's interpreter directly. `mamba run --no-capture-output ... "$@"`
# died with: exec: --: invalid option (its wrapper mishandles the passthrough).
# CONDA_PREFIX is set by hand because omnigibson/isaacsim read it at import.
export CONDA_PREFIX="$ENVS/simfoundry"
export PATH="$CONDA_PREFIX/bin:$PATH"
"$CONDA_PREFIX/bin/python" /scratch/pandya.kei/simfoundry-repro/policies/probe_so101.py
echo "=== artifacts ==="
ls -la /scratch/pandya.kei/simfoundry-repro/results/so101/probe/ 2>/dev/null
