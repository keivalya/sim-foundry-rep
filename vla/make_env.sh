#!/bin/bash
set -x
export PATH=/scratch/pandya.kei/miniforge3/bin:$PATH
export HF_HOME=/scratch/pandya.kei/hf_cache
export PIP_CACHE_DIR=/scratch/pandya.kei/pip_cache
P=/scratch/pandya.kei/conda_envs/openpi
if [ ! -x $P/bin/python ]; then
  mamba create -y -p $P python=3.12 -c conda-forge || exit 1
fi
$P/bin/python -V
$P/bin/pip install --no-input -U pip setuptools wheel || exit 1
# torch cu128 (same family as proven-working simfoundry env)
$P/bin/pip install --no-input torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128 || exit 1
$P/bin/pip install --no-input "lerobot[pi]==0.6.1" || $P/bin/pip install --no-input "lerobot==0.6.1" || exit 1
$P/bin/python -c "import torch,lerobot;print('torch',torch.__version__,'lerobot',lerobot.__version__)"
touch $P/.openpi_env_ready
echo DONE_MAKE_ENV
