#!/bin/bash
#SBATCH --job-name=sf-pickplace
#SBATCH --output=/scratch/pandya.kei/logs/pickplace_%j.out
#SBATCH --error=/scratch/pandya.kei/logs/pickplace_%j.err
#SBATCH --time=01:00:00
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
ARGS=( --scene-name "${SCENE:-fruits_demo}" )
if [ -n "${SCENE_JSON:-}"   ]; then ARGS+=( --scene-json   "$SCENE_JSON" );   fi
if [ -n "${SETTLED_JSON:-}" ]; then ARGS+=( --settled-json "$SETTLED_JSON" ); fi
if [ -n "${PICK:-}"         ]; then ARGS+=( --pick         "${PICK//+/,}" );  fi
if [ -n "${VIDEO_NAME:-}"   ]; then ARGS+=( --video-name   "$VIDEO_NAME" );   fi
if [ -n "${REACH:-}" ]; then ARGS+=( --reach "${REACH}" ); fi
if [ -n "${MAX_GRASP_W:-}" ]; then ARGS+=( --max-grasp-width "${MAX_GRASP_W}" ); fi
if [ -n "${DOWN_AXIS:-}" ]; then ARGS+=( "--down-axis=${DOWN_AXIS}" ); fi
if [ -n "${ROBOT_H:-}" ]; then ARGS+=( --robot-height "${ROBOT_H}" ); fi
if [ -n "${GRIP_OPEN:-}" ]; then ARGS+=( --grip-open "${GRIP_OPEN}" ); fi
if [ -n "${CAM_LINK:-}" ]; then ARGS+=( --cam-parent-link "${CAM_LINK}" ); fi
if [ -n "${CAM_SCALE:-}" ]; then ARGS+=( --cam-dist-scale "${CAM_SCALE}" ); fi
if [ -n "${GRIP_RELEASE:-}" ]; then ARGS+=( --grip-release "${GRIP_RELEASE}" ); fi
if [ -n "${APPROACH_H:-}" ]; then ARGS+=( --approach "${APPROACH_H}" ); fi
if [ -n "${LIFT_H:-}" ]; then ARGS+=( --lift "${LIFT_H}" ); fi
if [ -n "${POS_TOL:-}" ]; then ARGS+=( --pos-tol "${POS_TOL}" ); fi
if [ -n "${PLACE_AZ:-}" ]; then ARGS+=( --place-azimuth "${PLACE_AZ}" ); fi
if [ -n "${CAM_W:-}" ]; then ARGS+=( --cam-width "${CAM_W}" ); fi
if [ -n "${CAM_H:-}" ]; then ARGS+=( --cam-height "${CAM_H}" ); fi
if [ "${CAM_SCENE:-0}" = "1" ]; then ARGS+=( --cam-scene-frame ); fi
if [ "${POSES_ONLY:-0}" = "1" ]; then ARGS+=( --poses-only ); fi
echo "ARGS: ${ARGS[*]}"
"$CONDA_PREFIX/bin/python" /scratch/pandya.kei/simfoundry-repro/policies/pick_and_place.py "${ARGS[@]}"
echo "=== artifacts ==="
ls -la "$REPO/Data/${SCENE:-fruits_demo}/pick_place/" 2>/dev/null
