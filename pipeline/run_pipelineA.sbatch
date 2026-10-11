#!/bin/bash
#SBATCH --job-name=simfoundry-pipelineA
#SBATCH --output=/scratch/pandya.kei/logs/pipelineA_%j.out
#SBATCH --error=/scratch/pandya.kei/logs/pipelineA_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#
# Pipeline A reconstruction on docs/assets/example_videos/Fruits.mp4, stages 1-12.
#
# WHY 1-12 ONLY: stages 13/14 are the only ones that import omnigibson, and Isaac
# Sim needs RT cores, which a100/h200 do not have. Those two run separately on
# `sharing` (l40s/a6000) -- see the tail of this file.
#
# WHY --no-stream: streaming co-resides stages 5-8 on one GPU. We route stage 5's
# object removal through LOCAL FLUX (the free Gemini tier has no image model), so
# the Kontext pipeline sits on-GPU -- which cfg stream_subseq.stage_vram_gb[5]=14
# does not account for. Serial stages each get the whole card.
#
# WHY low_vram=true: stage 7 shape gen needs ~29 GiB at the default (resolution
# 1536) and ~6 GiB with CPU offload (resolution 1024). A100 ships as 40 GB and
# 80 GB and the scheduler does not expose which; 29 GiB is uncomfortably close on
# a 40 GB card. Stages are individually re-runnable, so get a COMPLETE result
# first, then optionally redo just stage 7 at full resolution:
#     ... run.sh --scene-name fruits_demo --include 7 -- s7_mesh.low_vram=false
set -euo pipefail

REPO=/scratch/pandya.kei/repos/SimFoundry
ENVS=/scratch/pandya.kei/conda_envs

# --- pre-flight: refuse to run against an incomplete install ----------------
for e in simfoundry hunyuan da3; do
    [[ -f "$ENVS/$e/.simfoundry_install_complete" ]] || {
        echo "FATAL: env '$e' has no completion marker; install did not finish." >&2; exit 1; }
done
# Stages 11 and 12 are CPU-only (verified: no torch/CUDA imports; coacd +
# pymeshlab + PyBullet p.connect(p.DIRECT)). Running just those on `short`
# skips the gpu queue entirely, which is otherwise the dominant wait.
if [ "${NO_GPU:-0}" = "1" ]; then
    echo "NO_GPU=1 -> skipping GPU pre-flight (CPU-only stages)"
    LOW_VRAM=true
else
    CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    awk "BEGIN{exit !($CAP < 7.5)}" && { echo "FATAL: compute_cap $CAP < 7.5" >&2; exit 1; }
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
fi

export HF_HOME=/scratch/pandya.kei/hf_cache
export PATH=/scratch/pandya.kei/miniforge3/bin:/shared/EL9/explorer/FFmpeg/7.1.1/bin:$PATH
export LD_LIBRARY_PATH=/shared/EL9/explorer/FFmpeg/7.1.1/lib:$ENVS/simfoundry/lib:${LD_LIBRARY_PATH:-}
export OMNI_KIT_ACCEPT_EULA=YES

# install_simfoundry.sh installs flash-attn 2.8.3 (for FoundationStereo), but
# xformers/ops/fmha/flash.py:73-82 hard-requires >=2.7.1,<=2.7.4 and raises on
# import. diffusers.loaders.ip_adapter pulls xformers in, so `from diffusers
# import FluxKontextPipeline` dies -- and vlm.py:1150 then reports it as
# "requires a diffusers version that provides FluxKontextPipeline", which is
# misleading: diffusers is 0.40.0 and ships the class fine.
# This env var is xformers' own documented bypass (same line 77). Safe here:
# FLUX runs through PyTorch SDPA, not xformers' flash kernels, and the only
# flash-attn consumer (FoundationStereo) is unused since s2_depth.backend=da3.
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1

# vlm.py:246 reads this; default 8. Stage 11 died twice on transient provider
# errors after exhausting the budget -- first 429 RESOURCE_EXHAUSTED (free-tier
# daily cap, since reset), then 503 UNAVAILABLE "model is currently experiencing
# high demand". Both route through the rate-limit retry path with exponential
# backoff, so a larger budget rides out a spike instead of failing the stage.
export SIMFOUNDRY_REMOTE_RATE_LIMIT_ATTEMPTS=${SIMFOUNDRY_REMOTE_RATE_LIMIT_ATTEMPTS:-25}
echo "gemini retry budget: $SIMFOUNDRY_REMOTE_RATE_LIMIT_ATTEMPTS attempts"
export GEMINI_API_KEY="$(grep -m1 '^GEMINI_API_KEY=' "$REPO/scripts/installation/api_keys.txt" | cut -d= -f2-)"
[ -n "$GEMINI_API_KEY" ] || { echo "FATAL: GEMINI_API_KEY empty" >&2; exit 1; }
eval "$(mamba shell hook --shell bash)"
cd "$REPO"

VRAM_MIB=$( [ "${NO_GPU:-0}" = "1" ] && echo 0 || nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 )
if [ "${VRAM_MIB:-0}" -ge 70000 ]; then
    LOW_VRAM=false   # stage 7 at resolution 1536 (~29 GiB) -- better meshes
else
    LOW_VRAM=true    # CPU offload, resolution 1024 (~6 GiB)
fi
echo "VRAM=${VRAM_MIB} MiB -> s7_mesh.low_vram=$LOW_VRAM"

# Clean slate unless RESUME=1. Stage 3 picks the canonical frame with a VLM in
# `hybrid` mode, so it is NON-DETERMINISTIC across runs: a retry can select a
# different frame than the leftover s5_scene/ was built on, and stage 5 then
# (correctly) refuses with "holds a decomposition of frame N, but this run is
# built on frame M. Resuming would mix object crops from two viewpoints."
# Stages 1b-4 only cost ~6 min to regenerate, so a clean run beats stale state.
# STAGES=7,8,10,11,12 resumes after a late failure instead of redoing hours of
# work; default runs everything except 13/14 (those need RT cores, absent on a100).
# NOTE: sbatch --export splits on commas, so a value like STAGES=7,8,10 silently
# becomes STAGES=7 plus bogus entries. Pass them '+'-separated instead:
#     sbatch --export=ALL,RESUME=1,STAGES=7+8+10+11+12 run_pipelineA.sbatch
if [ -n "${STAGES:-}" ]; then
    STAGE_SEL="--include $(echo "$STAGES" | tr '+' ',')"
else
    STAGE_SEL="--exclude 13,14"
fi
echo "stage selection: $STAGE_SEL   RESUME=${RESUME:-0}"

# --- GPU keep-alive -------------------------------------------------------
# Job 10837346 (whale recovery) was CANCELLED by root after 1:26:58, the same
# signature as 10773471 / 10801646. The shared trait is a long stretch with the
# GPU reserved but idle: that run printed its header at 19:26:48 and emitted no
# pipeline log line until 20:51:50 -- 85 minutes of silence -- then died two
# minutes into stage 1b, which is ffmpeg and touches no GPU at all.
# gpu_keepalive runs duty-cycled SGEMMs (~200 MB of a 40 GB card) so the device
# never reads as idle. Set KEEPALIVE=0 to disable.
#
# NOTE: this mitigates a REAPER WE HAVE NOT CONFIRMED. The 85-minute gap is
# itself unexplained and may be the real problem; the timestamped markers below
# exist to pin that down on the next run.
KEEPALIVE_PID=""
cleanup_keepalive() {
    if [ -n "$KEEPALIVE_PID" ] && kill -0 "$KEEPALIVE_PID" 2>/dev/null; then
        kill "$KEEPALIVE_PID" 2>/dev/null || true
        echo "[$(date -Is)] keepalive stopped (pid $KEEPALIVE_PID)"
    fi
}
trap cleanup_keepalive EXIT
if [ "${KEEPALIVE:-1}" = "1" ] && [ -x /scratch/pandya.kei/simfoundry-repro/pipeline/gpu_keepalive ]; then
    /scratch/pandya.kei/simfoundry-repro/pipeline/gpu_keepalive 4096 20 400 \
        > "/scratch/pandya.kei/logs/keepalive_${SLURM_JOB_ID}.log" 2>&1 &
    KEEPALIVE_PID=$!
    echo "[$(date -Is)] keepalive started (pid $KEEPALIVE_PID)"
fi

echo "[$(date -Is)] >>> entering setup"

SCENE="${SCENE:-fruits_demo}"
VIDEO="${VIDEO:-$REPO/docs/assets/example_videos/Fruits.mp4}"
SCENE_DIR="$REPO/Data/$SCENE"
if [ "${RESUME:-0}" != "1" ] && [ -d "$SCENE_DIR" ]; then
    echo "[$(date -Is)] >>> removing stale scene dir $SCENE_DIR (set RESUME=1 to keep)"
    rm -rf "$SCENE_DIR"
    echo "[$(date -Is)] >>> wipe done"
fi

# Stage 5's sequential-removal loop ran 32 ITERATIONS over only 6 real objects:
# local FLUX repeatedly failed to remove the banana, so it was re-detected 24
# times with valid_pixel_proportion decaying to 0.0. 16/32 iterations are flagged
# is_valid_removed_obj=false, and because s6_upsample has check_valid=false +
# include_invalid=true, ALL 32 were upsampled and passed downstream.
# These six are the only iterations with valid_pixel_proportion == 1.0, one per
# object in obj_cat_list (pear, orange plate, apple, banana, orange, teal plate).
# Without this, stage 7 would spend hours generating ~26 meshes of one smeared
# banana and pollute the final scene.
# [0,20,28,29,30,31] is SPECIFIC TO fruits_demo (the six iterations whose
# valid_pixel_proportion == 1.0 after FLUX mangled the banana). It must NOT be
# applied to any other scene. Default to null = process everything.
if [ -n "${OBJ_IDX:-}" ]; then :
elif [ "$SCENE" = "fruits_demo" ]; then OBJ_IDX="[0,20,28,29,30,31]"
else OBJ_IDX="null"; fi
echo "stage 7/8/11 object_indices=$OBJ_IDX"

# FAITHFUL=1 runs the repo's DEFAULT config -- no model overrides at all.
# Every override below existed solely to work around the Gemini FREE tier, which
# gave limit:0 on all pro and image models:
#   detection/vlm/front_pick -> downgraded pro to flash
#   removal_model, s6 model  -> swapped Gemini image gen for LOCAL FLUX, which is
#                               what mangled the banana (24 re-detections,
#                               16/32 iterations invalid) and forced the
#                               object_indices restriction
#   obj_removal_prompt_order -> [outline,bbox] because mask is unstable for FLUX
# With a PAID key the defaults (gemini-3.1-pro-preview / gemini-3-pro-image)
# all work, so the faithful path is simply to pass nothing.
F=gemini-3-flash-preview
if [ "${FAITHFUL:-0}" = "1" ]; then
    echo ">>> FAITHFUL mode: using repo default models (pro + gemini image gen)"
    OVERRIDES=( "s7_mesh.low_vram=$LOW_VRAM" )
    # s5_scene.boundary_proportion defaults to 0.02: any mask within 2% of the
    # image border is pruned as "near image boundary". On reference_video.mp4
    # the white whale toy hugs the left edge, so it was detected by the VLM in
    # every iteration and then pruned every time -- it never reached the scene.
    # BOUNDARY=<p> tightens that margin so edge-hugging objects survive.
    [ -n "${BOUNDARY:-}" ] && OVERRIDES+=( "s5_scene.boundary_proportion=$BOUNDARY" )
    [ "$OBJ_IDX" != "null" ] && OVERRIDES+=( "s7_mesh.object_indices=$OBJ_IDX" \
                                             "s8_pose.object_indices=$OBJ_IDX" \
                                             "s11_sim.object_indices=$OBJ_IDX" )
else
    OVERRIDES=(
      "s3_ground.detection_model=$F" "s5_scene.detection_model=$F"
      "s8_pose.front_pick_model=$F"  "s11_sim.vlm_model=$F"
      "s5_scene.removal_model=flux"  "s6_upsample.model=flux"
      "s5_scene.obj_removal_prompt_order=[outline,bbox]"
      "s7_mesh.low_vram=$LOW_VRAM"
      "s7_mesh.object_indices=$OBJ_IDX" "s8_pose.object_indices=$OBJ_IDX"
      "s11_sim.object_indices=$OBJ_IDX"
    )
fi
echo "overrides: ${OVERRIDES[*]}"

echo "[$(date -Is)] >>> handing off to run.sh (anything slow before the first"
echo "                stage log line is inside run.sh env activation / imports)"
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name "$SCENE" \
  --video-fpath "$VIDEO" \
  --no-stream \
  ${STAGE_SEL} \
  -- "${OVERRIDES[@]}"

echo "=== PIPELINE A (stages 1-12) FINISHED ==="
ls -la "$REPO/Data/$SCENE" 2>/dev/null | head -20
