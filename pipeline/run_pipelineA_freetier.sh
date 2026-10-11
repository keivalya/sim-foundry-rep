#!/usr/bin/env bash
# Pipeline A on the Fruits example video, with Gemini overrides for a FREE
# AI Studio key.
#
# Measured 2026-09-25 against this key:
#   gemini-2.5-pro / 2.5-flash / 3-pro-preview  -> 404 "no longer available to new users"
#   gemini-3.1-pro-preview                      -> 429, free-tier quota limit: 0
#   gemini-3-pro-image / 2.5-flash-image        -> 429 quota
#   gemini-3-flash-preview                      -> WORKS (text AND vision)
#
# So: every text/vision key -> gemini-3-flash-preview, every image-GENERATION
# key -> flux (FLUX.1-Kontext-dev, runs locally via diffusers). Derived from a
# full audit of scripts/cfg/real2sim_cfg.yaml -- 15 model keys, not the obvious
# handful. Delete all of this if you move to a paid tier or Vertex.
#
# Two keys deliberately NOT overridden:
#   s3_ground.frame_selection.vlm_model = null, which frame_selection.py resolves
#       to s3_ground.detection_model -- so our override propagates. (The dotted
#       path is under frame_selection, not s3_ground directly.)
#   s6_upsample.validity_model = "gemini", but hardcodes the dead
#       gemini-3-pro-preview in 6_upsample_object_images.py:156 and is
#       unreachable because s6_upsample.check_valid defaults to false.
#       If you ever set check_valid=true, that stage WILL break.
set -euo pipefail

export HF_HOME=/scratch/pandya.kei/hf_cache

# Stages 13/14 load Isaac Sim. Its libcarb.so needs GLIBCXX_3.4.30, but Rocky
# 9.7's /lib64/libstdc++.so.6 stops at 3.4.29; the conda env ships 3.4.36.
# Without this, Kit dies with "Unable to bootstrap inner kit kernel".
# OMNI_KIT_ACCEPT_EULA avoids an interactive EULA prompt that hangs batch jobs.
export LD_LIBRARY_PATH=/scratch/pandya.kei/conda_envs/simfoundry/lib:${LD_LIBRARY_PATH:-}
export OMNI_KIT_ACCEPT_EULA=YES

# vlm.py's load_api_keys() only walks vlm.py's PARENT dirs for api_keys.txt, and
# the real file lives in scripts/installation/ -- which is not a parent. A root
# symlink fixes that (repo root IS a parent); exporting here is the belt-and-
# braces so this works regardless of cwd or symlink state. Without one of the
# two, resolve_gemini_auth raises on the first VLM call, i.e. stage 3.
export GEMINI_API_KEY="$(grep -m1 '^GEMINI_API_KEY=' \
    /scratch/pandya.kei/repos/SimFoundry/scripts/installation/api_keys.txt | cut -d= -f2-)"
[ -n "$GEMINI_API_KEY" ] || { echo "FATAL: GEMINI_API_KEY empty" >&2; exit 1; }
export PATH=/scratch/pandya.kei/miniforge3/bin:/shared/EL9/explorer/FFmpeg/7.1.1/bin:$PATH
export LD_LIBRARY_PATH=/shared/EL9/explorer/FFmpeg/7.1.1/lib:${LD_LIBRARY_PATH:-}
cd /scratch/pandya.kei/repos/SimFoundry

F=gemini-3-flash-preview
SCENE="${1:-fruits_demo}"
VIDEO="${2:-docs/assets/example_videos/Fruits.mp4}"

bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name "$SCENE" \
  --video-fpath "$VIDEO" \
  -- \
    s3_ground.detection_model=$F \
    s5_scene.detection_model=$F \
    s8_pose.front_pick_model=$F \
    s11_sim.vlm_model=$F \
    s5_scene.removal_model=flux \
    s6_upsample.model=flux \
    's5_scene.obj_removal_prompt_order=[outline,bbox]'

# --- if you add --detect-articulation (stage 9), also pass: ---------------
#     s9_articulate_objects.vlm_model=$F \
#     s9_articulate_objects.tree_model=$F \
#     s9_articulate_objects.merge_model=$F \
#     s9_articulate_objects.s5_model=$F
#
# --- Pipeline B (cousins + task proposal) --------------------------------
#     propose_scene_task.vlm_model=$F              # OK, text only
#     prompt_cousin_structured.text_model=$F       # OK, text only
#
#     prompt_cousin_structured.image_model -> NO FREE-TIER PATH (verified).
#     1_prompt_object_cousins.py:243 passes it straight into Gemini(...);
#     there is no "flux" branch anywhere in B_augmentation. Every Gemini
#     image model (gemini-3-pro-image, gemini-2.5-flash-image) is 429
#     quota-blocked, and gemini-3-flash-preview cannot generate images.
#     => digital-cousin generation needs a PAID tier or Vertex.
#        Pipeline A is unaffected.
