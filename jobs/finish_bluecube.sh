#!/bin/bash
# Score + contact-sheet one blue-cube rollout, and (optionally) queue the
# offline 6-angle render. Everything here is CPU-only and safe on the login node
# except the render, which goes to `short` via jobs/run_multiview_vla.sbatch.
#
#   bash jobs/finish_bluecube.sh v2_match match [render]
set -euo pipefail
PROJ=/scratch/pandya.kei/simfoundry-repro
E=$PROJ/conda_envs/simfoundry-editor/bin/python
NAME="$1"; SCENE="${2:-match}"; DO_RENDER="${3:-}"
RUN="$PROJ/results/so101/vla/bluecube/$NAME"
LAY="$PROJ/scene_data/bluecube/bc_${SCENE}_layout.json"
"$E" -I "$PROJ/tools/score_bluecube.py" --run "$RUN" --layout "$LAY"
"$E" -I "$PROJ/tools/bluecube_sheet.py" --run "$RUN"
if [ -n "$DO_RENDER" ]; then
  GC="$PROJ/results/so101/vla/bluecube/geom_cache_${SCENE}.npz"
  [ -f "$GC" ] || "$E" -I "$PROJ/tools/make_bluecube_geom_cache.py" \
      --scene "$PROJ/scene_data/bluecube/bc_${SCENE}.json" --out "$GC"
  sbatch --export=ALL,RUN="$RUN",GEOM_CACHE="$GC",W=720,H=450,FPS=30,STRIDE=1,\
TITLE="SO-101 + pi0.5: put the blue cube in the gray box" \
    "$PROJ/jobs/run_multiview_vla.sbatch"
fi
