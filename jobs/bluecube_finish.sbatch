#!/bin/bash
# Score every rollout of one campaign, pick the best, render it. Runs as a
# dependent Slurm job so the campaign needs no interactive shell.
set -uo pipefail
PROJ=/scratch/pandya.kei/simfoundry-repro
E=$PROJ/conda_envs/simfoundry-editor/bin/python
LAY="$PROJ/scene_data/bluecube/bc_${SCENE}_layout.json"
SUM="$PROJ/results/so101/vla/bluecube/campaign_${STAMP}.txt"
: > "$SUM"
BEST=""; BESTKEY=-1
for RUN in "$PROJ"/results/so101/vla/bluecube/c${STAMP}_s*; do
  [ -f "$RUN/poses.npz" ] || { echo "$(basename $RUN): no poses" >> "$SUM"; continue; }
  "$E" -I "$PROJ/tools/score_bluecube.py" --run "$RUN" --layout "$LAY" >/dev/null 2>&1
  [ -f "$RUN/score.json" ] || continue
  read -r KEY LINE < <("$E" -I -c "
import json,sys
x=json.load(open('$RUN/score.json')); c=x.get('cube',{}); g=x.get('gripper',{}); s=x.get('success',{})
inbox=1 if s.get('cube_in_box_final') else 0
# rank: in-box first, then lift, then path
key=inbox*1e6 + c.get('max_lift_m',0)*1000 + c.get('path_length_m',0)
print('%.3f %s path=%.0fmm lift=%.0fmm closed=%d IN_BOX=%s' % (key,'$(basename $RUN)',
  c.get('path_length_m',0)*1000, c.get('max_lift_m',0)*1000,
  g.get('frames_closed_on_cube',0), s.get('cube_in_box_final')))")
  echo "$LINE" >> "$SUM"
  awk -v a="$KEY" -v b="$BESTKEY" 'BEGIN{exit !(a>b)}' && { BESTKEY="$KEY"; BEST="$RUN"; }
done
echo "BEST: $BEST" >> "$SUM"
cat "$SUM"
[ -n "$BEST" ] || exit 0
"$E" -I "$PROJ/tools/bluecube_sheet.py" --run "$BEST" >/dev/null 2>&1
GC="$PROJ/results/so101/vla/bluecube/geom_cache_${SCENE}.npz"
[ -f "$GC" ] || "$E" -I "$PROJ/tools/make_bluecube_geom_cache.py" \
    --scene "$PROJ/scene_data/bluecube/bc_${SCENE}.json" --out "$GC"
sbatch --export=ALL,RUN="$BEST",GEOM_CACHE="$GC",W=720,H=450,FPS=30,STRIDE=1,\
TITLE="SO-101 + pi0.5: pick up the blue cube and place it in the gray box" \
  "$PROJ/jobs/run_multiview_vla.sbatch"
