#!/bin/bash
# Self-contained pi0.5 blue-cube campaign: submit N rollouts, then a finisher
# that scores them all, picks the best, and renders it.
#
# Everything runs as SLURM JOBS chained by --dependency, so once this returns
# you can close the terminal / end the Claude session and the campaign still
# completes on the cluster. Nothing depends on an interactive shell staying up.
#
#   bash jobs/bluecube_campaign.sh [N] [SCENE] [STEPS]
set -euo pipefail
PROJ=/scratch/pandya.kei/simfoundry-repro
N="${1:-6}"; SCENE="${2:-shift}"; STEPS="${3:-700}"
STAMP=$(date +%m%d_%H%M)
IDS=()

for i in $(seq 1 "$N"); do
  NAME="c${STAMP}_s${i}"
  # Sweep the assisted-grasp close threshold. The latch and the release pull in
  # opposite directions: d1_shift LATCHED with the jaw at 24.2, so a threshold of
  # 20 refuses that latch and nothing is ever grasped; but the policy's
  # release-open peaks at ~23-37 (median 30), so too high a threshold never
  # releases. The usable window is between them.
  # Measured from c1007_1702_s2 (the best pick so far, threshold 27):
  #   the grasp LATCHED with the jaw at 15.5, and after the grasp the policy's
  #   commanded gripper never exceeds 26.49 -- so T=27 can never release (0
  #   frames), T=24 releases on 11 frames, T=20 on 42. T must sit above the
  #   latch value (15.5) and below the post-grasp open commands.
  # CONTROLLED REPLICATION, not a sweep. c1007_1702_s2 (threshold 27, no
  # debounce) carried the cube 1805 mm and lifted it 185 mm; the other 22 runs
  # across three threshold settings all sit under 260 mm / 15 mm. That is a 7x
  # outlier, so before tuning anything further we need to know whether the
  # config reproduces it or whether it was a lucky sample of pi0.5's stochastic
  # trajectory. Every run here is IDENTICAL -- only the sampling differs.
  export SO101_AG_CLOSE_NORM=27.0
  export SO101_AG_OPEN_FRAMES=1
  for try in 1 2 3; do
    if OUT=$(STEPS="$STEPS" bash "$PROJ/jobs/submit_bluecube.sh" \
               "$NAME" "$PROJ/scene_data/bluecube/bc_${SCENE}.json" 2>&1); then
      ID=$(echo "$OUT" | grep -oE '[0-9]{6,}' | tail -1)
      [ -n "$ID" ] && { IDS+=("$ID"); echo "  $NAME thresh=$SO101_AG_CLOSE_NORM -> job $ID"; }
      break
    fi
    sleep 20   # sbatch intermittently returns "Unexpected message received"
  done
done

[ ${#IDS[@]} -gt 0 ] || { echo "nothing submitted"; exit 1; }

DEP=$(IFS=:; echo "${IDS[*]}")
FIN=$(sbatch --parsable --dependency=afterany:"$DEP" --kill-on-invalid-dep=yes \
      --job-name=bc-finish --partition=short --time=04:00:00 --cpus-per-task=8 --mem=32G \
      --output="$PROJ/logs/bc_finish_%j.out" \
      --export=ALL,SCENE="$SCENE",STAMP="$STAMP" "$PROJ/jobs/bluecube_finish.sbatch")
echo "  finisher -> job $FIN (runs after: $DEP)"
echo "Campaign $STAMP submitted. Safe to close the terminal."
