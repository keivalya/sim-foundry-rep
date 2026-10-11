#!/bin/bash
# Submit one pi0.5 blue-cube rollout. Everything the variant needs is an env var,
# so variants differ only in this call. TASK contains spaces, which `sbatch
# --export=ALL,K=V` cannot carry, so it is exported into the environment first
# and picked up by plain --export=ALL.
set -euo pipefail
PROJ=/scratch/pandya.kei/simfoundry-repro
NAME="$1"; shift
export SCENE_JSON="$1"; shift
export OUT_DIR="$PROJ/results/so101/vla/bluecube/$NAME"
# The training prompt, verbatim from the dataset's meta/tasks.parquet.
export TASK="Pick up the blue cube and place it in the gray box"
# The WIDENED asset -- see tools/widen_so101_limits.py. Without it the policy's
# shoulder_lift command is clipped at -100 and the arm parks on the stop.
# so101_wrist_matchcam_wide.usd -- NOT so101_wristcam_wide.usd. The matchcam
# variant re-aims the wrist camera down the gripper's approach axis; that was the
# single biggest lever on grasp quality (halved closest approach, tripled
# jaw-shut frames) and is what produced the validated d1_shift pick.
export SO101_USD="${SO101_USD:-$PROJ/robot/so101_usd/so101_wrist_matchcam_wide.usd}"
export STEPS="${STEPS:-450}"
export EXEC_STEPS="${EXEC_STEPS:-10}"
export ATTEMPTS="${ATTEMPTS:-2}"
# top_cam reproduces the training rig: a third-person view from the FAR side of
# the mat, 30 cm up and 17 cm beyond the objects, pitched 59 deg down, f=17 mm
# (63 deg HFOV -> a 40 cm wide frame at the objects, which is what the training
# frames measure). It is NOT an overhead camera despite the name.
export TOP_EYE="${TOP_EYE:-0.0358,-0.3882,0.3000}"
export TOP_TARGET="${TOP_TARGET:-0.0358,-0.2182,0.0200}"
export TOP_FOCAL="${TOP_FOCAL:-17.0}"
# ASSISTED-GRASP GRIPPER CALIBRATION. The gripper command, in LeRobot units, at
# or above which the assisted grasp releases. OmniGibson's stock rule compares
# against the jaw's FULL mechanical limit (= 100 here), which this checkpoint
# never commands (its max is 37.3), so the grasp latched on an OPEN jaw and
# could never let go. 20.0 sits in the measured gap between the dataset's
# "shut on the cube" commands (11-17.5) and its "open / release" ones (24-36).
# Full note in robot/so101_og.py. Set to 100 to reproduce the old behaviour.
export SO101_AG_CLOSE_NORM="${SO101_AG_CLOSE_NORM:-24.0}"
export SO101_AG_OPEN_FRAMES="${SO101_AG_OPEN_FRAMES:-4}"
# Names the pi0.5 flow-matching sample; leave unset for an entropy-seeded one.
export SEED="${SEED:-}"
export FOLLOWER_ALPHA="${FOLLOWER_ALPHA:-}"
mkdir -p "$OUT_DIR"
for i in 1 2 3 4 5; do
  # A previous attempt may have SUCCEEDED and still returned non-zero (sbatch
  # intermittently prints "Unexpected message received" after queueing), which
  # produced two jobs writing to the same OUT_DIR. Check before retrying.
  if squeue -u "$USER" -h -n "bc-$NAME" -o %i 2>/dev/null | grep -q .; then
    echo "$NAME already queued; not resubmitting" >&2; exit 0
  fi
  if JID=$(sbatch --parsable --job-name="bc-$NAME" --export=ALL "$PROJ/jobs/run_pi05_closed_loop.sbatch" 2>&1); then
    echo "$NAME -> job $JID  out=$OUT_DIR"; echo "$JID" > "$OUT_DIR/JOBID"; exit 0
  fi
  echo "sbatch attempt $i failed: $JID" >&2; sleep $((i*5))
done
echo "FAILED to submit $NAME" >&2; exit 1
