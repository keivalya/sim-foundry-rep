#!/bin/bash
#SBATCH --job-name=so101-vla-mv
#SBATCH --output=/scratch/pandya.kei/logs/vlamv_%j.out
#SBATCH --error=/scratch/pandya.kei/logs/vlamv_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#
# Offline multi-angle render of a pi0.5 closed-loop episode, from the pose log.
#
# Separate from jobs/run_pi05_closed_loop.sbatch on purpose: this needs NO GPU
# and no Isaac (it rasterises the USD visual meshes in numpy), and it takes
# ~1-2 h, which does not fit under the `sharing` partition's 1 h cap. The GPU
# job already wrote the LIVE wrist_cam / top_cam videos; this adds the five
# static rig angles plus the synthetic wrist-eye view and the 2x3 grid.
#
#   sbatch --export=ALL,RUN=/scratch/.../results/so101/vla/run_<jobid> \
#          jobs/run_multiview_vla.sbatch
#
# Optional: MV_OUT=<dir> to write somewhere other than $RUN/multiview (so a
# quick STRIDE=5 preview and the full STRIDE=1 render can run at the same time
# without clobbering each other's frames), W/H/FPS/STRIDE, GEOM_CACHE.
set -euo pipefail
PROJ=/scratch/pandya.kei/simfoundry-repro
E=$PROJ/conda_envs/simfoundry-editor/bin/python
RUN="${RUN:?set RUN=<the closed-loop output dir>}"
[ -f "$RUN/poses.npz" ] || { echo "no $RUN/poses.npz"; exit 1; }
"$E" "$PROJ/tools/render_episode_multiview.py" \
  --poses "$RUN/poses.npz" \
  --outdir "${MV_OUT:-$RUN/multiview}" \
  --geom-cache "${GEOM_CACHE:-$PROJ/results/so101/multiview/geom_cache.npz}" \
  --title "${TITLE:-SO-101 driven closed-loop by pi0.5}" \
  --width "${W:-720}" --height "${H:-450}" --fps "${FPS:-30}" --stride "${STRIDE:-1}"
echo "=== artifacts ==="
ls -la "${MV_OUT:-$RUN/multiview}/videos" 2>/dev/null
