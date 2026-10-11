#!/bin/bash
#SBATCH --job-name=so101-wristcam
#SBATCH --output=/scratch/pandya.kei/logs/wristcam_%j.out
#SBATCH --error=/scratch/pandya.kei/logs/wristcam_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=sharing
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=d4050
#
# Wrist camera + live-render reliability probe. Needs RT cores -> `sharing`, 1 h cap.
#
# WHY THIS SCRIPT RETRIES
# -----------------------
# Isaac Sim 5.1 segfaults INSIDE SimulationApp.__init__, in librtx.scenedb /
# libcarb.scenerenderer-rtx, i.e. while Kit brings up the RTX scene renderer and
# its viewport -- BEFORE any scene, robot or camera exists. Evidence: the py-spy
# dumps Kit writes next to each .dmp in
# <appdata>/local/data/Kit/OmniGibson/3.8/*.py.txt put the main thread in
# SimulationApp._wait_for_viewport (jobs 10873560, 10872515, 10882447) or
# _prepare_ui (10872275), with _launch_app -> og.Environment.__init__ below it.
# So it is NOT "the RTX renderer corrupting this scene": there is no scene yet.
#
# AND IT IS ONE NODE. Every startup segfault so far was on d4050; every
# `sharing` job on d4047/d4049 either completed or only crashed at shutdown:
#   d4050  10871335 cancelled(hung) 10872275 10872515 10873560 10882447   0/5 ok
#   d4049  10871158 ok  10872270 ok  10876100 ok  10875292 crash-at-exit  3/4 ok
#   d4047  10871766 ok                                                    1/1 ok
# d4050 is the node that has been upgraded ahead of the rest:
#   d4050  kernel 5.14.0-611.5.1.el9_7   NVIDIA driver 595.58.03
#   d4049  kernel 5.14.0-362.13.1.el9_3  NVIDIA driver 570.86.15
#   d4051  (same older pair)             NVIDIA driver 570.86.15
# Isaac Sim 5.1 / Kit 107.3 predates the 595 driver branch. Hence --exclude.
# As the rest of the fleet is upgraded this will spread, so the job also prints
# the driver version; if it is >= 580, expect startup crashes and extend the
# exclude list. The retry loop stays as a cheap belt-and-braces (~75 s/attempt).
#
# OTHER SETTINGS AND WHY
#   OMNIGIBSON_APPDATA_PATH -> node-local. It defaults into the repo on
#     networked /scratch, 1.2 GB of Kit shader/scene cache shared by every
#     concurrent Isaac job; macros.py:131 says to move it on SLURM. Measured
#     "app ready" 65 s local vs 269 s shared on the worst observed run.
#   thread counts -> SimulationApp sets Kit's tasking and TBB thread pools to
#     min(os.cpu_count(), 32) (simulation_app.py:426-435). os.cpu_count()
#     reports the whole node (96 on these Xeons), NOT the cgroup allocation, so
#     an 8-CPU job used to run 32 Kit worker threads on 8 cores. Those args are
#     appended last (args.extend(unknown_args), simulation_app.py:533) so they
#     win over Isaac's own.
#   OMNIGIBSON_HEADLESS -> set for correctness, but note it is nearly a no-op
#     here: DISPLAY is unset in a batch job, so simulation_app.py:449-455
#     already adds --no-window and hideUi regardless.
set -euo pipefail
REPO=/scratch/pandya.kei/repos/SimFoundry
PROJ=/scratch/pandya.kei/simfoundry-repro
ENVS=/scratch/pandya.kei/conda_envs
export HF_HOME=/scratch/pandya.kei/hf_cache
export PATH=/scratch/pandya.kei/miniforge3/bin:$PATH
export LD_LIBRARY_PATH=$ENVS/simfoundry/lib:${LD_LIBRARY_PATH:-}   # GLIBCXX_3.4.30 for libcarb
export OMNI_KIT_ACCEPT_EULA=YES                                     # else Kit prompts and hangs
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export CONDA_PREFIX="$ENVS/simfoundry"
export PATH="$CONDA_PREFIX/bin:$PATH"

export OMNIGIBSON_HEADLESS="${OG_HEADLESS:-1}"
NT=${SLURM_CPUS_PER_TASK:-8}
export KIT_EXTRA_ARGS="${KIT_EXTRA_ARGS:---/plugins/carb.tasking.plugin/threadCount=$NT --/plugins/omni.tbb.globalcontrol/maxThreadCount=$NT}"
LOCAL=${TMPDIR:-/tmp}/og_${SLURM_JOB_ID}
mkdir -p "$LOCAL"
if [ "${OG_LOCAL_APPDATA:-1}" = "1" ]; then
  echo "seeding node-local appdata at $LOCAL/appdata ..."
  cp -a "$REPO/deps/BEHAVIOR-1K/OmniGibson/appdata" "$LOCAL/appdata" 2>/dev/null || mkdir -p "$LOCAL/appdata"
  export OMNIGIBSON_APPDATA_PATH="$LOCAL/appdata"
fi
export SO101_USD_PATH="${SO101_USD:-$PROJ/robot/so101_usd/so101_wristcam.usd}"

nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | head -1
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
if [ "${DRV%%.*}" -ge 580 ] 2>/dev/null; then
  echo "!!! WARNING: NVIDIA driver $DRV on $(hostname). Isaac Sim 5.1 segfaults in"
  echo "!!! librtx.scenedb during SimulationApp startup on the 595 branch."
  echo "!!! Add this node to the sbatch --exclude list."
fi
echo "node=$(hostname) cpus=$NT headless=$OMNIGIBSON_HEADLESS appdata=${OMNIGIBSON_APPDATA_PATH:-default}"
echo "usd=$SO101_USD_PATH"
echo "KIT_EXTRA_ARGS=$KIT_EXTRA_ARGS"
cd "$REPO"

OUT="${OUT_DIR:-$PROJ/results/so101/wristcam/run_${SLURM_JOB_ID}}"
mkdir -p "$OUT"
ARGS=( --scene-json "$SCENE_JSON" --out-dir "$OUT" )
if [ -n "${STEPS:-}"      ]; then ARGS+=( --steps "$STEPS" ); fi
if [ -n "${STRIDE:-}"     ]; then ARGS+=( --probe-stride "$STRIDE" ); fi
if [ -n "${CAM_W:-}"      ]; then ARGS+=( --cam-width "$CAM_W" ); fi
if [ -n "${CAM_H:-}"      ]; then ARGS+=( --cam-height "$CAM_H" ); fi
if [ -n "${RFREQ:-}"      ]; then ARGS+=( --rendering-frequency "$RFREQ" ); fi
if [ -n "${MODALITIES:-}" ]; then ARGS+=( --modalities "$MODALITIES" ); fi
if [ "${NO_EXTERNAL:-0}" = "1" ];    then ARGS+=( --no-external ); fi
if [ "${EXTERNAL_WRIST:-0}" = "1" ]; then ARGS+=( --external-wrist ); fi
echo "ARGS: ${ARGS[*]}"

# Retry only a HARD crash (no OK and no FAILED marker). A FAILED marker means
# the script caught a Python exception and printed the traceback -- retrying
# that just burns the allocation.
ATT=${ATTEMPTS:-4}
rm -f "$OUT/OK" "$OUT/FAILED"
for i in $(seq 1 "$ATT"); do
  echo "=== ATTEMPT $i/$ATT  $(date -u +%H:%M:%S) ==="
  rc=0
  "$CONDA_PREFIX/bin/python" "$PROJ/policies/wrist_cam_probe.py" "${ARGS[@]}" || rc=$?
  echo "=== attempt $i exit=$rc  OK=$([ -f "$OUT/OK" ] && echo yes || echo no)  FAILED=$([ -f "$OUT/FAILED" ] && echo yes || echo no)"
  if [ -f "$OUT/OK" ]; then echo "SUCCESS on attempt $i"; break; fi
  if [ -f "$OUT/FAILED" ]; then echo "python-level failure, not retrying"; break; fi
  echo "hard crash (exit $rc) -- retrying"
done

# Keep Kit's crash evidence: the .py.txt dumps say WHERE the process was.
mkdir -p "$OUT/kitcrash"
find "${OMNIGIBSON_APPDATA_PATH:-$REPO/deps/BEHAVIOR-1K/OmniGibson/appdata}" \
     -name '*.py.txt' -newermt '-2 hours' -exec cp {} "$OUT/kitcrash/" \; 2>/dev/null || true
echo "=== kit crash dumps kept: $(ls "$OUT/kitcrash" | wc -l)"
for f in "$OUT"/kitcrash/*.py.txt; do [ -f "$f" ] && { echo "--- $f"; sed -n '4,12p' "$f"; }; done
rm -rf "$LOCAL"
echo "=== artifacts ==="
ls -la "$OUT" | head -20
du -sh "$OUT"
