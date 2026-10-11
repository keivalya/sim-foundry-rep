#!/bin/bash
#SBATCH --job-name=so101-pi05-loop
#SBATCH --output=/scratch/pandya.kei/logs/pi05loop_%j.out
#SBATCH --error=/scratch/pandya.kei/logs/pi05loop_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=sharing
#SBATCH --gres=gpu:l40s:1
# Isaac Sim 5.1 / Kit 107.3 only works on the 570 driver branch.
#   d4050            595.58.03 (too new) -- 0/7 jobs ok, segfaults in librtx
#   d4074,d4075      same el9_7 kernel as d4050, assume affected
#   d4046            545.23.08 (too OLD) -- 0/5 jobs ok, segfaults in libusd_tf
# Known-good: d4047, d4049, d4051 (all 570.86.15).
#SBATCH --exclude=d4050,d4074,d4075,d4046
#
# pi0.5 closed-loop on the SO-101, in ONE job, on ONE L40S.
#
# TWO PROCESSES, ONE GPU, ONE JOB
# -------------------------------
# OmniGibson (numpy 1.26) and LeRobot (numpy 2.2) cannot share a conda env, so
# the policy runs as a server in `openpi` and the sim as a client in
# `simfoundry`, over a raw loopback socket. They fit on one l40s: 46 GB, and
# bf16 pi0.5 is ~8.8 GB next to Isaac's few GB.
#
# THE PROXY TRAP, AND ITS MIRROR IMAGE
# ------------------------------------
# http_proxy/https_proxy point at the cluster's squid, which INTERCEPTS
# 127.0.0.1. A proxied localhost call returns a squid 403 that reads exactly
# like an application auth error. The wire protocol here is a raw TCP socket,
# so it is never proxied and that problem cannot arise at all.
#
# DO NOT "fix" that by exporting no_proxy='*'. Job 10884015 did, and HUNG for
# 20 minutes inside og.Environment() immediately after "Welcome to OmniGibson":
#
#   ss -tnp  ->  SYN-SENT  10.99.104.51:45012 -> 52.92.202.250:443  (python)
#
# Omniverse Client makes an outbound HTTPS call (AWS) during scene load. Behind
# squid it fails FAST. With no_proxy='*' it tries to reach AWS DIRECTLY, this
# cluster blackholes the packets, and it sits in TCP SYN retry forever -- GPU at
# 0%, all 415 threads idle, no log line, no traceback, no timeout.
# So scope no_proxy to LOOPBACK ONLY. That leaves squid in charge of real
# outbound traffic, exactly as every Isaac job that has ever worked here had it.
#
# NODE EXCLUSIONS
# ---------------
# Isaac Sim 5.1 / Kit 107.3 segfaults inside SimulationApp.__init__ on the
# el9_7 kernel + NVIDIA 595 driver combination. d4050 is 0/6; d4074/d4075 are on
# the same new kernel. See jobs/run_wrist_cam.sbatch for the full evidence.
set -euo pipefail
REPO=/scratch/pandya.kei/repos/SimFoundry
PROJ=/scratch/pandya.kei/simfoundry-repro
ENVS=/scratch/pandya.kei/conda_envs

export no_proxy='127.0.0.1,localhost,::1' NO_PROXY='127.0.0.1,localhost,::1'
export HF_HOME=/scratch/pandya.kei/hf_cache
export HF_HUB_OFFLINE=0            # same env the smoke test loaded under
export HF_TOKEN=$(cat /scratch/pandya.kei/hf_cache/token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export OMNI_KIT_ACCEPT_EULA=YES
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export OMNIGIBSON_HEADLESS=1
export SO101_USD_PATH="${SO101_USD:-$PROJ/robot/so101_usd/so101_wristcam.usd}"
export PI05_PORT="${PI05_PORT:-$((40000 + RANDOM % 20000))}"

NT=${SLURM_CPUS_PER_TASK:-16}
# Kit sizes its thread pools from os.cpu_count() (the whole 96-core node), not
# the cgroup, so cap them at the allocation.
export KIT_EXTRA_ARGS="--/plugins/carb.tasking.plugin/threadCount=$NT --/plugins/omni.tbb.globalcontrol/maxThreadCount=$NT"
LOCAL=${TMPDIR:-/tmp}/og_${SLURM_JOB_ID}
mkdir -p "$LOCAL"
cp -a "$REPO/deps/BEHAVIOR-1K/OmniGibson/appdata" "$LOCAL/appdata" 2>/dev/null || mkdir -p "$LOCAL/appdata"
export OMNIGIBSON_APPDATA_PATH="$LOCAL/appdata"

OUT="${OUT_DIR:-$PROJ/results/so101/vla/run_${SLURM_JOB_ID}}"
mkdir -p "$OUT"
READY="$LOCAL/policy_ready"
rm -f "$READY" "$OUT/OK" "$OUT/FAILED"

nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | head -1
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRV_MAJ=${DRV%%.*}
if [ "${DRV_MAJ:-0}" -lt 560 ] || [ "${DRV_MAJ:-0}" -ge 580 ]; then
  echo "FATAL: driver $DRV on $(hostname) is outside the 570 branch Isaac 5.1 needs." >&2
  echo "       Add this node to the #SBATCH --exclude list." >&2
  exit 75   # distinct code: a node problem, not a code problem
fi
if [ "${DRV%%.*}" -ge 580 ] 2>/dev/null; then
  echo "!!! WARNING: NVIDIA driver $DRV on $(hostname) -- Isaac 5.1 segfaults on the 595 branch."
fi
echo "node=$(hostname) cpus=$NT port=$PI05_PORT out=$OUT"
echo "usd=$SO101_USD_PATH"
# robot/so101_og.py reads this: the gripper command (LeRobot units) at/above
# which the assisted grasp must RELEASE. 100 = OmniGibson's stock full-limit
# rule, which this policy can never reach. See the note in robot/so101_og.py.
echo "ag_close_norm=${SO101_AG_CLOSE_NORM:-20.0 (default)}  seed=${SEED:-unset}"

# Keep the robot class in sync; it is edited in robot/ and COPIED into OmniGibson.
cp "$PROJ/robot/so101_og.py" "$REPO/deps/BEHAVIOR-1K/OmniGibson/omnigibson/robots/so101.py"

# ---------------------------------------------------------------- policy ----
echo "=== starting pi0.5 server (openpi env) ==="
# POLICY_SERVER swaps in a different driver with the same wire protocol --
# vla/replay_server.py replays a recorded training episode, which is how the
# assisted-grasp latch/release rule is checked against a trajectory that is
# known to be correct. POLICY_ENV picks the env it runs in (the replay server
# needs only numpy + pandas, so it runs in `simfoundry`).
SRV_PY="${POLICY_SERVER:-$PROJ/vla/pi05_server.py}"
SRV_ENV="${POLICY_ENV:-$ENVS/openpi}"
SRV_ARGS=( --port "$PI05_PORT" --ready-file "$READY" --selftest )
if [ -n "${POLICY_SERVER:-}" ]; then
  # replay_server.py only: its cursor must advance at the client's chunk rate.
  [ -n "${EXEC_STEPS:-}" ]     && SRV_ARGS+=( --exec-steps "$EXEC_STEPS" )
  [ -n "${REPLAY_EPISODE:-}" ] && SRV_ARGS+=( --episode "$REPLAY_EPISODE" )
fi
# pi0.5 SAMPLES its trajectory, so the rollout is not deterministic (1 of 2
# attempts picked the cube up on 2026-10-07). SEED names the sample, which is
# how parallel jobs explore different ones instead of relying on entropy.
[ -n "${SEED:-}" ] && SRV_ARGS+=( --seed "$SEED" )
echo "policy driver: $SRV_PY  (env $SRV_ENV)"
"$SRV_ENV/bin/python" -I "$SRV_PY" "${SRV_ARGS[@]}" \
  > "$OUT/policy_server.log" 2>&1 &
SRV_PID=$!
cleanup() { kill "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT

for i in $(seq 1 600); do
  [ -f "$READY" ] && break
  kill -0 "$SRV_PID" 2>/dev/null || { echo "!!! policy server died:"; tail -40 "$OUT/policy_server.log"; exit 1; }
  sleep 2
done
[ -f "$READY" ] || { echo "!!! policy server never became ready"; tail -40 "$OUT/policy_server.log"; exit 1; }
echo "=== policy server ready after ${i}x2 s ==="
sed -n '1,40p' "$OUT/policy_server.log"

# ------------------------------------------------------------------- sim ----
export CONDA_PREFIX="$ENVS/simfoundry"
export PATH="$CONDA_PREFIX/bin:/scratch/pandya.kei/miniforge3/bin:$PATH"
export LD_LIBRARY_PATH=$ENVS/simfoundry/lib:${LD_LIBRARY_PATH:-}   # GLIBCXX_3.4.30 for libcarb
cd "$REPO"

ARGS=( --scene-json "$SCENE_JSON" --out-dir "$OUT" --port "$PI05_PORT" )
[ -n "${STEPS:-}"      ] && ARGS+=( --steps "$STEPS" )
[ -n "${EXEC_STEPS:-}" ] && ARGS+=( --exec-steps "$EXEC_STEPS" )
[ -n "${TASK:-}"       ] && ARGS+=( --task "$TASK" )
[ -n "${AFREQ:-}"      ] && ARGS+=( --action-frequency "$AFREQ" )
[ -n "${RFREQ:-}"      ] && ARGS+=( --rendering-frequency "$RFREQ" )
[ -n "${TOP_AZ:-}"     ] && ARGS+=( --top-az "$TOP_AZ" )
[ -n "${TOP_EL:-}"     ] && ARGS+=( --top-el "$TOP_EL" )
[ -n "${TOP_DIST:-}"   ] && ARGS+=( --top-dist "$TOP_DIST" )
# sbatch --export SPLITS ITS VALUE ON COMMAS, so a comma-separated vector
# arrives truncated to its first component: TOP_EYE=0.0368,-0.4252,0.3000 became
# "0.0368", the look vector had zero length, and look_at_quat died in np.cross
# (jobs 10900677/80/82/85). Accept '+' as the separator and translate it back.
[ -n "${TOP_EYE:-}"    ] && ARGS+=( --top-eye "${TOP_EYE//+/,}" )
[ -n "${TOP_TARGET:-}" ] && ARGS+=( --top-target "${TOP_TARGET//+/,}" )
[ -n "${TOP_FOCAL:-}"  ] && ARGS+=( --top-focal "$TOP_FOCAL" )
[ -n "${FRAME_OBJECTS:-}" ] && ARGS+=( --frame-objects "${FRAME_OBJECTS//+/,}" )
[ -n "${WARMUP:-}"     ] && ARGS+=( --warmup "$WARMUP" )
[ -n "${ACTION_OFFSET:-}" ] && ARGS+=( --action-offset "$ACTION_OFFSET" )
[ -n "${FOLLOWER_ALPHA:-}" ] && ARGS+=( --follower-alpha "$FOLLOWER_ALPHA" )
echo "ARGS: ${ARGS[*]}"

# Retry only a HARD crash (no OK, no FAILED marker) -- that is the Isaac startup
# segfault. A FAILED marker means Python caught and printed it; retrying burns
# the allocation for nothing.
ATT=${ATTEMPTS:-3}
for i in $(seq 1 "$ATT"); do
  echo "=== SIM ATTEMPT $i/$ATT  $(date -u +%H:%M:%S) ==="
  rc=0
  "$CONDA_PREFIX/bin/python" "$PROJ/policies/pi05_closed_loop.py" "${ARGS[@]}" || rc=$?
  echo "=== attempt $i exit=$rc OK=$([ -f "$OUT/OK" ] && echo yes || echo no) FAILED=$([ -f "$OUT/FAILED" ] && echo yes || echo no)"
  [ -f "$OUT/OK" ] && { echo "SUCCESS on attempt $i"; break; }
  [ -f "$OUT/FAILED" ] && { echo "python-level failure, not retrying"; break; }
  echo "hard crash (exit $rc) -- retrying"
done

# ---------------------------------------------------------------- videos ----
# The LIVE camera views, straight from the frames the policy actually saw.
FF=/shared/EL9/explorer/FFmpeg/7.1.1
export LD_LIBRARY_PATH=$FF/lib:$LD_LIBRARY_PATH
VFPS=${VIDEO_FPS:-30}   # control runs at 30 Hz, so 30 fps plays back in REAL TIME
mkdir -p "$OUT/videos"
for CAM in wrist_cam top_cam; do
  N=$(ls "$OUT/$CAM"/rgb_*.png 2>/dev/null | wc -l)
  [ "$N" -lt 2 ] && { echo "no frames for $CAM"; continue; }
  # The frames are rgb_<step>.png with gaps possible, so renumber via a list file.
  ls "$OUT/$CAM"/rgb_*.png | sort | awk '{printf "file '\''%s'\''\n", $0}' > "$LOCAL/$CAM.txt"
  "$FF/bin/ffmpeg" -y -v error -r "$VFPS" -f concat -safe 0 -i "$LOCAL/$CAM.txt" \
      -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -pix_fmt yuv420p -c:v libx264 -crf 18 \
      "$OUT/videos/so101_pi05_$CAM.mp4" && echo "wrote videos/so101_pi05_$CAM.mp4 ($N frames)"
done
# Side-by-side of the two live views.
if [ -f "$OUT/videos/so101_pi05_wrist_cam.mp4" ] && [ -f "$OUT/videos/so101_pi05_top_cam.mp4" ]; then
  "$FF/bin/ffmpeg" -y -v error -i "$OUT/videos/so101_pi05_top_cam.mp4" \
      -i "$OUT/videos/so101_pi05_wrist_cam.mp4" -filter_complex hstack=inputs=2 \
      -pix_fmt yuv420p -c:v libx264 -crf 18 "$OUT/videos/so101_pi05_live_pair.mp4" \
      && echo "wrote videos/so101_pi05_live_pair.mp4"
fi

kill "$SRV_PID" 2>/dev/null || true
sleep 3
echo "=== policy server log (tail) ==="
tail -25 "$OUT/policy_server.log"
rm -rf "$LOCAL"
echo "=== artifacts ==="
ls -la "$OUT" "$OUT/videos" 2>/dev/null | head -40
du -sh "$OUT"
echo "NEXT: offline multi-angle render (CPU, no GPU, ~1 h):"
echo "  $ENVS/simfoundry-editor/bin/python $PROJ/tools/render_episode_multiview.py \\"
echo "      --poses $OUT/poses.npz --outdir $OUT/multiview --width 720 --height 450 --fps 15"
