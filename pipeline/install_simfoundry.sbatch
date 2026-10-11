#!/bin/bash
#SBATCH --job-name=simfoundry-install
#SBATCH --output=/scratch/pandya.kei/logs/install_%j.out
#SBATCH --error=/scratch/pandya.kei/logs/install_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# ---------------------------------------------------------------------------
# PARTITION / GPU -- surveyed on Explorer 2026-10-02:
#   gpu          MaxTime 8h   a100(8.0) h200(9.0) t4(7.5) v100(7.0)
#   sharing      MaxTime 1h   l40s(8.9) a6000(8.6) a5000(8.6) l40 rtx6000 h100
#   short        MaxTime 2d   CPU only, plus one v100 node
#   courses-gpu  MaxTime 1d   p100(6.0) v100(7.0)   <- both too old
#
# The RT-core cards (l40s/a6000) live ONLY in `sharing`, which caps at 1 HOUR --
# far too short for this install. So we BUILD on an a100 in `gpu` (8h) and set
# TORCH_CUDA_ARCH_LIST explicitly below, so the compiled extensions also run on
# l40s/a6000 later. Lack of RT cores does not affect building, only running
# Isaac Sim. Split at run time:
#   stages 1-11 + install  -> gpu / a100        (8h, no RT cores needed)
#   stages 12-14, Pipeline C -> sharing / l40s  (1h chunks, RT cores required)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
# ---------------------------------------------------------------------------
#
# Builds the 3 conda envs Pipeline A needs on a video input:
#     simfoundry (stages 3,4,5,6,8,10-14)   hunyuan (stage 7)   da3 (stage 2)
# Skipped on purpose: any6d (off by default -- `use_any6d` is commented out in
# scripts/cfg/real2sim_cfg.yaml), and void/nerfstudio/3dgrut (auto-background
# only, opt-in via --bg-splat). That cuts the install from 7 envs to 3.
#
# NOTE ON LICENCES: install_simfoundry.sh hardcodes --accept-nvidia-eula and
# --accept-dataset-tos, so running this accepts the NVIDIA Omniverse License
# Agreement and the BEHAVIOR-1K dataset terms on your behalf. Submitting this
# job is your acceptance of both.
#
# Submit with:  sbatch /scratch/pandya.kei/install_simfoundry.sbatch

set -euo pipefail
mkdir -p /scratch/pandya.kei/logs
export PATH=/scratch/pandya.kei/miniforge3/bin:$PATH
export PATH=/shared/EL9/explorer/FFmpeg/7.1.1/bin:$PATH
export LD_LIBRARY_PATH=/shared/EL9/explorer/FFmpeg/7.1.1/lib:${LD_LIBRARY_PATH:-}

# Reuse the pre-warmed model cache on /scratch rather than re-downloading into
# $HOME (a separate, quota'd mount). Already holds Hunyuan3D-2.1 + DINOv3.
export HF_HOME=/scratch/pandya.kei/hf_cache

eval "$(mamba shell hook --shell bash)"

REPO=/scratch/pandya.kei/repos/SimFoundry
INST=$REPO/scripts/installation

# Build fat binaries so these envs stay usable across cards: a100 (8.0, the
# build host), a6000/a5000 (8.6), l40s/l40 (8.9), h100/h200 (9.0).
# install_everything.sh does NOT forward --cuda-arch-list, so each installer is
# invoked directly below -- otherwise they auto-detect only the build node's arch.
ARCHES="8.0;8.6;8.9;9.0"

# --- pre-flight ------------------------------------------------------------
# Guards against a GPU cu128 torch has no kernels for.
#
# DO NOT try to build this on a CPU node. Tested 2026-10-02 and it cannot work:
# install_simfoundry.sh:649 and install_da3.sh:152 both call install_faiss_gpu,
# whose validate_faiss_gpu (faiss_gpu.sh:36) runs faiss.get_num_gpus() and
# raises SystemExit when it sees 0 GPUs. A live GPU is mandatory at INSTALL time.
# The GPU is only needed to satisfy that check -- nothing here trains, and
# TORCH_CUDA_ARCH_LIST is explicit -- so any sm_75+ card will do, including a t4.
# SIMFOUNDRY_SKIP_GPU_CHECK=1 only bypasses THIS block, not the faiss check.
if [[ "${SIMFOUNDRY_SKIP_GPU_CHECK:-0}" == "1" ]]; then
    echo "Pre-flight skipped (SIMFOUNDRY_SKIP_GPU_CHECK=1); building for $ARCHES regardless."
    # A CPU build must also skip faiss's live-device probe, or install_simfoundry.sh
    # (line 649) and install_da3.sh (line 152) abort at install_faiss_gpu.
    # faiss-gpu still gets INSTALLED and its GPU symbols verified; only the
    # device probe is deferred. Finish it with validate_faiss_gpu on a GPU node:
    #     sbatch /scratch/pandya.kei/validate_faiss.sbatch
    export SIMFOUNDRY_SKIP_FAISS_GPU_VALIDATE=1
    echo "  -> SIMFOUNDRY_SKIP_FAISS_GPU_VALIDATE=1 (faiss device probe deferred)"
elif ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "FATAL: no nvidia-smi. Submit to a GPU partition, or set" >&2
    echo "       SIMFOUNDRY_SKIP_GPU_CHECK=1 to build on a CPU node." >&2
    exit 1
else
    CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  compute_cap=$CAP"
    if awk "BEGIN{exit !($CAP < 7.5)}"; then
        echo "FATAL: compute capability $CAP < 7.5. SimFoundry's cu128 torch has no" >&2
        echo "       kernels for this GPU. Use gpu/a100 or sharing/l40s." >&2
        exit 1
    fi
    # RT cores are needed by Isaac Sim at RUN time, not to build. a100/h100/h200
    # have none -- fine here, but stages 12-14 must then move to sharing/l40s.
    case "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" in
        *A100*|*H100*|*H200*) echo "NOTE: datacenter card, no RT cores. Build is fine;"
                              echo "      run stages 12-14 / Pipeline C on sharing/l40s." ;;
    esac
fi

echo "TORCH_CUDA_ARCH_LIST=$ARCHES"   # defined above, before the pre-flight

# --- BEHAVIOR-1K: pre-clone + glibc patch ----------------------------------
# Explorer is Rocky 9.7 => glibc 2.34, but BEHAVIOR-1K pins Isaac Sim 5.1 whose
# wheels are tagged manylinux_2_35, so pip refuses them:
#   "omniverse_kit-107.3.1...manylinux_2_35_x86_64.whl is not a supported wheel"
# That tag is mislabeled: objdump over all 109 .so files in omniverse_kit shows
# the highest symbol required is GLIBC_2.34 -- exactly what Rocky 9.7 provides.
# BEHAVIOR-1K ships a retag workaround but it is doubly broken:
#   * check_glibc_old matches only 2.31|32|33, missing 2.34
#   * the rename substitutes manylinux_2_34, but the file is named _2_35, so it
#     is a silent no-op even on the glibc versions it does match.
# Clone + pin here (so install_simfoundry.sh's `if [ ! -d BEHAVIOR-1K ]` guard
# skips its own clone and git_safe_checkout_detached early-returns on an
# already-correct HEAD), then patch idempotently.
B1K="$REPO/deps/BEHAVIOR-1K"
B1K_COMMIT=d89aae4e0e9a1de3cf8285cb9669c11d8c8bb864
if [[ ! -d "$B1K/.git" ]]; then
    echo ">>> cloning BEHAVIOR-1K and pinning $B1K_COMMIT"
    rm -rf "$B1K"
    git clone https://github.com/StanfordVL/BEHAVIOR-1K.git "$B1K"
    git -C "$B1K" checkout --detach "$B1K_COMMIT"
fi
python3 - "$B1K/setup.sh" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
before = s
s = s.replace('grep -qE "2\\.(31|32|33)"', 'grep -qE "2\\.(31|32|33|34)"')
s = s.replace('${filepath/manylinux_2_34/manylinux_2_31}',
              '${filepath/manylinux_2_35/manylinux_2_31}')
# OmniGibson 3.8.0 imports lerobot unconditionally (envs/data_wrapper.py:28) but
# declares it only in the `eval` extra (setup.py:85), so setup.sh's
# `python -c "import omnigibson"` gate can never pass. Pre-install it.
anchor = '    python -c "import omnigibson" || {'
if 'import lerobot' not in s and anchor in s:
    s = s.replace(anchor,
        '    python -c "import lerobot" 2>/dev/null || \\\n'
        '      pip install "lerobot @ git+https://github.com/huggingface/lerobot@577cd10974b84bea1f06b6472eb9e5e74e07f77a"\n\n'
        + anchor, 1)
if s != before:
    open(p, 'w').write(s)
    print(">>> patched BEHAVIOR-1K setup.sh for glibc 2.34")
else:
    print(">>> BEHAVIOR-1K setup.sh already patched (or upstream changed -- CHECK)")
PY

for spec in "install_simfoundry.sh simfoundry" "install_hunyuan.sh hunyuan" "install_da3.sh da3"; do
    script="${spec%% *}"; env="${spec##* }"
    prefix="/scratch/pandya.kei/conda_envs/$env"
    marker="$prefix/.simfoundry_install_complete"
    # Only a marker written AFTER the installer exits 0 proves completeness.
    # Two weaker tests were tried and both silently skip a broken env:
    #   `mamba run -n "$env" true` passed on a 15 MB prefix with no python;
    #   `bin/python -c "import sys"` passed on the 15 GB env left by job
    #   10769493, which died before OmniGibson was ever installed.
    # A skip here produces an install that reports success and cannot run.
    if [[ -f "$marker" ]]; then
        echo ">>> env '$env' already complete -- skipping (rm $marker to rebuild)"
        continue
    fi
    if [[ -e "$prefix" ]]; then
        echo ">>> env '$env' exists but is INCOMPLETE (no marker) -- removing before rebuild"
        rm -rf "$prefix"
    fi
    # da3 rebuilds xformers FROM SOURCE at install_da3.sh:113 -- redundant, since
    # line 109 already installs the xformers==0.0.30 cu128 wheel and the pinned SHA
    # 4cf69f09 IS refs/tags/v0.0.30 (verified). We keep the build (it links against
    # the local torch) but drop 9.0, because xformers/setup.py enables the
    # FlashAttention-3 *Hopper* instantiations when the arch list contains 9.0 --
    # hundreds of .cu files, hours of nvcc. Costs us sm_90 (H100/H200) support in
    # da3's xformers only; our targets are l40s/a6000 (8.9/8.6) and a100 (8.0).
    env_arches="$ARCHES"
    [[ "$env" == "da3" ]] && env_arches="8.0;8.6;8.9"
    echo ""; echo ">>> [$env] $script  (arches: $env_arches)"
    if bash "$INST/$script" --project-root "$REPO" --env-name "$env" \
            --cuda-arch-list "$env_arches" --default; then
        date -Iseconds > "$marker"
        echo ">>> env '$env' COMPLETE"
    else
        echo "FATAL: installer $script failed for env '$env'; leaving it unmarked." >&2
        exit 1
    fi
done

echo ""; echo ">>> envs built:"; mamba env list
echo ""
echo "NEXT: log in to services, then fetch checkpoints:"
echo "  bash $INST/login_services.sh          # hf auth login + gcloud/Gemini"
echo "  bash $INST/download_checkpoints.sh --default"
