#!/usr/bin/env bash
# Post-install sanity check. The install can report success while leaving an
# env that cannot import. Specifically flagged during job 10769493:
#   opencv-python 5.0.0.93 and cmeel-boost require numpy>=2, but the env pins
#   numpy 1.26.4 (OmniGibson needs numpy<2). OpenCV 5.x is compiled against the
#   numpy-2 ABI, so `import cv2` may raise even though pip exited 0.
# Fix if cv2 fails:  pip install "opencv-python<4.11" "opencv-contrib-python<4.11" \
#                                 "opencv-python-headless<4.11"
export PATH=/scratch/pandya.kei/miniforge3/bin:$PATH
export HF_HOME=/scratch/pandya.kei/hf_cache
P=/scratch/pandya.kei/conda_envs

check() {
  local env=$1; shift
  local py="$P/$env/bin/python"
  echo "=== $env ==="
  if [[ ! -x "$py" ]]; then echo "  MISSING env"; return; fi
  for mod in "$@"; do
    out=$("$py" -c "
import importlib,sys
m=importlib.import_module('$mod')
v=getattr(m,'__version__','?')
print('ok',v)
" 2>&1 | tail -1)
    case "$out" in
      ok*) printf '  %-18s %s\n' "$mod" "$out" ;;
      *)   printf '  %-18s FAIL: %s\n' "$mod" "$(echo "$out" | tail -1 | cut -c1-110)" ;;
    esac
  done
}

check simfoundry numpy cv2 torch scipy trimesh omnigibson sam3
check hunyuan    numpy torch
check da3        numpy torch

echo
echo "=== numpy/cv2 ABI pairing in simfoundry ==="
"$P/simfoundry/bin/python" - <<'PY' 2>&1 | tail -5
try:
    import numpy, cv2
    print("numpy", numpy.__version__, "+ cv2", cv2.__version__, "-> OK")
except Exception as e:
    print("ABI PROBLEM:", type(e).__name__, str(e)[:200])
PY

echo
echo "=== torch sees the GPU? (only meaningful on a GPU node) ==="
"$P/simfoundry/bin/python" -c "
import torch
print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available())
print('arch_list', torch.cuda.get_arch_list())
" 2>&1 | tail -3
