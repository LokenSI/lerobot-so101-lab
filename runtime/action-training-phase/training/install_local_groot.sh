#!/usr/bin/env bash
set -euo pipefail
# Separate ignored local inference environment. Never allocates a GPU or loads weights.
PACKAGE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$PACKAGE/../../.." && pwd)
export TRAIN_WORK_ROOT="$ROOT/runtime/action-training-phase/local-groot"
export UV_CACHE_DIR="$TRAIN_WORK_ROOT/uv-cache"
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$TRAIN_WORK_ROOT"
bash "$PACKAGE/install_cloud.sh" groot
PY="$TRAIN_WORK_ROOT/sources/Isaac-GR00T/.venv/bin/python"
# Physics/video dependencies for the existing observations-only office evaluator.
uv pip install --python "$PY" 'mujoco==3.6.0' 'pillow==12.3.0' 'imageio==2.38.0' 'imageio-ffmpeg==0.6.0'
uv pip freeze --python "$PY" > "$TRAIN_WORK_ROOT/environment-inference.txt"
"$PY" - <<'PY'
import importlib.metadata,json,os,torch
assert os.environ['CUDA_VISIBLE_DEVICES']==''
assert not torch.cuda.is_available()
import flash_attn,torchcodec
from gr00t.policy.gr00t_policy import Gr00tPolicy
print(json.dumps({'scope':'Imports only, CUDA hidden, no model weights loaded',
                 'torch':torch.__version__,'cuda_build':torch.version.cuda,
                 'flash_attn':importlib.metadata.version('flash-attn'),
                 'torchcodec':importlib.metadata.version('torchcodec')}))
PY
