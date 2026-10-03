#!/usr/bin/env bash
set -euo pipefail
# Run on an authorized Linux Brev VM, not on the local desktop.
MODE=${1:?smolvla or groot}
PACKAGE=$(cd "$(dirname "$0")" && pwd)
WORK=${TRAIN_WORK_ROOT:-"$HOME/office-action-training"}
command -v uv >/dev/null || { echo 'Install uv using its official instructions first.' >&2; exit 2; }
command -v ffmpeg >/dev/null || { echo 'ffmpeg must be installed before video dataset training.' >&2; exit 2; }
mkdir -p "$WORK/sources"
case "$MODE" in
  smolvla) REPO=https://github.com/huggingface/lerobot.git; PIN=6e1fa4faf2a42927d463591aebaa0f62c2e654b7; NAME=lerobot ;;
  groot) REPO=https://github.com/NVIDIA/Isaac-GR00T.git; PIN=51d4c89f72fda44cbf77285c6a8114b52676b8a1; NAME=Isaac-GR00T ;;
  *) exit 2 ;;
esac
SRC="$WORK/sources/$NAME"
if [[ ! -d "$SRC/.git" ]]; then git init "$SRC"; git -C "$SRC" remote add origin "$REPO"; fi
git -C "$SRC" fetch --depth 1 origin "$PIN"
git -C "$SRC" checkout --detach "$PIN"
if [[ "$MODE" == smolvla ]]; then
  uv sync --project "$SRC" --locked --python 3.12 --extra smolvla --extra training
else
  command -v git-lfs >/dev/null || { echo 'Install git-lfs before GR00T setup.' >&2; exit 2; }
  # uv resolves a local aarch64 wheel even on x86_64. A shallow Git checkout
  # contains its LFS pointer; hydrate the pinned wheel before locked sync.
  WHEEL=scripts/deployment/dgpu/wheels/torchcodec-0.8.0-cp312-cp312-linux_aarch64.whl
  git -C "$SRC" lfs install --local
  git -C "$SRC" lfs fetch origin "$PIN" --include="$WHEEL" --exclude=""
  git -C "$SRC" lfs checkout "$WHEEL"
  uv run --no-project --python 3.12 python -c 'import sys,zipfile; assert zipfile.is_zipfile(sys.argv[1]), "GR00T local wheel is still an LFS pointer"' "$SRC/$WHEEL"
  uv sync --project "$SRC" --locked --python 3.12
fi
"$SRC/.venv/bin/python" -m pip --version >/dev/null 2>&1 || true
uv pip freeze --python "$SRC/.venv/bin/python" > "$WORK/environment-$MODE.txt"
"$SRC/.venv/bin/python" -c 'import torch,json; print(json.dumps({"torch":torch.__version__,"cuda_build":torch.version.cuda,"cuda_available":torch.cuda.is_available()}))'
printf 'Source: %s\nPython: %s\n' "$SRC" "$SRC/.venv/bin/python"
