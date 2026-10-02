"""Prepare pinned local FLUX sources/models; default prints the plan only.

Run inside Linux/WSL Python 3.12. Nothing is downloaded or installed unless its
explicit flag is supplied. Model and encoder binaries remain in ignored models/.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "e2dd1d8dbc5977b54315d61f7548c63c043d6d4f"
MODEL = "2d6be04357eef9fb8c8468adbc69e9d0c7093bec"
ENCODERS = "62878e2925e59b7a89ec14463ce89932624c490d"
WEIGHTS = "0f57664573dde660590060d81b2a0b336fc674f06e0494de9307e52f43b6fd01"


def run(args, cwd=None):
    subprocess.run([str(x) for x in args], cwd=cwd, check=True)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-source", action="store_true")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--check", action="store_true", help="Hash downloaded SO101 files, including 13.9 GB weights")
    args = parser.parse_args()
    source = ROOT/"runtime/flux-action-src"
    python = source/".venv/bin/python"
    plan = {"source": SOURCE, "so101": MODEL, "shared_encoders": ENCODERS,
            "platform": "WSL2 Ubuntu 24.04, Linux x86_64 Python 3.12, CUDA 12.8 Torch",
            "source_directory": str(source), "model_directory": str(ROOT/"models/flux-action"),
            "environment": "upstream uv.lock; original BF16 SO101, not DROID FP8",
            "simulator": "Run existing scripts/bootstrap_runtime.py separately; simulator application is downloaded, not vendored",
            "original_neural_weights_sha256": WEIGHTS}
    print(json.dumps(plan, indent=2))
    if not any(vars(args).values()):
        return
    if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3,12):
        raise SystemExit("Installation/download flags require Linux x86_64 Python 3.12 (WSL2 supported).")
    if args.fetch_source:
        source.parent.mkdir(parents=True, exist_ok=True)
        existed = source.exists()
        if not source.exists():
            run(["git", "clone", "--no-checkout", "https://github.com/black-forest-labs/flux-action.git", source])
        current = subprocess.run(["git", "status", "--porcelain"], cwd=source, check=True, capture_output=True, text=True).stdout if existed else ""
        if current.strip():
            raise SystemExit("Refusing to replace a modified local upstream checkout.")
        run(["git", "fetch", "origin", SOURCE], cwd=source)
        run(["git", "checkout", "--detach", SOURCE], cwd=source)
    if args.install:
        if shutil.which("uv") is None:
            raise SystemExit("Install uv 0.12.9, then rerun. The bootstrap does not install package managers.")
        if not (source/"uv.lock").exists():
            raise SystemExit("Fetch the pinned source first.")
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True).stdout.strip()
        if revision != SOURCE:
            raise SystemExit("Upstream source revision differs from the measured revision.")
        lock = json.loads((ROOT/"experiment/flux-so101-retry/source-lock.json").read_text())
        if digest(source/"uv.lock") != lock["upstream_uv_lock_sha256"]:
            raise SystemExit("Upstream dependency lock differs from the measured lock.")
        run(["uv", "sync", "--locked", "--extra", "encoders", "--extra", "data"], cwd=source)
        run(["uv", "pip", "install", "--python", python, "--no-deps",
             "natten==0.21.6+torch2100cu128", "-f", "https://whl.natten.org/"], cwd=source)
        # Simulator/compositor dependencies were added independently to the locked
        # inference environment; retain the measured versions without re-resolving Torch.
        run(["uv", "pip", "install", "--python", python, "--no-deps",
             "mujoco==3.6.0", "imageio==2.37.3", "imageio-ffmpeg==0.6.0",
             "pillow==12.3.0", "pyopengl==3.1.10", "glfw==2.10.2", "absl-py==2.5.0", "etils==1.14.0"], cwd=source)
    if args.download_models:
        if not python.exists():
            raise SystemExit("Install the pinned inference environment first.")
        downloader = """import sys
from huggingface_hub import snapshot_download
snapshot_download('black-forest-labs/flux-3-action-so101', revision=sys.argv[1], local_dir=sys.argv[3], allow_patterns=['*.json','*.md','*.safetensors'])
snapshot_download('black-forest-labs/flux-3-action-base', revision=sys.argv[2], local_dir=sys.argv[4], allow_patterns=['video_vae.safetensors','text_encoder/*'])
"""
        run([python, "-c", downloader, MODEL, ENCODERS,
             ROOT/"models/flux-action/so101", ROOT/"models/flux-action/base"])
    if args.check:
        lock = json.loads((ROOT/"experiment/flux-so101-retry/source-lock.json").read_text())
        for name, expected in lock["so101_files_sha256"].items():
            actual = digest(ROOT/"models/flux-action/so101"/name)
            if actual != expected:
                raise SystemExit(f"SHA256 mismatch: {name}")
        print("All pinned SO101 package hashes match.")


if __name__ == "__main__":
    main()
