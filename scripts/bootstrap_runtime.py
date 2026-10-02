"""Fetch pinned official source and optionally install the measured environments.

No runtime source, meshes, model weights, Windows fonts, or virtualenv is bundled
in this repository. Run with Python3.12 on Linux/WSL. --check is read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(folder, files):
    for name, expected in files.items():
        path = folder / name
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"Source missing or differs from measured revision: {path}")


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "lerobot-so101-lab-bootstrap"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as target:
        shutil.copyfileobj(response, target)


def sources(root, lock):
    runtime = root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    lerobot = lock["lerobot"]
    folder = root / lerobot["runtime_directory"]
    if folder.exists():
        verify(folder, lerobot["checked_files"])
    else:
        # Temporary extraction stays inside the requested runtime directory.
        with tempfile.TemporaryDirectory(prefix="lerobot-source-", dir=runtime) as temporary:
            temporary = Path(temporary)
            archive = temporary / "source.tar.gz"
            download(lerobot["archive_url"], archive)
            unpack = temporary / "unpack"
            unpack.mkdir()
            with tarfile.open(archive, "r:gz") as bundle:
                bundle.extractall(unpack, filter="data")
            candidates = list(unpack.iterdir())
            if len(candidates) != 1 or not candidates[0].is_dir():
                raise RuntimeError("Unexpected official LeRobot archive structure")
            verify(candidates[0], lerobot["checked_files"])
            shutil.move(str(candidates[0]), str(folder))
    space = lock["so101_space"]
    folder = root / space["runtime_directory"]
    for name, expected in space["files"].items():
        path = folder / name
        if path.exists():
            verify(folder, {name: expected})
            continue
        url = f"{space['repository']}/resolve/{space['revision']}/{urllib.parse.quote(name, safe='/')}"
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="source-download-", dir=path.parent) as temporary:
            fetched = Path(temporary) / "payload"
            download(url, fetched)
            if sha(fetched) != expected:
                raise RuntimeError(f"Official source checksum mismatch: {name}")
            shutil.move(str(fetched), str(path))
    verify(folder, space["files"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PACKAGE, help="Target checkout root; also useful for checking original source")
    parser.add_argument("--check", action="store_true", help="Verify expected local sources without downloading or installing")
    parser.add_argument("--install", action="store_true", help="Create Linux x86_64 simulation and desktop policy venvs after downloading sources")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Use Python3.12, matching the measured experiment")
    if args.check and args.install:
        parser.error("--check and --install are mutually exclusive")
    root = args.root.resolve()
    lock = json.loads((PACKAGE / "experiment/upstream-source-lock.json").read_text())
    if args.check:
        verify(root / lock["lerobot"]["runtime_directory"], lock["lerobot"]["checked_files"])
        verify(root / lock["so101_space"]["runtime_directory"], lock["so101_space"]["files"])
    else:
        sources(root, lock)
    if args.install:
        if sys.platform != "linux" or platform.machine() not in ("x86_64", "AMD64") or root != PACKAGE:
            parser.error("Install from this repository on Linux/WSL; desktop lock is for x86_64 CUDA12.8")
        for folder, requirement in [("flux-action-sim-env", "requirements-simulation.txt"),
                                    ("lerobot-teaser-env", "requirements-policy.txt")]:
            env = root / "runtime" / folder
            if not env.exists():
                subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
            subprocess.run([str(env / "bin/python"), "-m", "pip", "install", "-r", str(root / requirement)], cwd=root, check=True)
    print(json.dumps({"sources_verified": True, "lerobot_revision": lock["lerobot"]["revision"],
                      "so101_revision": lock["so101_space"]["revision"], "root": str(root),
                      "environments_installed": args.install}, indent=2))


if __name__ == "__main__":
    main()
