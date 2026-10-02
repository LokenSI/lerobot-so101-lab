"""Package a checkpoint and saved observations for later Orin inference tests."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/robotics-teasers/orin-replay-pack.zip")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    observation = args.output.parent / "orin-replay-observation.npz"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = ROOT / "runs/so101-teaser-dataset/seed-000/episode.npz"
    with np.load(source) as data:
        np.savez_compressed(observation, **{key: data[key][[0]] for key in ("states", "images_front", "images_wrist")})
    files = {f"checkpoint/{name}": checkpoint / name for name in ("model.safetensors", "config.json", "normalization.json", "report.json")}
    files.update({"observation.npz": observation, "benchmark_lerobot_act_replay.py": ROOT / "scripts/benchmark_lerobot_act_replay.py",
                  "LEROBOT-LICENSE": ROOT / "runtime/lerobot-teaser-src/LICENSE"})
    manifest = {"scope": "Portable inference replay; no robot commands, no Orin measurements yet",
                "checkpoint_source": str(checkpoint), "observation_source": str(source), "observation_index": 0,
                "lerobot_revision": "6e1fa4faf2a42927d463591aebaa0f62c2e654b7",
                "files": {name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size} for name, path in files.items()}}
    instructions = """# Later Orin Nano Super inference replay

This pack contains the workstation-trained compact ACT checkpoint and one saved
simulation observation. No physical robot or Orin has been tested. The script
does not connect to motors. A finite inference result is not task success.

Use the Orin's supported JetPack/PyTorch/torchvision environment, NumPy and Pillow,
with official LeRobot source pinned to the revision in manifest.json. Do not copy
the workstation's x86-64 environment onto the Arm64 Jetson. Verify imports and
CUDA availability on the device before running this command from the unpacked
directory:

```sh
python benchmark_lerobot_act_replay.py --checkpoint checkpoint --observation observation.npz --output orin-replay-result.json
```

The benchmark uses FP32, ten warmups and 100 measurements of a fresh action chunk.
Desktop closed-loop evaluation uses BF16 autocast, so precision and timing scopes
must be stated when comparing results. Record JetPack/CUDA/library versions,
power mode, thermals and concurrent workloads with the Orin result. CPU replay
is also available with --device cpu. No model export or TensorRT is claimed.

Model config and normalization belong together. Actual camera calibration,
physical robot control, hardware latency and task success remain separate tests.
Actions are absolute radians for all six simulator actuators, including the
gripper angle. Physical LeRobot calibration may expose degrees and a normalized
gripper opening; validate an explicit conversion and camera-domain adaptation
before considering motor execution. This pack supplies inference replay only.
Direct ACT outputs first need the inverse action mean/std transform from the
saved normalization file. The replay benchmark does not execute those outputs.
See manifest.json for checkpoint provenance and file hashes.
"""
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        archive.writestr("README.md", instructions)
    (args.output.parent / "orin-replay-pack-manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"zip": str(args.output), "bytes": args.output.stat().st_size, "orin_tested": False}))


if __name__ == "__main__":
    main()
