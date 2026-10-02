"""Benchmark portable ACT inference on saved camera/joint observations.

Run this later on the Orin with its supported PyTorch/torchvision environment and
the pinned LeRobot source. It needs no robot connection and issues no motor commands.
No Orin measurements are produced until the script is actually run on that device.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = ACTConfig.from_pretrained(args.checkpoint)
    config.device = args.device
    policy = ACTPolicy.from_pretrained(args.checkpoint, config=config).to(args.device).eval()
    stats = json.loads((args.checkpoint / "normalization.json").read_text())
    size = next(iter(config.image_features.values())).shape[-1]
    with np.load(args.observation) as source:
        state_key = next(key for key in ("states", "state") if key in source)
        state = source[state_key]
        if state.ndim == 2:
            state = state[0]
        state = (state - np.asarray(stats["states"]["mean"])) / np.asarray(stats["states"]["std"])
        batch = {"observation.state": torch.tensor(state, dtype=torch.float32, device=args.device).unsqueeze(0)}
        for name, keys in {"scene": ("images_scene", "images_front", "images.scene"), "wrist": ("images_wrist", "images.wrist")}.items():
            frame = source[next(key for key in keys if key in source)]
            if frame.ndim == 4:
                frame = frame[0]
            frame = np.asarray(Image.fromarray(frame).resize((size, size), Image.Resampling.BILINEAR))
            normalized = (frame.transpose(2, 0, 1).astype(np.float32) / 255 - 0.5) / 0.5
            batch[f"observation.images.{name}"] = torch.tensor(normalized, device=args.device).unsqueeze(0)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(args.iterations + 10):
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            actions = policy.predict_action_chunk(batch)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert actions.shape == (1, config.chunk_size, 6) and torch.isfinite(actions).all()
        if i >= 10:
            times.append(elapsed)
    report = {
        "device_name": torch.cuda.get_device_name() if args.device.startswith("cuda") else "CPU",
        "device": args.device, "torch": torch.__version__, "iterations": args.iterations,
        "precision": "float32", "fresh_chunk_median_seconds": float(np.median(times)),
        "fresh_chunk_p95_seconds": float(np.percentile(times, 95)),
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
        "model_file_bytes": (args.checkpoint / "model.safetensors").stat().st_size,
        "shape": list(actions.shape), "finite": True,
        "measurement_scope": "Inference on a fixed saved observation; no closed-loop success measurement or robot control",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
