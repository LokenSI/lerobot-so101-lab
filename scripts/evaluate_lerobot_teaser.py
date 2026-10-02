"""Closed-loop evaluation of a local official LeRobot ACT checkpoint in MuJoCo.

Only scene/wrist RGB and the six robot joints enter the learned policy. Object
coordinates and contacts are available solely to the independent task grader.
No commands are sent to physical hardware.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/flux-action-so101-sim"))
import sim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/lerobot-teaser/train")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/lerobot-teaser/eval")
    parser.add_argument("--environment-module", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    parser.add_argument("--max-steps", type=int, default=525)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-units", choices=("radians", "degrees"), default="radians")
    parser.add_argument("--action-steps", type=int, help="Override execution horizon for an explicit development experiment")
    args = parser.parse_args()
    checkpoint, out = args.checkpoint.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    train_report = json.loads((checkpoint / "training-report.json").read_text())
    demonstration_seeds = []
    for key in ("train_episodes", "validation_episodes"):
        for row in train_report[key]:
            match = re.search(r"seed[-_](\d+)", row["file"])
            assert match is not None, f"No demonstration seed recorded in {row['file']}"
            demonstration_seeds.append(int(match.group(1)))
    assert not set(args.seeds).intersection(demonstration_seeds), "Evaluation seeds must be disjoint from all demonstrations"
    environment = importlib.import_module(args.environment_module)
    assert callable(environment.make_env) and callable(environment.task_success)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = ACTConfig.from_pretrained(checkpoint)
    if args.action_steps is not None:
        assert 1 <= args.action_steps <= config.chunk_size
        config.n_action_steps = args.action_steps
    config.device = args.device
    policy = ACTPolicy.from_pretrained(checkpoint, config=config).to(args.device).eval()
    stats = json.loads((checkpoint / "normalization.json").read_text())
    stats = {key: {name: np.asarray(value, dtype=np.float32) for name, value in row.items()} for key, row in stats.items()}
    image_size = next(iter(config.image_features.values())).shape[-1]
    gpu_samples = []
    stop_event = threading.Event()
    def monitor() -> None:
        while not stop_event.is_set():
            try:
                result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True, timeout=2)
                gpu_samples.append(int(result.splitlines()[0].strip()))
            except (subprocess.SubprocessError, ValueError):
                pass
            stop_event.wait(0.5)
    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    report = {
        "policy": "Official LeRobot ACT, compact deterministic configuration; VAE disabled",
        "checkpoint": str(checkpoint), "parameters": sum(p.numel() for p in policy.parameters()),
        "model_file_bytes": (checkpoint / "model.safetensors").stat().st_size,
        "environment_module": args.environment_module,
        "environment_script_sha256": hashlib.sha256(Path(environment.__file__).read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest(),
        "observation": "Scene RGB, wrist RGB, six measured joint values. No object coordinates, expert waypoints or phase counters.",
        "action_units": args.action_units,
        "grader": "Independent environment.task_success(env); model cannot access grading state",
        "seeds": args.seeds, "max_steps": args.max_steps,
        "demonstration_seeds": sorted(set(demonstration_seeds)),
        "architecture": {name: getattr(config, name) for name in ("dim_model", "n_heads", "dim_feedforward", "n_encoder_layers", "n_decoder_layers", "chunk_size", "n_action_steps", "use_vae", "vision_backbone")},
        "device": args.device, "torch": torch.__version__, "episodes": [], "complete": False,
        "camera_shape": [image_size, image_size, 3],
        "orin_tested": False,
        "generalization_scope": "Same simulator, fixed task and cameras, separate evaluation seeds; hardware and offshore conditions untested",
    }
    all_times, fresh_times = [], []
    try:
        for seed in args.seeds:
            env = environment.make_env(seed, width=image_size, height=image_size)
            video_renderer = mujoco.Renderer(env.model, 240, 320)
            policy.reset()
            writer = imageio.get_writer(str(out / f"seed-{seed}.mp4"), fps=sim.FPS, codec="libx264", quality=8, macro_block_size=16)
            states, actions, positions, times, qpos, qvel = [], [], [], [], [], []
            clipping_events, maximum_target_limit_excess = 0, 0.0
            started = time.perf_counter()
            try:
                for step in range(args.max_steps):
                    state = env.data.qpos[env.qadr].astype(np.float32).copy() if args.action_units == "radians" else env.state_deg().astype(np.float32)
                    batch = {"observation.state": torch.from_numpy((state - stats["states"]["mean"]) / stats["states"]["std"]).unsqueeze(0).to(args.device)}
                    for name in ("scene", "wrist"):
                        frame = np.asarray(Image.fromarray(env.render(name)).resize((image_size, image_size), Image.Resampling.BILINEAR))
                        normalized = (frame.transpose(2, 0, 1).astype(np.float32) / 255.0 - 0.5) / 0.5
                        batch[f"observation.images.{name}"] = torch.from_numpy(normalized).unsqueeze(0).to(args.device)
                    fresh = len(policy._action_queue) == 0
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    start_inference = time.perf_counter()
                    with torch.inference_mode(), torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                        normalized_action = policy.select_action(batch)
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    latency = time.perf_counter() - start_inference
                    times.append(latency)
                    all_times.append(latency)
                    if fresh:
                        fresh_times.append(latency)
                    action = normalized_action.float().cpu().numpy()[0] * stats["actions"]["std"] + stats["actions"]["mean"]
                    assert action.shape == (6,) and np.isfinite(action).all()
                    q_target = action if args.action_units == "radians" else sim.deg_to_q(action)
                    lo, hi = env.model.actuator_ctrlrange.T
                    excess = np.maximum(np.maximum(lo - q_target, q_target - hi), 0)
                    clipping_events += int(np.any(excess > 1e-6))
                    maximum_target_limit_excess = max(maximum_target_limit_excess, float(excess.max()))
                    states.append(state)
                    actions.append(action)
                    env.step(sim.q_to_deg(action) if args.action_units == "radians" else action)
                    qpos.append(env.data.qpos.copy())
                    qvel.append(env.data.qvel.copy())
                    positions.append(env.object_positions()["obj0"].copy())
                    video_renderer.update_scene(env.data, camera="overview")
                    frame = Image.fromarray(video_renderer.render().copy())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 320, 37), fill=(12, 20, 28))
                    draw.text((5, 3), f"LEARNED ACT | SO-101 SIM | seed {seed}", fill="white")
                    draw.text((5, 20), "RGB + joint state only | no physical robot", fill=(242, 184, 82))
                    writer.append_data(np.asarray(frame))
                result = environment.task_success(env)
                if isinstance(result, bool):
                    result = {"success": result}
                assert "success" in result, result
                row = {"seed": seed, "steps": args.max_steps, "grader": result, "success": bool(result["success"]), "wall_seconds": time.perf_counter() - started, "video": f"seed-{seed}.mp4", "actuator_target_limit_events": clipping_events, "max_target_limit_excess_rad": maximum_target_limit_excess}
                report["episodes"].append(row)
                report["success_count"] = sum(episode["success"] for episode in report["episodes"])
                report["total"] = len(report["episodes"])
                np.savez_compressed(out / f"seed-{seed}-trajectory.npz", states=np.asarray(states), actions=np.asarray(actions), qpos=np.asarray(qpos), qvel=np.asarray(qvel), object_positions_for_grading_only=np.asarray(positions), inference_seconds=np.asarray(times))
                print(json.dumps(row), flush=True)
                (out / "report.json").write_text(json.dumps(report, indent=2))
            finally:
                writer.close()
                video_renderer.close()
                env.close()
        report["success_count"] = sum(row["success"] for row in report["episodes"])
        report["total"] = len(report["episodes"])
        report["action_selection_seconds"] = {"median": float(np.median(all_times)), "p95": float(np.percentile(all_times, 95))}
        report["fresh_chunk_inference_seconds"] = {"median": float(np.median(fresh_times)), "p95": float(np.percentile(fresh_times, 95)), "count": len(fresh_times)}
        report["whole_device_peak_mib"] = max(gpu_samples) if gpu_samples else None
        report["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None
        report["complete"] = True
    finally:
        stop_event.set()
        worker.join(timeout=3)
        (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
