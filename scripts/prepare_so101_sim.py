"""Render a reproducible SO-101 camera/control calibration in real MuJoCo physics.

This deliberately uses scripted joint targets, not FLUX 3 Action.  It establishes
that the simulator, camera images and unit conversion work before policy testing.
Run under WSL with MUJOCO_GL=egl PYOPENGL_PLATFORM=egl and
runtime/flux-action-sim-env/bin/python scripts/prepare_so101_sim.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime" / "flux-action-so101-sim"
SPACE_REVISION = "d7da38e032da0e136d6de21c218dc616982a8cc9"
SOURCE_URL = "https://huggingface.co/spaces/multimodalart/flux-3-action-so101-sim"
sys.path.insert(0, str(SOURCE))
import sim  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_camera(frame: np.ndarray, name: str) -> dict:
    assert frame.ndim == 3 and frame.shape[2] == 3, (name, frame.shape)
    assert frame.dtype == np.uint8, (name, frame.dtype)
    assert np.isfinite(frame).all(), name
    stats = {
        "shape": list(frame.shape),
        "dtype": str(frame.dtype),
        "minimum": int(frame.min()),
        "maximum": int(frame.max()),
        "standard_deviation": float(frame.std()),
        "sha256_pixels": hashlib.sha256(frame.tobytes()).hexdigest(),
    }
    assert stats["standard_deviation"] > 1, f"Camera {name} is blank"
    return stats


def save_snapshot(env: sim.Sim, out: Path, prefix: str) -> dict:
    frames = {name: env.render(name) for name in ("scene", "wrist", "overview")}
    stats = {}
    for name, frame in frames.items():
        stats[name] = check_camera(frame, name)
        Image.fromarray(frame).save(out / f"{prefix}-{name}.png")
    state = env.state_deg().astype(np.float32)
    assert state.shape == (6,) and np.isfinite(state).all()
    np.savez_compressed(
        out / f"{prefix}-observation.npz",
        **{"images.scene": frames["scene"], "images.wrist": frames["wrist"], "state": state},
    )
    return {"cameras": stats, "state_degrees_gripper_percent": state.tolist()}


def make_contact_sheet(out: Path) -> None:
    canvas = Image.new("RGB", (1024, 680), (18, 25, 36))
    draw = ImageDraw.Draw(canvas)
    fonts = Path("C:/Windows/Fonts") if sys.platform == "win32" else Path("/mnt/c/Windows/Fonts")
    title_font = ImageFont.truetype(str(fonts / "segoeuib.ttf"), 30)
    body_font = ImageFont.truetype(str(fonts / "segoeui.ttf"), 21)
    draw.text((28, 20), "SO-101: virtual cameras and physics", font=title_font, fill="white")
    draw.text((28, 65), "Scripted calibration | MuJoCo | FLUX policy not connected", font=body_font, fill=(253, 196, 105))
    for index, name in enumerate(("scene", "wrist", "overview")):
        tile = Image.open(out / f"initial-{name}.png").convert("RGB").resize((320, 240))
        x = 20 + index * 337
        canvas.paste(tile, (x, 135))
        draw.text((x + 4, 105), name.capitalize() + " camera", font=body_font, fill="white")
        final = Image.open(out / f"final-{name}.png").convert("RGB").resize((320, 240))
        canvas.paste(final, (x, 415))
    draw.text((24, 380), "Final calibration pose (targets supplied by script)", font=body_font, fill="white")
    canvas.save(out / "camera-validation.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/flux-action-study/simulation")
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()
    assert 0 < args.seconds <= 30
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = sim.Sim(scene="Single cube", width=640, height=480, seed=0)
    initial = save_snapshot(env, out, "initial")
    # Keep the object untouched: exercise pan/roll and a small shoulder lift only.
    # These targets show unit conventions and physical actuator tracking.
    rest = np.asarray(sim.REST_DEG, dtype=np.float64)
    records = []
    video_path = out / "so101-scripted-calibration.mp4"
    writer = imageio.get_writer(str(video_path), fps=sim.FPS, codec="libx264", quality=8, macro_block_size=16)
    object_start = {name: value.tolist() for name, value in env.object_positions().items()}
    try:
        for tick in range(round(args.seconds * sim.FPS)):
            t = (tick + 1) / sim.FPS
            phase = 2 * math.pi * t / args.seconds
            command = rest.copy()
            command[0] += 20 * math.sin(phase)
            command[1] += 5 * (1 - math.cos(phase)) / 2
            command[4] += 15 * math.sin(phase)
            assert np.isfinite(command).all()
            env.step(command)
            state = env.state_deg()
            q = env.data.qpos[env.qadr].copy()
            lo, hi = env.model.jnt_range[:6].T
            assert np.isfinite(state).all() and np.isfinite(q).all()
            # MuJoCo enforces soft joint limits; allow small physical penetration.
            assert np.all(q >= lo - 0.05) and np.all(q <= hi + 0.05), q
            frame = env.render("overview")
            check_camera(frame, "overview")
            annotated = Image.fromarray(frame)
            draw = ImageDraw.Draw(annotated)
            draw.rectangle((0, 0, 640, 51), fill=(8, 14, 23))
            draw.text((12, 8), "SO-101 | MuJoCo physics | Scripted calibration", fill="white")
            draw.text((12, 29), f"FLUX policy not connected | virtual time {t:.2f} s", fill=(255, 199, 111))
            writer.append_data(np.asarray(annotated))
            records.append({
                "tick": tick + 1,
                "simulation_time_seconds": float(env.data.time),
                "control_time_seconds": t,
                "command_degrees_gripper_percent": command.tolist(),
                "observed_degrees_gripper_percent": state.tolist(),
                "physical_joint_radians": q.tolist(),
                "contacts": int(env.data.ncon),
            })
        final = save_snapshot(env, out, "final")
        object_end = {name: value.tolist() for name, value in env.object_positions().items()}
    finally:
        writer.close()
        env.close()
    states = np.asarray([item["observed_degrees_gripper_percent"] for item in records])
    telemetry_path = out / "calibration-telemetry.json"
    telemetry_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    make_contact_sheet(out)
    source_files = [SOURCE / "sim.py", SOURCE / "so101.xml", *sorted((SOURCE / "assets").glob("*.stl"))]
    report = {
        "status": "passed",
        "scope": "Simulator and virtual-camera calibration only; no learned policy evaluation",
        "control_source": "scripted sinusoidal joint targets",
        "flux_policy_loaded": False,
        "task_success_evaluated": False,
        "simulation_engine": "MuJoCo",
        "mujoco_version": mujoco.__version__,
        "scene": "Single cube",
        "seed": 0,
        "frames": len(records),
        "control_frequency_hz": sim.FPS,
        "physics_timestep_seconds": sim.PHYSICS_DT,
        "physics_substeps_per_control": sim.SUBSTEPS,
        "clip_seconds": len(records) / sim.FPS,
        "wall_seconds": time.perf_counter() - started,
        "joint_names": sim.JOINTS,
        "state_units": ["degree"] * 5 + ["gripper_percentage_point"],
        "observed_joint_peak_to_peak": np.ptp(states, axis=0).tolist(),
        "initial_snapshot": initial,
        "final_snapshot": final,
        "object_initial_positions_m": object_start,
        "object_final_positions_m": object_end,
        "source": {
            "url": SOURCE_URL,
            "revision": SPACE_REVISION,
            "robot_mesh_origin": "https://github.com/TheRobotStudio/SO-ARM100",
            "robot_mesh_license": "Apache-2.0 (per Space README)",
            "file_sha256": {str(path.relative_to(SOURCE)): sha256(path) for path in source_files},
        },
        "artifacts": {
            "video": video_path.name,
            "camera_contact_sheet": "camera-validation.png",
            "telemetry": telemetry_path.name,
            "initial_observation": "initial-observation.npz",
        },
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "frames", "clip_seconds", "wall_seconds", "flux_policy_loaded")}, indent=2))
    print(str(out))


if __name__ == "__main__":
    main()
