"""Render the exact recorded states of one learned-policy evaluation in HD.

This is a presentation replay, not a new control test. Every robot/object pose
comes from the saved closed-loop physics trajectory. Success comes from the
original independent evaluator, never from this renderer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=ROOT / "runs/lerobot-teaser/eval")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/robotics-teasers")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    report = json.loads((args.evaluation / "report.json").read_text())
    rows = report["episodes"]
    selected = next((row for row in rows if row["seed"] == args.seed), None) if args.seed is not None else next((row for row in rows if row["success"]), None)
    if selected is None:
        raise ValueError("No selected successful learned-policy rollout available")
    seed = selected["seed"]
    trajectory = args.evaluation / f"seed-{seed}-trajectory.npz"
    with np.load(trajectory) as source:
        qpos = source["qpos"]
        qvel = source["qvel"]
        positions = source["object_positions_for_grading_only"]
    assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
    env = importlib.import_module(report["environment_module"]).make_env(seed, width=128, height=128)
    env.model.vis.global_.offwidth = 1280
    env.model.vis.global_.offheight = 960
    renderer = mujoco.Renderer(env.model, 960, 1280)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "lerobot-learned-pick-place.mp4"
    font_root = Path("/mnt/c/Windows/Fonts") if sys.platform != "win32" else Path("C:/Windows/Fonts")
    title = ImageFont.truetype(str(font_root / "segoeuib.ttf"), 34)
    label = ImageFont.truetype(str(font_root / "segoeui.ttf"), 25)
    lift_index = int(np.argmax(positions[:, 2]))
    indices = sorted(set([0, lift_index, len(qpos) // 2, len(qpos) - 1]))
    try:
        with imageio.get_writer(str(output), fps=30, codec="libx264", quality=9, macro_block_size=16) as writer:
            for index, (pose, velocity) in enumerate(zip(qpos, qvel)):
                assert pose.shape == env.data.qpos.shape and velocity.shape == env.data.qvel.shape
                env.data.qpos[:] = pose
                env.data.qvel[:] = velocity
                mujoco.mj_forward(env.model, env.data)
                renderer.update_scene(env.data, camera="overview")
                # Presentation-only: remove aliasing artifacts from the default
                # shadow map. Policy observations and recorded physics are unchanged.
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                frame = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 1280, 105), fill=(8, 20, 32))
                draw.text((30, 10), "Learned ACT | SO-101 simulation", font=title, fill="white")
                draw.text((30, 60), "Two virtual cameras + joint feedback | no physical robot tested", font=label, fill=(137, 224, 214))
                draw.rectangle((0, 895, 1280, 960), fill=(8, 20, 32))
                text = f"Held-out seed {seed} | recorded physics rollout | {(index + 1) / 30:.1f} s"
                if index >= len(qpos) - 45 and selected["success"]:
                    text = "PASS: lifted, released and resting in the tray | simulator test"
                draw.text((30, 912), text, font=label, fill=(255, 207, 126))
                writer.append_data(np.asarray(frame))
                if index in indices:
                    frame.save(args.output / f"lerobot-frame-{index:04d}.png")
                    if index == lift_index:
                        frame.save(args.output / "lerobot-lift.png")
                    if index == indices[-1]:
                        frame.save(args.output / "lerobot-final.png")
    finally:
        renderer.close()
        env.close()
    metadata = {"scope": "HD presentation replay of saved learned-policy evaluation states; not a new test", "seed": seed,
                "success": selected["success"], "grader": selected["grader"], "frames": len(qpos), "seconds": len(qpos) / 30,
                "video": output.name, "trajectory": str(trajectory), "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
                "source_evaluation": str(args.evaluation / "report.json"), "full_qpos_and_qvel_replayed": True, "lift_frame": lift_index,
                "presentation_shadow_map_disabled": True,
                "physical_robot_tested": False, "orin_tested": False}
    (args.output / "lerobot-video-provenance.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
