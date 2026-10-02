"""Judge one saved hosted rollout from its object poses and inspect its MP4."""

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/flux-action-study/hosted-so101"


def main():
    events = json.loads((OUT / "events.json").read_text(encoding="utf-8"))
    reset = next(item["data"][0] for item in events if isinstance(item["data"], list) and item["data"] and isinstance(item["data"][0], dict) and item["data"][0].get("kind") == "reset")
    final = json.loads((OUT / "final-response.json").read_text(encoding="utf-8"))[0]
    ids = reset["bodies"]
    red = next(g for g in reset["geoms"] if g["kind"] == "box" and g["rgba"][:3] == [0.8, 0.12, 0.1])
    tray = next(g for g in reset["geoms"] if g["kind"] == "box" and g["rgba"][:3] == [0.55, 0.56, 0.58] and g["size"][2] == 0.002)
    poses = np.asarray(final["poses"], dtype=np.float64).reshape(-1, len(ids), 7)
    assert poses.shape[0] == 256 and np.isfinite(poses).all()
    initial = np.asarray(reset["poses"][0], dtype=np.float64).reshape(len(ids), 7)
    object_idx, tray_idx = ids.index(red["body"]), ids.index(tray["body"])
    xyz = poses[:, object_idx, :3]
    tray_xyz = poses[:, tray_idx, :3]
    assert np.ptp(tray_xyz, axis=0).max() == 0, "Judge assumes a stationary tray"
    relative = xyz - tray_xyz
    # A necessary placement condition: object center inside tray outer XY bounds,
    # and within 1 cm of floor(4 mm)+cube-halfheight(15 mm). Passing would still
    # need a stricter footprint/orientation and release check; failing is certain.
    inside_xy = (np.abs(relative[:, 0]) <= 0.075) & (np.abs(relative[:, 1]) <= 0.06)
    at_rest_height = np.abs(relative[:, 2] - 0.019) <= 0.01
    condition = inside_xy & at_rest_height
    goal_passed = bool(condition[-30:].all())
    if goal_passed:
        task_outcome = "necessary geometric placement condition passed; grasp/release not independently verified"
    else:
        task_outcome = "failed: cube did not reach the tray placement region"
    initial_xyz = initial[object_idx, :3]
    result = {
        "attempts": 1,
        "is_benchmark_success_rate": False,
        "outcome": task_outcome,
        "placement_necessary_condition_passed": goal_passed,
        "protocol": "Final 30 poses (1 simulated second) must all have cube center within tray outer half-widths 0.075 m / 0.060 m, and z within 0.010 m of expected center height 0.019 m. This is a necessary coarse condition; failure disproves placement, passing alone does not prove a released stable grasp-and-place.",
        "cube_body_id": red["body"],
        "tray_body_id": tray["body"],
        "initial_cube_xyz_m": initial_xyz.tolist(),
        "final_cube_xyz_m": xyz[-1].tolist(),
        "tray_xyz_m": tray_xyz[-1].tolist(),
        "final_cube_minus_tray_xyz_m": relative[-1].tolist(),
        "cube_max_displacement_from_initial_m": float(np.linalg.norm(xyz - initial_xyz, axis=1).max()),
        "cube_max_height_increase_from_initial_m": float((xyz[:, 2] - initial_xyz[2]).max()),
        "frames_inside_xy": int(inside_xy.sum()),
        "frames_meeting_necessary_condition": int(condition.sum()),
        "pose_frames": len(poses),
        "pose_rounding_precision_m": 0.00001,
        "source": "Saved public Space body poses; RGB video also inspected visually",
        "policy_actions_available": False,
        "visual_observation": "The arm moves and lifts away; the red cube remains on the board outside the tray. No completed grasp or placement is visible.",
    }
    (OUT / "placement-evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    report["task_success"] = "failed" if not goal_passed else "placement_condition_passed_not_full_task_verified"
    report["placement_evaluation"] = result
    report["benchmark_success_rate_claim"] = False
    report["source_note"] = "The evaluation is one hosted out-of-distribution attempt with modified normalization; it is not evidence of real SO-101 performance."
    reader = imageio.get_reader(str(OUT / "rollout.mp4"))
    meta = reader.get_meta_data()
    count = reader.count_frames()
    fonts = Path("C:/Windows/Fonts")
    font = ImageFont.truetype(str(fonts / "segoeui.ttf"), 22)
    small = ImageFont.truetype(str(fonts / "segoeui.ttf"), 19)
    sheet = Image.new("RGB", (1000, 1140), (15, 23, 35))
    draw = ImageDraw.Draw(sheet)
    draw.text((22, 16), "FLUX 3 Action SO-101 | hosted MuJoCo test", font=font, fill="white")
    draw.text((22, 51), "One attempt: red cube stayed outside the tray", font=small, fill=(255, 197, 105))
    for index, k in enumerate((0, count // 2, count - 1)):
        frame = Image.fromarray(reader.get_data(k))
        frame.save(OUT / f"video-frame-{k:03d}.png")
        frame = frame.resize((880, 660))
        frame.thumbnail((880, 300))
        x = (1000 - frame.width) // 2
        y = 100 + index * 335
        sheet.paste(frame, (x, y))
        draw.text((x, y + frame.height + 4), f"{k / meta['fps']:.2f} s simulated | {'initial' if index == 0 else 'middle' if index == 1 else 'final'}", font=small, fill="white")
    sheet.save(OUT / "rollout-first-middle-final.png")
    reader.close()
    report["video"] = {"frames": count, "fps": meta["fps"], "duration_seconds": meta["duration"], "size": list(meta["size"])}
    report["inspection_artifacts"] = ["rollout-first-middle-final.png", "placement-evaluation.json", "video-frame-000.png", f"video-frame-{count // 2:03d}.png", f"video-frame-{count - 1:03d}.png"]
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
