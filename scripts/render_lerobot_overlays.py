"""HD saved-state ACT replay with explicitly labelled diagnostic overlays.

Run in the WSL simulator environment after evaluation has released the GPU.
Amber targets are forward kinematics of clipped joint action chunks; they are
not a Cartesian planner, object detections, or validated future physical motion.
The two RGB insets are saved, unannotated policy inputs from the current tick.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
CYAN = (81, 232, 232)
AMBER = (255, 188, 67)
GREEN = (146, 234, 155)
INK = (9, 20, 31)
PROJECTION_SOURCE = "https://raw.githubusercontent.com/google-deepmind/mujoco/3.6.0/src/render/classic/render_gl3.c"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(scene, points, width, height):
    """Match MuJoCo mono setView()/mjr_lookAt(), using the active GL camera.

    MuJoCo 3.6.0 render_gl3.c setView lines 656-709 averages both cameras,
    chooses explicit frustum_width or viewport aspect, and uses glFrustum.
    Pixel coordinates here have PIL's top-left origin. No scene transform is
    enabled for these replays; reject one instead of silently misprojecting it.
    """
    if scene.enabletransform or scene.stereo != mujoco.mjtStereo.mjSTEREO_NONE:
        raise ValueError("Projection requires an untransformed mono scene")
    cam = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
    forward = np.asarray(cam.forward, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, cam.up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = np.asarray(points, dtype=float).reshape(-1, 3) - cam.pos
    depth = relative @ forward
    x, y = relative @ right, relative @ up
    if not cam.orthographic:
        scale = cam.frustum_near / np.maximum(depth, 1e-12)
        x, y = x * scale, y * scale
    half = float(cam.frustum_width) or (width / height * (cam.frustum_top - cam.frustum_bottom) / 2)
    pixels = np.column_stack([
        width * (x - (cam.frustum_center - half)) / (2 * half),
        height * (1 - (y - cam.frustum_bottom) / (cam.frustum_top - cam.frustum_bottom)),
    ])
    visible = (depth > cam.frustum_near) & (depth < cam.frustum_far)
    visible &= np.all((pixels >= [0, 0]) & (pixels < [width, height]), axis=1)
    return pixels, visible


def tcp(model, data, tcp_local):
    body = model.body("gripper").id
    return data.xpos[body] + data.xmat[body].reshape(3, 3) @ tcp_local


def checked_fk(env, arrays, tcp_local):
    """Independent MjData; never overwrite rollout data with action targets."""
    kin = mujoco.MjData(env.model)
    original = env.data.qpos.copy()
    controls = arrays["chunks_controls_rad"]
    raw = arrays["chunks_actions_rad"]
    steps = arrays["replan_steps"].astype(int)
    if controls.shape != (len(steps), 16, 6):
        raise ValueError(f"Expected R x 16 x 6 action chunks, got {controls.shape}")
    # Sim.step also clips gripper percent via q_to_deg/deg_to_q before ctrlrange.
    sim = importlib.import_module("sim")
    expected = raw.astype(float).copy()
    expected[..., -1] = np.clip(expected[..., -1], sim.GRIPPER_CLOSED, sim.GRIPPER_OPEN)
    expected = np.clip(expected, *env.model.actuator_ctrlrange.T)
    np.testing.assert_allclose(controls, expected, atol=1e-6, rtol=0)
    targets = np.empty((len(steps), 16, 3))
    for r, step in enumerate(steps):
        kin.qpos[:] = arrays["qpos_before"][step]
        kin.qvel[:] = 0
        for k, control in enumerate(controls[r]):
            kin.qpos[env.qadr] = control
            mujoco.mj_forward(env.model, kin)
            targets[r, k] = tcp(env.model, kin, tcp_local)
    if "chunks_tcp_target_m" in arrays:
        np.testing.assert_allclose(targets, arrays["chunks_tcp_target_m"], atol=2e-6, rtol=0)
    np.testing.assert_array_equal(env.data.qpos, original)
    return targets


def temporal_checks(arrays):
    n = len(arrays["qpos_before"])
    steps = arrays["replan_steps"].astype(int)
    expected = np.arange(0, n, 8)
    np.testing.assert_array_equal(steps, expected)
    indices = arrays["frame_replan_index"].astype(int)
    offsets = arrays["frame_chunk_offset"].astype(int)
    np.testing.assert_array_equal(indices, np.arange(n) // 8)
    np.testing.assert_array_equal(offsets, np.arange(n) % 8)
    np.testing.assert_allclose(arrays["actions_rad"], arrays["chunks_actions_rad"][indices, offsets], atol=1e-6, rtol=0)
    np.testing.assert_allclose(arrays["controls_rad"], arrays["chunks_controls_rad"][indices, offsets], atol=1e-6, rtol=0)
    np.testing.assert_allclose(arrays["simulation_times_after"][:-1], arrays["simulation_times_before"][1:], atol=1e-12, rtol=0)
    np.testing.assert_allclose(arrays["simulation_times_after"] - arrays["simulation_times_before"], 1 / 30, atol=1e-10, rtol=0)
    np.testing.assert_allclose(arrays["qpos_after"][:-1], arrays["qpos_before"][1:], atol=1e-12, rtol=0)
    np.testing.assert_allclose(arrays["qvel_after"][:-1], arrays["qvel_before"][1:], atol=1e-12, rtol=0)
    for key in ("images_scene", "images_wrist"):
        if key in arrays and (arrays[key].shape != (n, 96, 96, 3) or arrays[key].dtype != np.uint8):
            raise ValueError(f"{key} must contain actual N x 96 x 96 x 3 uint8 RGB")
    for key, value in arrays.items():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite saved values in {key}")
    return {"frames": n + 1, "recorded_control_ticks": n, "replan_count": len(steps),
            "replan_every_ticks": 8, "horizon_ticks": 16, "fps": 30,
            "state_continuity_checked": True, "frame_replan_mapping_checked": True}


def camera_pixel_check(renderer, width, height):
    """Render isolated world markers and compare centroids to projection."""
    scene = renderer.scene
    cam = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
    forward = np.asarray(cam.forward)
    right = np.cross(forward, cam.up)
    points = np.array([cam.pos + .55 * forward + dx * right + dy * cam.up
                       for dx, dy in [(-.08, -.04), (.07, .05), (0, 0)]])
    pixels, valid = project(scene, points, width, height)
    errors = []
    for point, pixel, ok in zip(points, pixels, valid):
        if not ok:
            raise AssertionError("Projection calibration marker outside camera")
        scene.ngeom = 1
        scene.nlight = 0
        geom = scene.geoms[0]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.full(3, .002), point, np.eye(3).ravel(), np.array([1., 0., 1., 1.]))
        geom.emission = 1
        rgb = renderer.render().copy()
        mask = (rgb[..., 0] > 180) & (rgb[..., 1] < 70) & (rgb[..., 2] > 180)
        ys, xs = np.where(mask)
        if len(xs) < 4:
            raise AssertionError("Projection calibration marker not found")
        error = float(np.linalg.norm([xs.mean() + .5 - pixel[0], ys.mean() + .5 - pixel[1]]))
        if error > 1.5:
            raise AssertionError(f"Camera projection differs from renderer by {error:.3f}px")
        errors.append(error)
    return {"method": "isolated world-sphere rendered pixel centroid vs projected center",
            "errors_px": errors, "max_error_px": max(errors), "tolerance_px": 1.5,
            "camera": "active overview mjv_averageCamera", "source": PROJECTION_SOURCE}


def font(size, bold=False):
    root = Path("/mnt/c/Windows/Fonts")
    path = root / ("segoeuib.ttf" if bold else "segoeui.ttf")
    if not path.exists():
        path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size)


def line(draw, pixels, valid, color, width=3):
    for a, b, va, vb in zip(pixels[:-1], pixels[1:], valid[:-1], valid[1:]):
        if va and vb:
            draw.line([tuple(a), tuple(b)], fill=color, width=width)


def dashed(draw, pixels, valid, color, width=2):
    for a, b, va, vb in zip(pixels[:-1], pixels[1:], valid[:-1], valid[1:]):
        if va and vb:
            length = float(np.linalg.norm(b - a))
            for start in np.arange(0, length, 10):
                draw.line([tuple(a + (b - a) * start / max(length, 1e-9)),
                           tuple(a + (b - a) * min(start + 5, length) / max(length, 1e-9))], fill=color, width=width)


def number_targets(draw, pixels, valid, text_font, offset=0):
    """Spread labels around dense joint chunks without moving target dots."""
    used = []
    for k, (pixel, ok) in enumerate(zip(pixels, valid)):
        if not ok:
            continue
        x, y = pixel
        tick = k + offset + 1
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=AMBER if tick <= 8 else INK,
                     outline=AMBER, width=2)
        if tick not in (1, 4, 8, 12, 16):
            continue
        label = str(tick)
        box = draw.textbbox((0, 0), label, font=text_font)
        w, h = box[2] + 8, box[3] + 4
        chosen = None
        for radius in (12, 28, 44, 60, 76, 92):
            for angle in np.linspace(-np.pi, np.pi, 12, endpoint=False):
                lx, ly = x + radius * np.cos(angle), y + radius * np.sin(angle)
                rect = (lx - w / 2, ly - h / 2, lx + w / 2, ly + h / 2)
                if all(rect[2] + 2 < b[0] or rect[0] > b[2] + 2 or
                       rect[3] + 2 < b[1] or rect[1] > b[3] + 2 for b in used):
                    chosen = rect
                    break
            if chosen is not None:
                break
        if chosen is None:
            chosen = (x + 8, y + 8, x + 8 + w, y + 8 + h)
        used.append(chosen)
        cx, cy = (chosen[0] + chosen[2]) / 2, (chosen[1] + chosen[3]) / 2
        draw.line([(x, y), (cx, cy)], fill=AMBER, width=1)
        draw.rounded_rectangle(chosen, radius=4, fill=INK, outline=AMBER)
        draw.text((chosen[0] + 4, chosen[1]), label, font=text_font, fill=AMBER)


def render_suite(suite, output, width):
    summary = json.loads((suite / "report.json").read_text())
    if not summary.get("complete"):
        raise ValueError("Suite must finish and release GPU before HD rendering")
    rows = summary["episodes"]
    selected = []
    for case in dict.fromkeys(row["case"] for row in rows):
        matching = [row for row in rows if row["case"] == case]
        # Same held-out seed makes visual variants a paired comparison.
        selected.append(next((row for row in matching if row["seed"] == 401), matching[0]))
    if all(row["success"] for row in selected):
        failure = next((row for row in rows if not row["success"]), None)
        if failure:
            selected.append(failure)
    if summary.get("reference_replay"):
        selected.append(summary["reference_replay"])
    if summary.get("reference_failure_replay"):
        selected.append(summary["reference_failure_replay"])
    clips = []
    for row in selected:
        relative = Path(row["episodepath"])
        destination = output / (f"reference-{row['seed']}" if row["seed"] in (300, 301) else relative)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--episode", str(suite / relative),
                        "--output", str(destination), "--width", str(width)], check=True)
        clip = json.loads((destination / "report.json").read_text())
        for key in ("video", "poster", "provenance", "contact_sheet"):
            clip[key] = str((destination / clip[key]).relative_to(output))
        clip["snapshots"] = [str((destination / name).relative_to(output)) for name in clip["snapshots"]]
        clip["episodepath"] = str(relative)
        clips.append(clip)
        (output / "report.json").write_text(json.dumps({"clips": clips, "complete": False}, indent=2))
    (output / "report.json").write_text(json.dumps({"clips": clips, "complete": True,
        "selection": "Paired fresh seed401 per case; extra failure if necessary; reference301 excluded from fresh count"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", type=Path)
    source.add_argument("--suite", type=Path, help="Render representative clips and write aggregate overlay manifest")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, choices=[1280, 1440], default=1440)
    parser.add_argument("--validate-only", action="store_true", help="CPU saved-state/FK checks; no HD renderer")
    args = parser.parse_args()
    if args.suite:
        if args.validate_only:
            parser.error("--validate-only requires --episode")
        render_suite(args.suite, args.output or args.suite / "overlays", args.width)
        return
    report_path = args.episode / "report.json"
    trajectory = args.episode / "trajectory.npz"
    report = json.loads(report_path.read_text())
    with np.load(trajectory, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    checks = temporal_checks(arrays)
    environment = importlib.import_module(report.get("environment_module", "lerobot_case_environment"))
    baseline = importlib.import_module("so101_pick_place_baseline")
    case = report["case"]
    if isinstance(case, dict):
        case = case["id"]
    env = environment.make_env(case, report["seed"], width=96, height=96, offscreen=False)
    targets = checked_fk(env, arrays, baseline.TCP_LOCAL)
    checks["independent_fk_checked"] = True
    checks["actuator_clipping_checked"] = True
    reference = "reference" in report.get("scope", "").lower() or report["seed"] in (300, 301)
    output = args.output or ROOT / "runs/lerobot-cases/overlays" / (Path(f"reference-{report['seed']}") if reference else Path(case) / f"seed-{report['seed']}")
    output.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        (output / "cpu-validation.json").write_text(json.dumps(checks, indent=2))
        env.close()
        print(json.dumps(checks, indent=2))
        return
    width, height = args.width, args.width * 3 // 4
    scale = width / 1440
    env.model.vis.global_.offwidth = width
    env.model.vis.global_.offheight = height
    renderer = mujoco.Renderer(env.model, height, width)
    env.data.qpos[:] = arrays["qpos_before"][0]
    env.data.qvel[:] = arrays["qvel_before"][0]
    mujoco.mj_forward(env.model, env.data)
    renderer.update_scene(env.data, camera="overview")
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    calibration = camera_pixel_check(renderer, width, height)
    n = len(arrays["qpos_before"])
    times = arrays["simulation_times_before"]
    title_font, body_font = font(round(32 * scale), True), font(round(22 * scale))
    small_font, tiny_font = font(round(18 * scale)), font(round(14 * scale), True)
    observed = []
    max_seen_cube_z = -float("inf")
    records = []
    snapshots = sorted(set([0, 7, 8, n // 2, n]))
    poster_index = n // 2
    if "object_positions_for_grading_only" in arrays:
        poster_index = int(np.argmax(arrays["object_positions_for_grading_only"][:, 2]))
        snapshots = sorted(set(snapshots + [poster_index]))
    grader = report["grader"]
    success = bool(grader["success"])
    video = output / "overlay.mp4"
    try:
        with imageio.get_writer(str(video), fps=30, codec="libx264", quality=9, macro_block_size=1) as writer:
            for index in range(n + 1):
                source_index = min(index, n - 1)
                terminal = index == n
                pose = arrays["qpos_after"][-1] if terminal else arrays["qpos_before"][index]
                velocity = arrays["qvel_after"][-1] if terminal else arrays["qvel_before"][index]
                env.data.qpos[:] = pose
                env.data.qvel[:] = velocity
                simtime = float(arrays["simulation_times_after"][-1]) if terminal else float(times[index])
                env.data.time = simtime
                mujoco.mj_forward(env.model, env.data)
                observed.append(tcp(env.model, env.data, baseline.TCP_LOCAL).copy())
                renderer.update_scene(env.data, camera="overview")
                if hasattr(environment, "decorate_scene"):
                    environment.decorate_scene(env, renderer)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                frame = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(frame)
                r = int(arrays["frame_replan_index"][source_index])
                offset = int(arrays["frame_chunk_offset"][source_index])
                future_px, future_ok = project(renderer.scene, targets[r], width, height)
                trail_px, trail_ok = project(renderer.scene, observed[-120:], width, height)
                line(draw, trail_px, trail_ok, CYAN, round(4 * scale))
                if not terminal:
                    line(draw, future_px[offset:8], future_ok[offset:8], AMBER, round(3 * scale))
                    dashed(draw, future_px[7:], future_ok[7:], AMBER, round(2 * scale))
                    number_targets(draw, future_px[offset:], future_ok[offset:], tiny_font, offset)
                # Tray world context is privileged grader diagnostic data.
                tray = env.data.body("tray").xpos.copy()
                corners = tray + np.array([[-.070, -.055, .047], [.070, -.055, .047],
                                           [.070, .055, .047], [-.070, .055, .047], [-.070, -.055, .047]])
                tray_px, tray_ok = project(renderer.scene, corners, width, height)
                line(draw, tray_px, tray_ok, GREEN, round(3 * scale))
                draw.rectangle((0, 0, width, round(132 * scale)), fill=INK)
                draw.text((26 * scale, 12 * scale), "SO-101 / ACT  |  joint-chunk replay", font=title_font, fill="white")
                case_title = report.get("title", case.replace("_", " "))
                if reference:
                    case_title += " / previously evaluated reference, excluded from fresh count"
                seed_scope = "reference seed" if reference else "fresh seed"
                draw.text((26 * scale, 58 * scale), f"{case_title}  |  {seed_scope} {report['seed']}  |  sim time {simtime:.2f} s", font=body_font, fill=CYAN)
                draw.text((26 * scale, 96 * scale), "30 Hz simulated-time replay / no real-time hardware test", font=small_font, fill=(185, 196, 208))
                # Present only the RGB actually saved before the policy call.
                for inset_i, key in enumerate(("images_scene", "images_wrist")):
                    if key not in arrays:
                        continue
                    x, y = round(width - 224 * scale), round((160 + inset_i * 267) * scale)
                    size = round(192 * scale)
                    image = Image.fromarray(arrays[key][source_index]).resize((size, size), Image.Resampling.NEAREST)
                    draw.rectangle((x - 10 * scale, y - 31 * scale, x + size + 10 * scale, y + size + 37 * scale), fill=INK)
                    frame.paste(image, (x, y))
                    draw.text((x, y - 28 * scale), "SCENE RGB" if inset_i == 0 else "WRIST RGB", font=small_font, fill="white")
                    draw.text((x, y + size + 6 * scale), "96 x 96 / policy input" if not terminal else "last input / prior tick", font=small_font, fill=(185, 196, 208))
                bottom = round(height - 209 * scale)
                draw.rectangle((0, bottom, width, height), fill=INK)
                elapsed = float(arrays["inference_seconds"][int(arrays["replan_steps"][r])]) * 1000
                sim = importlib.import_module("sim")
                opening = float(np.clip((env.data.qpos[env.qadr[-1]] - sim.GRIPPER_CLOSED) /
                                         (sim.GRIPPER_OPEN - sim.GRIPPER_CLOSED) * 100, 0, 100))
                cube_z = float(env.data.body("obj0").xpos[2])
                max_seen_cube_z = max(max_seen_cube_z, cube_z)
                telemetry = f"REPLAN {r + 1:03d} / tick {offset + 1}/8   |   logged inference {elapsed:.1f} ms   |   opening {opening:.1f}%   |   cube z {cube_z * 100:.1f} cm"
                draw.text((26 * scale, bottom + 13 * scale), telemetry, font=body_font, fill="white")
                draw.text((26 * scale, bottom + 52 * scale), "CYAN  observed TCP trail (4 s)", font=small_font, fill=CYAN)
                draw.text((410 * scale, bottom + 52 * scale), "AMBER  1-16: FK of predicted joint targets", font=small_font, fill=AMBER)
                draw.text((26 * scale, bottom + 86 * scale), "30 Hz / refresh every 8 ticks; solid 1-8 queued, dashed 9-16 unexecuted; motion unvalidated", font=small_font, fill=AMBER)
                draw.text((26 * scale, bottom + 119 * scale), "GREEN tray + cube height: simulator ground truth / grader diagnostics; never supplied to ACT", font=small_font, fill=GREEN)
                if terminal:
                    verdict = "PASS" if success else "FAIL"
                    status = f"FINAL FROZEN GRADER: {verdict}  |  lifted {grader.get('lifted')} / inside {grader.get('inside_tray_for_1_5_s')} / released {grader.get('released_for_1_5_s')} / stable {grader.get('stable')}"
                else:
                    status = f"Past lift evidence (>9 cm): {'YES' if max_seen_cube_z > .09 else 'pending'}  |  final independent grader pending"
                draw.text((26 * scale, bottom + 159 * scale), status, font=small_font, fill=GREEN if success and terminal else (233, 195, 180))
                # Hold the completed final state for 1.5 s for readable verdict.
                # Simulation time stays fixed; this is an explicit replay pause.
                if terminal:
                    draw.text((width - 290 * scale, 96 * scale), "END STATE / replay paused", font=small_font, fill=AMBER)
                for _ in range(45 if terminal else 1):
                    writer.append_data(np.asarray(frame))
                if index in snapshots:
                    frame.save(output / f"frame-{index:04d}.png")
                if index == poster_index:
                    frame.save(output / "poster.png")
                records.append({"frame": index, "simulation_time_s": simtime, "replan_index": r,
                                "replan_control_tick": int(arrays["replan_steps"][r]), "chunk_offset": offset,
                                "replan_changed": index == 0 or (not terminal and index % 8 == 0),
                                "future_tcp_world_m": targets[r].tolist(), "future_tcp_pixels": future_px.tolist(),
                                "target_nominal_time_s": (float(times[int(arrays['replan_steps'][r])]) + (np.arange(16) + 1) / 30).tolist(),
                                "visible": future_ok.tolist(), "observed_tcp_world_m": observed[-1].tolist(),
                                "rgb_source_tick": source_index, "terminal_state": terminal})
    finally:
        renderer.close()
        env.close()
    (output / "overlay-record.json").write_text(json.dumps(records, separators=(",", ":")))
    # A small PNG contact sheet is convenient in file previews and the viewer.
    sheet_indices = [0, poster_index, n]
    contact = Image.new("RGB", (720 * 3, 540), INK)
    for column, i in enumerate(sheet_indices):
        with Image.open(output / f"frame-{i:04d}.png") as snapshot:
            contact.paste(snapshot.resize((720, 540), Image.Resampling.LANCZOS), (720 * column, 0))
    contact.save(output / "contact-sheet.png")
    provenance = {"scope": "HD presentation replay of actual saved MuJoCo states; no new control evaluation",
                  "source_report": str(report_path), "source_trajectory": str(trajectory),
                  "source_report_sha256": sha256(report_path), "trajectory_sha256": sha256(trajectory),
                  "checkpoint_sha256": report.get("checkpoint_sha256"), "frozen_baseline_and_grader_sha256": report.get("baseline_and_grader_sha256"),
                  "renderer_sha256": sha256(__file__), "environment_sha256": sha256(environment.__file__),
                  "video_sha256": sha256(video), "overlay_record_sha256": sha256(output / "overlay-record.json"),
                  "video": video.name, "case": case, "seed": report["seed"], "resolution": [width, height],
                  "video_frames": n + 45, "final_state_hold_frames": 45, "simulated_seconds": float(arrays["simulation_times_after"][-1] - times[0]),
                  "success": success, "grader": grader, "temporal_validation": checks, "reference_excluded_from_fresh_count": reference,
                  "projection_validation": calibration, "full_qpos_qvel_replayed": True,
                  "terminal_state_appended": True, "policy_rgb_insets": "actual saved unannotated 96px inputs before control",
                  "predictions": "Independent FK of physical actuator-clipped ACT joint chunk targets; not validated future motion",
                  "world_diagnostics": "Tray objective and cube height are grader context, never policy inputs",
                  "shadow_disabled_presentation_only": True, "physical_robot_tested": False,
                  "snapshots": {f"frame-{i:04d}.png": sha256(output / f"frame-{i:04d}.png") for i in snapshots}}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (output / "report.json").write_text(json.dumps({"case": case, "seed": report["seed"], "success": success,
        "reference_excluded_from_fresh_count": reference, "video": video.name, "provenance": "provenance.json",
        "snapshots": [f"frame-{i:04d}.png" for i in snapshots], "contact_sheet": "contact-sheet.png", "poster": "poster.png",
        "width": width, "height": height, "frames": n + 45, "final_state_hold_frames": 45, "complete": True}, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
