"""Explore frozen ACT visual robustness; retain every rollout and full chunks.

Only two RGB cameras and six joints enter ACT. Recording, FK visualization and
grading happen outside the policy. This runs simulation, never a physical robot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

import lerobot_case_environment as environment
import so101_pick_place_baseline as baseline

ROOT = Path(__file__).resolve().parents[1]
sim = baseline.sim


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def tcp(env, data):
    b = env.model.body("gripper").id
    return data.xpos[b] + data.xmat[b].reshape(3, 3) @ baseline.TCP_LOCAL


def scripted_check(case, seed, out):
    """Reuse the exact established scripted controller and frozen strict grader."""
    original = baseline.make_env
    baseline.make_env = lambda seed=0, **kw: environment.make_env(case, seed, **kw)
    try:
        result = baseline.run(seed, out, render=False)
    finally:
        baseline.make_env = original
    result.update(case=case, case_config=environment.CASES[case],
                  scope="Scripted physical feasibility only; privileged IK controller, no learned policy")
    save_json(out / "report.json", result)
    return result


def run_episode(policy, stats, case, seed, args, metadata, relative_path=None):
    out = args.output / (relative_path or f"{case}/seed-{seed}")
    assert not out.exists(), f"Do not overwrite a completed or interrupted experiment: {out}"
    out.mkdir(parents=True)
    env = environment.make_env(case, seed, 96, 96)
    initial_xyz = env.object_positions()["obj0"].copy()
    renderer = mujoco.Renderer(env.model, 240, 320)
    kin = mujoco.MjData(env.model)
    policy.reset()
    records = {name: [] for name in ("time_s", "simulation_times_before", "simulation_times_after", "qpos_before", "qpos_after", "qvel_before", "qvel_after", "states_before", "states_after", "actions_rad", "controls_rad", "tcp_before_m", "tcp_after_m", "images_scene", "images_wrist", "inference_seconds", "replan_steps", "chunks_normalized", "chunks_actions_rad", "chunks_controls_rad", "chunks_tcp_target_m", "frame_replan_index", "frame_chunk_offset", "object_positions_for_grading_only")}
    original_predict = policy.predict_action_chunk
    captured = []
    def capture(batch):
        result = original_predict(batch)
        captured.append(result.detach().float().cpu().numpy()[0].copy())
        return result
    policy.predict_action_chunk = capture
    events, excess_max = 0, 0.
    started = time.perf_counter()
    try:
        with imageio.get_writer(str(out / "rollout.mp4"), fps=sim.FPS, codec="libx264", quality=8, macro_block_size=16) as writer:
            for step in range(args.max_steps):
                state = env.data.qpos[env.qadr].astype(np.float32).copy()
                before_pos, before_vel = env.data.qpos.copy(), env.data.qvel.copy()
                time_before = float(env.data.time)
                before_tcp = tcp(env, env.data).copy()
                batch = {"observation.state": torch.from_numpy((state - stats["states"]["mean"]) / stats["states"]["std"]).unsqueeze(0).cuda()}
                frames = {}
                for camera in ("scene", "wrist"):
                    frames[camera] = env.render(camera)
                    rgb = (frames[camera].transpose(2, 0, 1).astype(np.float32) / 255. - .5) / .5
                    batch[f"observation.images.{camera}"] = torch.from_numpy(rgb).unsqueeze(0).cuda()
                fresh = len(policy._action_queue) == 0
                torch.cuda.synchronize()
                tic = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    selected = policy.select_action(batch)
                torch.cuda.synchronize()
                latency = time.perf_counter() - tic
                lo, hi = env.model.actuator_ctrlrange.T
                if fresh:
                    assert len(captured) == 1
                    normalized = captured.pop()
                    chunk = normalized * stats["actions"]["std"] + stats["actions"]["mean"]
                    clipped = np.clip(chunk, lo, hi)
                    assert chunk.shape == (16, 6) and np.isfinite(chunk).all()
                    target_tcp = []
                    for q in clipped:
                        kin.qpos[:] = before_pos
                        kin.qpos[env.qadr] = q
                        mujoco.mj_forward(env.model, kin)
                        target_tcp.append(tcp(env, kin).copy())
                    records["replan_steps"].append(step)
                    records["chunks_normalized"].append(normalized)
                    records["chunks_actions_rad"].append(chunk)
                    records["chunks_controls_rad"].append(clipped)
                    records["chunks_tcp_target_m"].append(target_tcp)
                action = selected.float().cpu().numpy()[0] * stats["actions"]["std"] + stats["actions"]["mean"]
                assert action.shape == (6,) and np.isfinite(action).all()
                excess = np.maximum(np.maximum(lo - action, action - hi), 0.)
                events += int(np.any(excess > 1e-6))
                excess_max = max(excess_max, float(excess.max()))
                env.step(sim.q_to_deg(action))
                values = {"time_s": float(env.data.time), "qpos_before": before_pos, "qvel_before": before_vel,
                          "simulation_times_before": time_before, "simulation_times_after": float(env.data.time),
                          "qpos_after": env.data.qpos.copy(), "qvel_after": env.data.qvel.copy(), "states_before": state,
                          "states_after": env.data.qpos[env.qadr].copy(), "actions_rad": action, "controls_rad": env.data.ctrl.copy(),
                          "tcp_before_m": before_tcp, "tcp_after_m": tcp(env, env.data).copy(),
                          "images_scene": frames["scene"], "images_wrist": frames["wrist"], "inference_seconds": latency,
                          "frame_replan_index": len(records["replan_steps"]) - 1,
                          "frame_chunk_offset": step - records["replan_steps"][-1],
                          "object_positions_for_grading_only": env.object_positions()["obj0"].copy()}
                for key, value in values.items():
                    records[key].append(value)
                renderer.update_scene(env.data, camera="overview")
                environment.decorate_scene(env, renderer)
                writer.append_data(renderer.render().copy())
        grader = environment.task_success(env)
        arrays = {key: np.asarray(value) for key, value in records.items()}
        np.savez_compressed(out / "trajectory.npz", **arrays)
        replan_times = arrays["inference_seconds"][arrays["replan_steps"]]
        result = dict(metadata, case=case, case_config=environment.CASES[case], seed=seed,
                      title=environment.CASES[case]["label"], episodepath=str(out.relative_to(args.output)), initial_object_xyz_m=initial_xyz.tolist(),
                      success=bool(grader["success"]), grader=grader, steps=args.max_steps,
                      video="rollout.mp4", trajectory="trajectory.npz", wall_seconds=time.perf_counter() - started,
                      actuator_target_limit_events=events, max_target_limit_excess_rad=excess_max,
                      fresh_chunk_inference_seconds={"median": float(np.median(replan_times)), "p95": float(np.percentile(replan_times, 95)), "count": len(replan_times)},
                      video_state="Frame i renders qpos_after[i]; RGB inputs and chunk i are from qpos_before[i]",
                      chunk_projection="Forward kinematics of clipped predicted joint targets; no dynamics or object forecast",
                      joint_order=sim.JOINTS, actuator_limits_rad=env.model.actuator_ctrlrange.tolist(),
                      tcp_local_m=baseline.TCP_LOCAL.tolist(), simulation_fps=sim.FPS,
                      complete=True)
        save_json(out / "report.json", result)
        print(json.dumps({k: result[k] for k in ("case", "seed", "success", "grader", "wall_seconds", "fresh_chunk_inference_seconds")}), flush=True)
        return result
    finally:
        policy.predict_action_chunk = original_predict
        renderer.close()
        env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "runs/lerobot-teaser/train-8000")
    ap.add_argument("--output", type=Path, default=ROOT / "runs/lerobot-cases")
    ap.add_argument("--cases", nargs="+", choices=list(environment.CASES), default=list(environment.CASES))
    ap.add_argument("--max-steps", type=int, default=525)
    ap.add_argument("--skip-scripted", action="store_true")
    ap.add_argument("--reference-only", action="store_true", help="Replay previously successful seed301 separately from fresh cases")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    training = json.loads((args.checkpoint / "training-report.json").read_text())
    old_seeds = {int(re.search(r"seed[-_](\d+)", row["file"]).group(1)) for key in ("train_episodes", "validation_episodes") for row in training[key]}
    new_seeds = [s for case in args.cases for s in environment.CASES[case]["seeds"]]
    assert not old_seeds.intersection(new_seeds)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(42)
    config = ACTConfig.from_pretrained(args.checkpoint)
    assert config.chunk_size == 16 and config.n_action_steps == 8 and config.temporal_ensemble_coeff is None
    assert all(f.shape == (3, 96, 96) for f in config.image_features.values())
    policy = ACTPolicy.from_pretrained(args.checkpoint, config=config).cuda().eval()
    stats = {k: {n: np.asarray(v, dtype=np.float32) for n, v in r.items()} for k, r in json.loads((args.checkpoint / "normalization.json").read_text()).items()}
    metadata = {"policy": "Official LeRobot ACT frozen train-8000", "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": sha256(args.checkpoint / "model.safetensors"), "checkpoint_config_sha256": sha256(args.checkpoint / "config.json"),
                "normalization_sha256": sha256(args.checkpoint / "normalization.json"), "environment_sha256": sha256(environment.__file__),
                "baseline_and_grader_sha256": sha256(baseline.__file__), "runner_sha256": sha256(__file__),
                "torch": torch.__version__, "mujoco": mujoco.__version__, "gpu": torch.cuda.get_device_name(),
                "observation": "Scene RGB96x96, wrist RGB96x96, six joint states only",
                "architecture": {k: getattr(config, k) for k in ("chunk_size", "n_action_steps", "dim_model", "use_vae")},
                "scope": "Exploratory visual robustness on new random starts; not a new untouched benchmark, no training or tuning",
                "control": "Absolute joint targets in radians; established actuator clipping", "hardware_tested": False,
                "camera_pixels_logged": True, "all_failures_retained": True, "scripted_baselines": [], "episodes": [], "complete": False}
    if args.reference_only:
        result = run_episode(policy, stats, "nominal", 301, args, dict(metadata, scope="Reproduction of a previously evaluated successful seed; excluded from fresh robustness count"), "reference-301")
        print("Reference replay complete", result["success"], flush=True)
        return
    save_json(args.output / "report.json", metadata)
    for case in args.cases:
        if not args.skip_scripted:
            check = scripted_check(case, environment.CASES[case]["seeds"][0], args.output / "scripted-feasibility" / case)
            metadata["scripted_baselines"].append(check)
            save_json(args.output / "report.json", metadata)
        for seed in environment.CASES[case]["seeds"]:
            result = run_episode(policy, stats, case, seed, args, {k: v for k, v in metadata.items() if k not in ("episodes", "scripted_baselines", "complete")})
            metadata["episodes"].append(result)
            metadata["success_count"] = sum(row["success"] for row in metadata["episodes"])
            metadata["total"] = len(metadata["episodes"])
            save_json(args.output / "report.json", metadata)
    metadata["case_totals"] = {case: {"successful": sum(r["success"] for r in metadata["episodes"] if r["case"] == case), "total": sum(r["case"] == case for r in metadata["episodes"])} for case in args.cases}
    metadata["reference_replay"] = run_episode(policy, stats, "nominal", 301, args,
        {k: v for k, v in dict(metadata, scope="Previously evaluated successful seed301 reproduction; excluded from fresh cases").items() if k not in ("episodes", "scripted_baselines", "complete", "case_totals")}, "reference-301")
    metadata["complete"] = True
    save_json(args.output / "report.json", metadata)


if __name__ == "__main__":
    main()
