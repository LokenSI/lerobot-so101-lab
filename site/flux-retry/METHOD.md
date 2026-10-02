# Local FLUX SO-101 retry

This experiment asks whether the released FLUX 3 Action SO-101 checkpoint can complete a real contact grasp and placement in our simulator. A written instruction, measured joints, camera images and previous commands enter the model. Object coordinates are used for reset, grading and diagnostics only. No scripted pickup correction, attachment or post-reset object movement is allowed.

The completed `dev-v1` run is exploratory: neither of its two adapters completed the task (0/2). Its two adapters, seed 910, 24 prediction chunks, instruction and success rule were frozen before outcomes. Do not describe this small adapter comparison as an established success rate, and retain unsuccessful episodes. The hosted-median trial did not lift the cube. The speculative legacy trial reached a maximum cube world height of approximately 24.7 mm and clipped 400 commands; it did not demonstrate a contact grasp or verified calibration.

## What FLUX contributes

The unchanged pretrained BF16 checkpoint supplies 42 future command targets from each observation history. We execute the first 32 at 30 Hz and ask again. Five arm channels use consecutive command deltas; the gripper uses an absolute opening. The official inference code denormalizes and integrates these predictions onto the last command. The adapter does not perform that integration a second time.

The model was trained by its creators. This retry adds no training, demonstrations, ACT controller or Cartesian planner. CPU offload changes placement of the original BF16 weights, rather than substituting the optimized DROID model.

## Observation and camera contract

The fixed scene view appears on the left and wrist view on the right. The simulator captures each at 320 by 240 pixels; the model resizes each to 256 by 256 and joins them into its 512 by 256 canvas. These are scene and wrist cameras, not the separate stereo experiment's left and right cameras.

Every control tick records measured state and the previous command actually applied after joint clipping. Eight ticks are retained, with two visual snapshots at history indices 0 and 7. At reset, the initial observation and measured command repeat eight times. Each episode resets temporal state while retaining only cached encodings of the same immutable instruction.

## Two calibration adapters

**A: hosted median.** Native new-calibration simulator coordinates use the pinned public demo's median state and action quantiles. That demo intentionally selected these statistics for new-calibration robots. This is a declared normalization adaptation; original checkpoint files remain unchanged.

**B: legacy candidate.** Saved checkpoint statistics remain in use. The exploratory coordinate conversion is `p = scale * q + offset`, where `scale = [1,-1,1,1,1,1]` and `offset = [0,90,90,0,-2.789136,0]`. The inverse conversion applies to predicted absolute targets before physics. The same conversion applies to measured states and command history.

Official old/new SO-101 geometry supports the elbow and wrist offsets. Historical LeRobot calibration supports investigating motor-direction differences. The source dataset's precise motor homing file is absent, so B is a hypothesis and cannot be called verified calibration. Neither adapter's results justify changing the other adapter after inspecting an outcome without declaring a new development iteration.

## Physical task and success

The robot uses the established fixed-finger hull decomposition that preserves visible geometry, inertia and motors while removing a convex hull spanning the concave grasp opening. A free cube moves through contact dynamics. This is a simulator geometry correction, not learned grasp assistance.

Success requires the existing strict lift, full containment, release and stable-rest checks, plus simultaneous positive contact force from both jaw sides while the cube exceeds 90 mm in world Z. That is an absolute centre-height threshold, not a claimed 90 mm lift distance. A final 45-tick hold uses the last FLUX-issued command and adds no scripted opening. The original ACT baseline's grader is not rewritten.

The final 1.5 seconds must satisfy the established tray-footprint and height checks, release, positional spread below 1 mm, orientation spread below 1 degree and linear speed below 2 mm/s. Record exact grader fields and failures in the result page rather than judging a final screenshot.

## Timing and video interpretation

The 12 GiB resident-weight runtime produced its first prediction in 17.30 seconds and subsequent predictions in approximately 4.1 seconds. These are recorded inference measurements, not real-time control. Report exact per-episode timings from the run; model preparation and text encoding are separate costs. Physics advances at a simulated 30 Hz while inference pauses it.

Video overlays replay saved physical states. Cyan is the observed tool path. Amber points are forward kinematics of raw FLUX joint targets; they are neither model-predicted camera frames nor guaranteed future physical motion. Amber targets update once per 32 executed control ticks, or about 1.07 simulated seconds. The remaining ten target predictions are not executed. Native RGB insets show saved unannotated scene and wrist captures. Diagnostic cube coordinates never return to FLUX.

The CPU-only replay uses the actual saved 320 by 240 scene RGB image, enlarged for viewing. This is deliberately labelled as native RGB enlargement, not a new high-resolution physics render. Its overlay projection uses the recorded scene camera's optical geometry; the video adds no policy evaluation or graphics context. A separate optional renderer can produce an overview from saved states only after the inference GPU is released.

## Frozen report schema clarification

The runner's episode field `model_sha256` hashes `cell.xml`, the MuJoCo physical scene. It is not the neural checkpoint hash. Preserve those frozen reports as written and use `source_xml_sha256` for this field's meaning in presentation provenance. The neural checkpoint SHA256 is `runtime.json` → `sha256` → `model.safetensors`: `0f57664573dde660590060d81b2a0b336fc674f06e0494de9307e52f43b6fd01`. The page writes a separate clarification sidecar rather than silently editing the measured evidence.

## Keep earlier experiments separate

- The original hosted SO-101 attempt failed to place the cube. It ran for eight chunks with the hosted median statistics.
- The accelerated DROID test predicted actions from recorded observations. It did not execute those actions on SO-101.
- ACT's earlier 2/5 assessment and later 10/12 exploratory visual tests belong to a separately trained small policy.
- Stereo RGB tests use measured perception and scripted IK; they are not evidence that FLUX learned stereo control.

## Source and model attribution

- SO-101 checkpoint: [revision 2d6be04357eef9fb8c8468adbc69e9d0c7093bec](https://huggingface.co/black-forest-labs/flux-3-action-so101/tree/2d6be04357eef9fb8c8468adbc69e9d0c7093bec). Model terms: its original `LICENSE.md` (FLUX Kommunity License v1.0).
- Official inference code: [revision e2dd1d8dbc5977b54315d61f7548c63c043d6d4f](https://github.com/black-forest-labs/flux-action/tree/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f), Apache-2.0.
- Shared encoders: [base revision 62878e2925e59b7a89ec14463ce89932624c490d](https://huggingface.co/black-forest-labs/flux-3-action-base/tree/62878e2925e59b7a89ec14463ce89932624c490d).
- Hosted simulator: [revision d7da38e032da0e136d6de21c218dc616982a8cc9](https://huggingface.co/spaces/multimodalart/flux-3-action-so101-sim/tree/d7da38e032da0e136d6de21c218dc616982a8cc9). Fetch this source through the pinned bootstrap; do not redistribute simulator Python without established permission.
- Official robot geometry: [old/new calibration at revision 5f6d2b876a53a4872e405b991dd925556c9e38a4](https://github.com/TheRobotStudio/SO-ARM100/tree/5f6d2b876a53a4872e405b991dd925556c9e38a4/Simulation/SO101).
- Historical motor calibration: [LeRobot revision e004247ed42008ff309b606aa2bb7124a8ae7c41](https://github.com/huggingface/lerobot/blob/e004247ed42008ff309b606aa2bb7124a8ae7c41/lerobot/common/robot_devices/robots/feetech_calibration.py).

Model weights and shared encoders remain local. Repository code licensing does not replace their model terms. Commercial robot use requires checking those terms for the intended deployment; this experiment establishes no production deployment rights or hardware qualification.
