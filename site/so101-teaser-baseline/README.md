# SO-101 simulation baseline

**Twenty demonstrations passed a physical lift, transfer, release and rest test.** This controller uses scripted Cartesian IK and the simulator's initial cube position. It is the demonstration generator for the separate LeRobot ACT experiment; it is not a successful FLUX inference result.

The simulated arm lifts a 30 mm, 20 g red cube into the gray tray. It drives the six existing position actuators. There is no attachment, weld, teleport or object-state write after reset. Contact friction supports the grasp.

| Check | Result |
| --- | --- |
| Corrected collision model, reset seeds 0–9 | 10/10 successful |
| Original collision model, same waypoint script and grader, seeds 0–9 | 1/10 successful |
| Demonstrations, reset seeds 0–19 | 20/20 successful |
| Additional baseline-only evaluation scenes, seeds 100–104 | 5/5 successful |
| Independent replay of all saved action/state pairs | Exact agreement: maximum joint-state error 0 radians |
| Each episode | 525 frames, 30 Hz, 17.5 simulated seconds |

Reset varies cube x and y independently by ±15 mm. The cube orientation, cameras, lighting and physical parameters stay fixed. These are small-scope simulator results, not hardware performance or offshore validation.

## Collision correction

The community model uses the entire concave fixed finger and motor bracket as one convex collision mesh. That convex hull fills part of the gripper's free space. The test code replaces that one collision geometry with two overlapping hulls drawn from the original body-local mesh vertices: one uses vertices below z = −0.055 m and the other above z = −0.060 m. The visual mesh, mass/inertia, joints, actuator limits and position gains remain unchanged. Pinned source files are unchanged; the corrected model is assembled in memory by `make_env`.

This is an approximate collision decomposition, not a calibrated digital twin of the office arm. The original versus corrected comparison uses the same trajectory generator and final grader. Earlier development checks used cube-center containment; the results in the table use the stricter final full-cube check.

## Success criterion

The cube's center must exceed 90 mm above the board during the episode. During all final 45 frames (1.5 s), the entire rotated cube must lie inside the tray, with its bottom within 1 mm of the tray floor and no gripper or finger contacts. Position spread must be below 1 mm, orientation change below 1 degree, and linear speed below 2 mm/s. An independent agent checked acceptance/rejection cases for the grader in `../robotics-teasers/grader-review-checks.json`.

## Artifacts

- `scripted-pick-place.mp4`: clearly labeled 960×720 presentation replay of seed 0's actual physics states.
- `contact-sheet.png`: approach, grip, transfer and released-result frames.
- `video-provenance.json`: source telemetry and video hashes.
- `summary.json`, `seed-*/report.json`, `seed-*/telemetry.json`: ten-run baseline evaluation.
- `../so101-teaser-baseline-original/summary.json`: the identical test using the original geometry.
- `../so101-teaser-dataset/manifest.json`: twenty verified demonstrations and their hashes.

The dataset contains two **natively rendered 96×96 RGB** cameras, six joint states and six absolute actuator targets in radians. Each image/state pair is captured immediately **before** its matching action. The `images_front` array comes from the simulator's `scene` camera; `images_wrist` comes from `wrist`. The presentation replay is separate and is never used as training input.

## Reproduce

Run from the workspace in WSL Ubuntu, with `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`:

```sh
runtime/flux-action-sim-env/bin/python scripts/so101_pick_place_baseline.py --seeds 0,1,2,3,4,5,6,7,8,9
runtime/flux-action-sim-env/bin/python scripts/verify_so101_dataset.py
runtime/flux-action-sim-env/bin/python scripts/render_so101_baseline.py
```

The simulator is MuJoCo 3.6.0. The source is [multimodalart/flux-3-action-so101-sim](https://huggingface.co/spaces/multimodalart/flux-3-action-so101-sim), revision `d7da38e032da0e136d6de21c218dc616982a8cc9`. Its README attributes robot meshes to [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) under Apache-2.0.

- Baseline/geometry/grader script SHA-256: `30705476f46075b495d92be8f6a15b766f2032dd7a963573c51ef98a88e73188`.
- Original `so101.xml` SHA-256: `011c5fb06c999f5207d2ad6a331ba6e602cf0b29edf120097016d23cfd8eaeaf`.

Recorded on 2 October 2026. Learned-policy evaluation is reported separately.
