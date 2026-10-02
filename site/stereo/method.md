# Virtual stereo RGB contact baseline

This new experiment adds an ideal, calibrated virtual stereo pair to the SO-101 MuJoCo contact environment. It is a marked-object sparse stereo measurement feeding a scripted controller. It does not change or evaluate the earlier ACT model: the original ACT evaluation remains 2/5, and the later visual-condition evaluation remains 10/12.

The frozen specification was written before trial outcomes: nominal and shifted starts (+20 mm X, +25 mm Y), each at seeds 810 and 811. Four motion trials passed the original strict lift, release, complete tray containment and 1.5-second stable-rest criterion. Unmarked and real left-camera occlusion cases both rejected localization and made no grasp movement. These two abstentions are separate perception negatives, not grasp successes. For the unmarked case, the runner hides the marker by setting its runtime geometry alpha to zero after saving the scene XML. The original and portable XML therefore preserve the scene setup before that visibility toggle. Reconstructing the unmarked RGB requires applying the condition recorded in report.json; XML alone is insufficient. The saved initial PNGs are the actual post-toggle camera captures, and no motion trace is generated for an abstention.

## Image-derived measurement

Both generic virtual cameras render 640×480 RGB, with 42-degree vertical field of view, 625.2213755 pixel focal length and 60 mm baseline. They look vertically down with identical orientation; the native images are rectified ideal pinhole views, with zero lens distortion. Calibration matrices and world extrinsics are stored in every report.

A small magenta fiducial is drawn on the red cube's top face. Connected-component color segmentation uses RGB only; it does not read simulator IDs, segmentation masks, depth or object pose. One component must appear in each image. Positive disparity, a 1.5-pixel epipolar gate and the calibrated work-volume bound must pass. Depth is f×baseline/disparity; the known marker offset supplies the cube center. Five independent known-position render probes measured at most 1.425 mm 3D error and 0.0833 pixel epipolar residual. Initial localization errors for the four motion trials ranged from 0.074 to 0.656 mm. Swapped camera order, a 16-pixel vertical mismatch and blank RGB pairs were correctly rejected.

This method depends on a known colored marker, its top-face offset, calibration and the cube's fixed yaw prior. It is not dense depth, general object detection or textureless cube reconstruction. Occlusion, repeated similar markers, lighting/color changes and real lens/calibration errors require separate testing. The color/epipolar rejection gates are fixed before the motion trial outcomes. Rendering markers has no contact geometry, collision or object attachment.

## Control and independent grading

Only the initial valid RGB-derived XY estimate enters pickup waypoints. Grasp Z and the tray target are fixed task priors; Z localization is diagnosed but is not used to modify contact geometry. The controller does not receive the diagnostic object truth saved in reports. During execution, RGB is captured at 10 Hz and freshly localized for presentation; the scripted controller continues from its initial estimate when the marker becomes occluded.

The cube remains a MuJoCo free body. Only robot actuators are commanded after reset; there is no weld, equality attachment or object pose write during motion. The existing SO-101 baseline provides the collision decomposition and IK. The runner records full qpos/qvel before and after each 30 Hz control tick, physical controls, fresh TCP, phase, target references, actual 640×480 RGB pairs and named cube contact pairs. The independent verification replays all saved states without integrating physics, reproduces all RGB measurements, verifies bilateral jaw contact during lift, and recomputes final release and rotated-cube containment. Ground truth is used only for reset setup, grading and diagnostics.

## Reproduce

From the lerobot-so101-lab repository root on Linux or WSL, use the existing pinned-source bootstrap and its simulation interpreter. The simulator's Space `sim.py` has no general code license and is downloaded from the existing pinned official source rather than redistributed here. Meshes follow the existing repository's third-party notices. These new scripts reuse the existing immutable baseline; no third-party simulator code or new model weights are redistributed by this experiment.

```sh
python3.12 scripts/bootstrap_runtime.py --install
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl runtime/flux-action-sim-env/bin/python scripts/so101_stereo_rgb.py --base-repo . --output runs/so101-stereo-new
runtime/flux-action-sim-env/bin/python scripts/verify_so101_stereo.py --base-repo . --run runs/so101-stereo-new
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl runtime/flux-action-sim-env/bin/python scripts/render_so101_stereo.py --base-repo . --episode runs/so101-stereo-new/shifted_start/seed-810 --output runs/so101-stereo-new/presentation/shifted-seed-810
```

The runner refuses to overwrite a frozen output. The retained first setup attempt stopped before motion because SciPy was unavailable; the final implementation uses a dependency-free connected-component pass. The successful frozen experiment is `runs/so101-stereo-v1`. Portable scene XML copies alter only the compiler mesh search path and resolve the public repository bootstrap assets. Original graded XML and its hash are preserved. The renderer resolves those same assets through `--base-repo`.

No new ACT training was performed. A future independent stereo ACT experiment needs newly recorded stereo demonstrations, a separate training checkpoint and untouched seeds/lighting/marker-occlusion holdouts. Existing scene+wrist ACT weights must not be relabeled as a validated stereo policy. No hardware, Orin deployment, ROS or Isaac execution is claimed.
