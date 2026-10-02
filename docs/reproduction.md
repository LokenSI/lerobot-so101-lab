# Reproduction and experiment boundaries

The archived outcomes are evidence from one desktop experiment, with all failures retained. The original experiment scripts are copied byte for byte except the public presentation helper, whose unrelated project prose was removed from the staged copy. `experiment/source-inventory.json` records both original and staged hashes. No training data, checkpoint or old outcome was changed during packaging.

## Install and verify

Use Python 3.12 on Linux/WSL. `scripts/bootstrap_runtime.py` downloads the two exact upstream revisions in `experiment/upstream-source-lock.json`; it does not fetch policy weights. Run `--install` to create `runtime/flux-action-sim-env` for physics/rendering and `runtime/lerobot-teaser-env` for the desktop ACT stack. Run `--check` to check already downloaded sources without network or changes.

`requirements-simulation.txt` pins the small rendering stack. `requirements-policy.txt` preserves the measured package versions while relocating the editable LeRobot path. `docs/lerobot-teaser-environment.lock.txt` is the original measured freeze, including its original absolute path. `docs/lerobot-upstream-uv.lock` and `docs/lerobot-upstream-pyproject.toml` retain the official source's broader dependency declaration; they are not the environment used by this custom trainer. The desktop CUDA lock is not an Arm64/Jetson install recipe.

Set `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` before importing MuJoCo for offscreen rendering. EGL needs an appropriate graphics driver. The new overlay renderer falls back to the system's DejaVu Sans fonts if Windows Segoe UI is unavailable. Older presentation/calibration scripts retain their original `/mnt/c/Windows/Fonts` assumptions; no Microsoft fonts are distributed. Core dataset generation, training and evaluation do not require those presentation fonts.

## Generate demonstrations

This controller uses privileged simulator object pose and Cartesian IK to generate supervised demonstrations. The object moves through contacts; there is no attachment and no object pose write during the actual control rollout. The fixed-finger collision mesh is decomposed into two convex hulls in memory; the original visible mesh, inertial properties, actuators and upstream files remain unchanged. The independent strict grader requires a lift above 9 cm and 45 consecutive control frames of full tray containment, release and stable rest.

```sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
runtime/flux-action-sim-env/bin/python scripts/so101_pick_place_baseline.py \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
  --output runs/so101-teaser-dataset --save-data
runtime/flux-action-sim-env/bin/python scripts/verify_so101_dataset.py
```

Use a clean reproduction checkout for generation: these original scripts write their fixed `runs/` paths. Keep the release bundle separately if comparing regenerated demonstrations. The twenty original episodes contain 525 control frames each at 30 Hz. Observations are recorded before the corresponding action. `images_front` maps to the scene camera; `images_wrist` maps to the wrist camera. States and actions are six absolute radians, including the gripper actuator angle. The trainer also accepts the case logger's `images_scene` name.

## Train the selected architecture

Seventeen episodes train the policy; three episodes validate supervised fit. The split is by complete episode, with seed 42, rather than adjacent-frame sampling across partitions. Input images are normalized from uint8 to [-1, 1]. State/action mean and standard deviation come from training episodes only, with a 1e-3 standard-deviation floor. Object positions, expert waypoints and phase labels never enter ACT.

The official ACT implementation uses a compact configuration: dim_model 128, four heads, two encoder layers, one decoder layer, feed-forward 512, sixteen targets and eight executed actions. VAE and dropout are disabled, and the ResNet18 backbone starts without pretrained weights. There are 11,902,150 trainable parameters. The selected model was trained for 4000 steps and resumed from model+optimizer state to 8000 steps:

```sh
runtime/lerobot-teaser-env/bin/python scripts/train_lerobot_teaser.py \
  --dataset runs/so101-teaser-dataset --output runs/lerobot-teaser/train --steps 4000
runtime/lerobot-teaser-env/bin/python scripts/train_lerobot_teaser.py \
  --dataset runs/so101-teaser-dataset --output runs/lerobot-teaser/train-8000 \
  --steps 8000 --resume runs/lerobot-teaser/train/training-state.pt
```

The archive retains all three training runs and saved optimizer states. A training resume is not a claim of bit-for-bit reproducibility across GPU drivers or library versions: the scripts set deterministic seeds but also enable TF32 and cuDNN benchmarking, and do not restore a complete RNG checkpoint. Released checkpoint bytes and their checksums are the reference for the measured results.

## Separate development from assessment

`experiment/experiment-roles.json` records the seed roles. Demonstration seeds are 0–19. Seed 100 and seeds 200–204 were used during development/model selection. The prior final evaluation used 300–304 and achieved 2/5. Those observations become known after evaluation and must not later be described as untouched. The subsequent 400–402 exploration reused three paired starting states across four visual conditions and achieved 10/12 with the frozen checkpoint. Reference 301 is a known successful reproduction and is excluded from that count.

The visual conditions are nominal, blue target, diffuse/ambient lighting multiplied by 0.75, and a render-only blue distractor with no physical collisions. Lighting scale is not calibrated lux. The full suite retains both failures: dim_light 401 lifts but remains held/moving; visual_distractor 402 ends in the tray but fails the required full release/rest window. A final screenshot alone cannot satisfy the grader.

The README's evaluation commands write new results into `runs/reproduction/`. Do not combine those new outcomes with the archived counts. Evaluation uses BF16 autocast on the desktop GPU. The recorded fresh-chunk latency includes synchronization and, for the robustness collector, prediction capture/transfer overhead. Queued action-selection timings differ from fresh sixteen-target inference. Renderer speed and video playback are not measurements of policy latency or hardware control frequency.

## Script roles

| Script | Role |
| --- | --- |
| `so101_pick_place_baseline.py` | Frozen contact environment, privileged IK demonstrations and strict independent grader |
| `verify_so101_dataset.py` | Replays demo actions and verifies RGB/state shape, units and grades |
| `train_lerobot_teaser.py` | Custom NPZ sampler feeding the official ACT implementation |
| `evaluate_lerobot_teaser.py` | Closed-loop RGB+joint evaluation; all outcomes retained |
| `lerobot_case_environment.py`, `run_lerobot_cases.py` | Frozen-policy visual variants and full action/state/RGB logger |
| `render_lerobot_overlays.py` | HD saved-state presentation, independent target FK and camera alignment check |
| `verify_lerobot_case_artifacts.py` | Verifies original suite hashes, outcomes, chunk timing and saved verification records |
| `diagnose_lerobot_act.py` | Teacher-forced fit diagnostic; not closed-loop task success |
| `verify_lerobot_act_runtime.py` | Zero-tensor import/gradient/serialization smoke check; not task success |
| `package_lerobot_replay.py`, `benchmark_lerobot_act_replay.py` | Saved-observation inference pack for later device measurements; no motor commands |
| `prepare_so101_sim.py`, `so101_kinematic_probe.py`, `so101_inspect_rollout.py` | Camera/control calibration and geometry/debug probes |
| `render_so101_baseline.py`, `render_lerobot_teaser.py`, `lerobot_case_report.py` | Recorded-state presentation helpers |
| `run_hosted_so101.py`, `analyze_hosted_so101.py` | Historical optional hosted FLUX comparison; separate from ACT measurements, dependent on a public remote service |

Some verification/presentation helpers require the original archived metadata and fixed paths rather than fresh minimal outputs. The original reports preserve hashes and absolute paths. Do not edit them to make relocation appear byte-identical; provide a separate mapping or use the new evaluation output folders. `diagnose_lerobot_act.py` directly opens original demonstration paths listed in its training report, so relocation must be handled explicitly if using the released checkpoint for that diagnostic.
