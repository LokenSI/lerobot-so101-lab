# Native SO101 FLUX adaptation

After the original pretrained checkpoint completed **0/2** simulated placements, we adapted the real FLUX model using 17 demonstrations and three validation episodes. The selected 256-update adapter with original guidance scale three completed **2/3** placements on the three planned fresh starts. All three evidence audits passed. The remaining trial grasped and carried the block but did not release it within the fixed run. The original failures, the failed 128-update candidate and the separate guidance-scale-one failure remain visible.

Training uses the original Black Forest Labs FLUX 3 Action SO101 checkpoint at `2d6be04357eef9fb8c8468adbc69e9d0c7093bec`, official inference/training source at `e2dd1d8dbc5977b54315d61f7548c63c043d6d4f`, and shared encoders at `62878e2925e59b7a89ec14463ce89932624c490d`. The 13.9 GB BF16 neural checkpoint hash is `0f57664573dde660590060d81b2a0b336fc674f06e0494de9307e52f43b6fd01`. The original pretrained weights are immutable and downloaded separately using the already published pinned WSL bootstrap.

Twenty scripted contact demonstrations provide the adaptation data. They are expert data, not FLUX policy success. Their measured states and issued commands are replayed through the same contact-based simulator to obtain fresh 320×240 scene/wrist camera observations; each view is resized by uint8 PIL bilinear to 256×256. Original 96×96 images are not upsampled as model inputs. Source episodes, regenerated observations, reports, XML, replay hashes and data manifests are retained.

Seventeen training seeds are `1,2,3,4,5,6,8,9,10,11,12,13,15,16,17,18,19`. Validation seeds `0,7,14` remain separate. Each episode contributes 15 windows at action starts `8,40,…,456`: 255 training and 45 validation windows. State/action normalization quantiles use only the 17 training episodes. Arm joints use simulator degrees; the gripper uses percentage. No uncertain legacy calibration transform is used.

At action start `t`, a window contains measured state/images `[t−7,t]`, previous issued commands `[t−8,t−1]`, and future command targets `[t,t+41]`. The native visual training window extends through image `t+42`. Five arm targets are consecutive command deltas anchored on command `t−1`; the gripper remains absolute. The official processor owns this parameterization; converted deltas are not fed to it a second time. The instruction is exactly `Grasp the red block and place it in the gray bin.` including its final period.

Real frozen encoders cache the model's native text/video conditioning before optimizer updates. Training retains the original native joint video/action flow-matching loss, not a separate ACT network or waypoint regression substitute. Rank-8, alpha-8 LoRA adapts attention/MLP projections; robot action input/history and output heads are trainable. Original BF16 projection weights remain frozen. AdamW uses learning rates `1e−4` for LoRA and `5e−4` for heads, zero weight decay, batch size one, gradient norm clipping at one, and no EMA. The training schedule and flow-noise seeds are frozen; fixed validation flow noise makes loss comparisons reproducible.

For the 16 GB desktop, the measured broad training run used an 8 GiB resident frozen-base budget, with approximately 10.03 GiB peak PyTorch allocation during training updates. Encoder-cache preparation is a separate phase and this value does not describe its peak or whole-device usage. A fixed subset of frozen BF16 projection buffers resides on CUDA while the remaining buffers stay on CPU. Placement occurs before graph creation; training does not reuse inference migration hooks. An explicit frozen-linear backward preserves input gradients, activation checkpointing and saved-tensor CPU offload reduce memory, and FP32 trainable adapters/heads execute under BF16 autocast. CPU toy gradient and restoration proofs are limited source/math checks; separate real-model update reports establish that actual pretrained FLUX weights were used.

The pre-frozen evaluation plan allows a maximum of 400 optimizer updates and development candidates at 128, 256 and 400. Save optimizer/RNG and adapter state at each tested stage. Test development seed 910 in that order; continue training only if the candidate fails. Select the earliest successful development candidate, then freeze its exact adapter/config/source hashes before fresh starts 920, 921 and 922. Keep all candidate failures and fresh outcomes. The three fresh starts are a small exploratory assessment; they are not production reliability evidence and are not pooled with development or the earlier pretrained denominator.

Inference uses unmerged original BF16 bases with FP32 LoRA/full action heads under BF16 autocast, matching the training modules. Optional merged restoration is a separate arithmetic mode that records BF16 rounding. Replanning predicts 42 absolute calibrated targets and executes 32 at a simulated 30 Hz. The history records applied commands after limit clipping. No object coordinates enter FLUX, and no IK correction, scripted pickup, object attachment or live cube-pose write is permitted after reset. The final 45 ticks hold the last issued model command.

Task success requires the unchanged strict lift/release/containment/stable-rest grader plus positive contacts at both actual jaws while cube centre is above 90 mm world Z. This is a height threshold, not 90 mm of displacement. Independent CPU replay must verify every actual episode before success presentation. Flow loss, expert replay and a successful finite model output do not establish a successful robotic placement.

The model and derived adapter retain the original FLUX Kommunity terms, with their full license/attribution alongside the artifacts. Upstream Python source is Apache-2.0; this does not make the pretrained or adapted weights Apache-licensed. Base weights/encoders remain pinned external downloads. Local environments, credentials and unrelated project material are excluded.

The portable training helpers live in `vendor/flux_adaptation_runtime/`, preserving `ROOT = parents[2]`. Restore the original demonstration bundle at `runs/so101-teaser-dataset/` and the native replay data at `datasets/flux-so101-native-v1/`. Use the published FLUX source/model bootstrap and its recorded WSL Python environment:

```bash
# Rebuild native camera data when desired; this requires GPU rendering and must
# run separately from training/inference GPU ownership.
MUJOCO_GL=egl runtime/flux-action-src/.venv/bin/python vendor/flux_adaptation_runtime/replay_native.py \
  --width 320 --height 240 --output datasets/flux-so101-native-v1

# New run, stopping at the first frozen development candidate.
runtime/flux-action-src/.venv/bin/python vendor/flux_adaptation_runtime/train_broad.py \
  --native datasets/flux-so101-native-v1 --output runtime/flux-training-new \
  --stop-at 128 --planned-steps 400 --resident-base-gib 8

# Run a genuinely trained model; choose a new output directory and explicit role.
MUJOCO_GL=egl runtime/flux-action-src/.venv/bin/python scripts/run_flux_so101_adapted.py \
  --adapter runtime/flux-training-new/checkpoint-0128 --restoration unmerged \
  --role development --seeds 910 --chunks 24 --resident-gib 12 \
  --output runs/flux-so101-retry/adapted-development-new
```

Only after a failed development result, resume to the next candidate with the saved checkpoint, optimizer and RNG:

```bash
runtime/flux-action-src/.venv/bin/python vendor/flux_adaptation_runtime/train_broad.py \
  --native datasets/flux-so101-native-v1 --output runtime/flux-training-new \
  --resume runtime/flux-training-new/checkpoint-0128 --stop-at 256 \
  --planned-steps 400 --resident-base-gib 8
```

On an independently verified selected development adapter, freeze a new fresh-assessment specification and execute exactly seeds 920–922. Keep the model, prompt, sampler, history, camera processing and grader unchanged. Replay uses recorded model RGB plus CPU FK: project using physical 320×240 optics, then scale coordinates by `[256/320,256/240]` into the processed input. The display's amber targets update every 32 simulated ticks; the final ten points in each prediction are unexecuted. Playback omits model inference pauses and cannot establish real-time control or hardware performance.
