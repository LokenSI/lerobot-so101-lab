# Reproduce the local pretrained FLUX retry

The retained development result is **0/2 placements**. Both trials completed all 24 model predictions and 45 final holding ticks. Running the experiment establishes local inference feasibility, not task success or hardware transfer. The prepared adaptation runner has no task-result evidence in this package.

Clone with real Git LFS payloads before replaying or publishing the viewer:

```bash
git lfs install
git lfs pull
git lfs fsck
```

The source inventory distinguishes original measured files from portable runners. The portable runners differ only in their import search path, using tracked `vendor/flux_retry_runtime`. This helper's `ROOT = parents[2]` resolves to the repository root. Byte-identical measured files remain in `experiment/flux-so101-retry/source-measured/`; the original frozen specification continues to identify those files. New rollouts use a new output directory and new source hashes.

Use WSL2 Ubuntu 24.04, Linux x86_64 Python 3.12, a CUDA-compatible Windows driver and the recorded RTX 5070 Ti 16 GB configuration. The original BF16 checkpoint is 13.9 GB on disk; shared text/video encoders and Python dependencies need additional disk and host RAM. These binaries are downloaded into ignored `models/` and `runtime/`, and are not included in the repository. See the [source lock](../experiment/flux-so101-retry/source-lock.json) and exact [model terms](licenses/FLUX-SO101-KOMMUNITY-LICENSE.md).

The following commands are preparation steps, not a claim that this bootstrap was executed on a clean machine:

```bash
# Existing project bootstrap fetches the pinned SO101 simulator and meshes.
python3.12 scripts/bootstrap_runtime.py
# Prints the FLUX plan without downloading or installing anything.
python3.12 scripts/bootstrap_flux_so101.py
# uv 0.12.9 must already be available; the upstream uv.lock is used unchanged.
python3.12 scripts/bootstrap_flux_so101.py --fetch-source --install
python3.12 scripts/bootstrap_flux_so101.py --download-models --check
```

Inference uses the upstream locked Torch 2.10.0+cu128 and Transformers 5.16.1 environment, plus NATTEN 0.21.6+torch2100cu128 and measured simulator dependencies. NATTEN's wheel suffix must match Torch and CUDA; the bootstrap uses the recorded official wheel index. No DROID FP8 checkpoint or quantization is substituted for SO101.

Start a **new** development run only when the GPU is free. The model loader preencodes the exact instruction set, stages resident BF16 weights, and offloads the remaining blocks to immutable original CPU storage. The 12 GiB resident budget reserves a checked additional 1.5 GiB activation margin before staging; desktop memory pressure can still prevent loading.

```bash
MUJOCO_GL=egl runtime/flux-action-src/.venv/bin/python scripts/run_flux_so101_retry.py \
  --modes hosted_median,legacy_candidate --seeds 910 --chunks 24 \
  --resident-gib 12 --output runs/flux-so101-retry/reproduction-new
```

FLUX receives eight synchronized native 320×240 scene/wrist RGB observations, eight measured joint-state vectors, eight previously issued command vectors and the instruction. The policy resizes each view to 256×256. At reset, all eight history slots repeat the first observation and measured initial command. The official checkpoint integrates joint deltas once, starting from the last command; the gripper is absolute. Each prediction contains 42 commands; only the first 32 are executed at a simulated 30 Hz. The history records actually issued commands after actuator-limit clipping. Cube truth enters reset, diagnostics and grading only.

Adapter A retains the hosted demo's published median normalization override. Adapter B retains saved processor statistics and applies a frozen, geometry-supported legacy-coordinate hypothesis. Unknown dataset motor homing offsets prevent calling B a recovered calibration. Neither changes pretrained weights. The documented fixed-finger collision-hull split comes from the earlier contact baseline and is retained consistently across these trials.

The stricter success predicate requires the existing lift/release/containment/rest grader plus positive contact at both actual jaws while cube centre exceeds 90 mm **world Z**. This is not 90 mm of vertical displacement. The final hold uses the last issued model command; there is no scripted release, object attachment or pickup correction.

Original episode `model_sha256` is a legacy field containing the **MuJoCo XML hash**. The actual neural checkpoint hash is `runtime.json.sha256["model.safetensors"]`. The [schema sidecar](../site/flux-retry/evidence/schema-clarification.json) clarifies this without altering frozen reports.

Verify byte preservation and the evidence inventory without loading the neural model:

```bash
python3.12 tools/verify_flux_package.py
```

The independent CPU checker rebuilds histories, clipping, contacts and physical step replay from recorded inputs/states. Its portable copy relocates mesh/provenance lookup paths only; the exact historical checker identified by the original reports is retained separately. The checker writes audit files, so use a disposable copy of the evidence and preserve the original run:

```bash
mkdir -p runtime/flux-audit-copy
cp -a runs/flux-so101-retry/dev-v1 runtime/flux-audit-copy/dev-v1
MUJOCO_GL=disable runtime/flux-action-src/.venv/bin/python tools/verify_flux_retry_cpu.py \
  runtime/flux-audit-copy/dev-v1
```

Replay video can be regenerated using CPU forward kinematics and recorded RGB, without rendering a new scene or stepping physics:

```bash
MUJOCO_GL=disable runtime/flux-action-src/.venv/bin/python scripts/render_saved_rgb.py \
  --episode runs/flux-so101-retry/dev-v1/hosted_median/seed-910 \
  --mesh-assets runtime/flux-action-so101-sim/assets \
  --output runtime/flux-replay-new/hosted-median
```

Amber is FK of the raw 42-command prediction; the last ten points were not executed. Cyan is the observed TCP from saved states. Native RGB insets show exact recorded inputs. Replanning happens every 32 simulated ticks; inference delays are omitted from playback, so the 30 fps video does not imply real-time model speed. The main image enlarges recorded 320×240 RGB rather than claiming a new high-resolution physics render.

The prepared `scripts/run_flux_so101_adapted.py` and `scripts/flux_adapted_inference.py` require an explicit separately trained adapter with matching base/config/normalization hashes. Their source is retained for later work; no adapter or adapted-rollout result is included here. Any later training and evaluation require their own frozen specifications, data split, timing, provenance and complete success/failure denominator.

The latest prepared adapted runner matches the replay-data preprocessing: native 320×240 RGB is resized using uint8 PIL bilinear to 256×256 before the official tiled canvas, and its default instruction includes the final period used in the expert replay. This changes only the later prepared adapter path; original pretrained run inputs/reports/sources remain byte-identical. Its default restoration keeps the original BF16 base plus FP32 LoRA/full heads under BF16 autocast; optional merged restoration reports rounding separately.

[CPU preparation proofs](../experiment/flux-so101-retry/prepared-adaptation-cpu-proofs/index.json) retain tiny algebra, mocked loader, preprocessing and placement checks. They are limited source/math evidence, not real task execution or trained placement success. Their original checker source paths are retained for provenance; later full training/evaluation packaging remains separate.
