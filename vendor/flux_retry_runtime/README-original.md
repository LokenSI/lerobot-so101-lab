# Local FLUX SO101 inference retry

This folder contains an offline and simulator-only runtime adapter. It is not a robot hardware driver. The root experiment runner owns physics, coordinate calibration, command limits, execution, grading and outcome reporting.

The original public SO101 package is downloaded to `models/flux-action/so101` at revision `2d6be04357eef9fb8c8468adbc69e9d0c7093bec`. Original configuration, split processors, processor quantiles and model weights are retained. The model is a 13,894,196,280-byte BF16 checkpoint. Shared encoders use the already downloaded official base revision `62878e2925e59b7a89ec14463ce89932624c490d`. Runtime code is the existing official `flux-action` checkout at `e2dd1d8dbc5977b54315d61f7548c63c043d6d4f`.

The delivered DROID FP8 backend is specific to a different action geometry and is not used for SO101. This adapter stages Qwen text encoding before loading GPU policy weights, then transfers complete original BF16 transformer blocks between CPU and CUDA. It leaves the sampler, weights and processor statistics unchanged. It changes memory placement only; expect inference substantially slower than the optimized DROID benchmark.

Use WSL Ubuntu 24.04 with `runtime/flux-action-src/.venv/bin/python`. Import `local_runtime.py` from a persistent simulator process:

```python
policy, metadata = load_staged_policy([instruction], resident_gib=6)
batch = history_batch(policy, scene=scene, wrist=wrist,
                      states=states, commands=commands, prompt=instruction)
actions, metrics = predict(policy, batch)
```

Camera arrays are synchronized RGB uint8 `(8,H,W,3)` in scene/wrist order. States and previous absolute commands are finite `(8,6)` arrays in the checkpoint's calibrated dataset units. `predict` returns absolute calibrated command targets `(42,6)`. The official checkpoint owns eight-tick history, snapshots at indices 0 and 7, joint-delta integration anchored on the last command, absolute gripper, four Euler steps, shift 6.93, guidance 3 and seed 42. The parent process executes 32 actions at 30 Hz before replanning.

Call `reset_cached_policy(policy)` between episodes. Official `policy.reset()` also clears text contexts; uploading Qwen again while transformer blocks are resident can exceed memory. The wrapper preserves immutable preencoded text contexts while resetting temporal/action state. `predict` refuses unregistered prompts; preencode the exact set before GPU staging.

The first smoke input is explicitly an unvalidated raw new-calibration simulator coordinate diagnostic. A successful finite prediction establishes runtime feasibility only. It cannot establish robot task success, valid cross-calibration behaviour or generalization.

The first smoke completed with 42 × 6 finite absolute commands. First-chunk inference took 78.17 seconds with the 6 GiB permanent weight budget. PyTorch peak allocation during prediction was 7,101,794,304 bytes (6.61 GiB), and peak reservation was 7,365,197,824 bytes (6.86 GiB); whole-device samples reached approximately 8.24 GB including the desktop. Text staging peaked at 9,047,628,288 allocated bytes and took 44.15 seconds. Total hash/load/text/predict runtime was 209.74 seconds. These timings are from one uncompiled offline raw-coordinate diagnostic, not a real-time robot benchmark. See `smoke-raw-v1/report.json`, `smoke-observation.json` and the saved `actions.npy`.

MuJoCo 3.6.0, imageio 2.37.3 and MuJoCo support packages were added to the existing isolated FLUX virtual environment using `uv pip install --no-deps`. Existing Torch 2.10.0+cu128, NumPy 2.2.6 and imageio-ffmpeg 0.6.0 were retained. CPU-only import and minimal XML compilation checks passed. No Isaac Sim installation is involved.

## Exact-weight offload optimization

The separate `local_runtime_fast.py` preserves the original helper and run. It retains each offloaded block's immutable original CPU parameter/buffer storage. Before forward it uploads that block to CUDA; afterward it restores the original CPU storage references, avoiding a redundant CUDA-to-CPU weight copy. An ownership check rejects overlapping block parameters. A 12 GiB resident weight budget leaves a checked additional 1.5 GiB activation margin before staging. It uses the same API, including `reset_cached_policy`.

`smoke-raw-fast-v1` returned **bit-identical float32 commands** for the same 42 × 6 action chunk: maximum absolute difference 0.0. Input SHA256, prompt, checkpoint/processor hashes and effective sampler configuration also match the original smoke. First-chunk prediction took 16.85 seconds versus 78.17 seconds (4.64× in this single pair of runs). Peak Torch allocation was 13,483,016,192 bytes (12.56 GiB), reservation 13,698,596,864 bytes (12.76 GiB), and whole-device sampling peaked at 14,280 MiB with 1,718 MiB free. This is still an offline runtime diagnostic, with no measured task execution. See `offload-comparison.json` and `compare_smokes.py`.

Model usage terms remain the package's original `LICENSE.md`; no weights are redistributed by this helper. Code upstream is Apache-2.0. Source references:

- https://huggingface.co/black-forest-labs/flux-3-action-so101/tree/2d6be04357eef9fb8c8468adbc69e9d0c7093bec
- https://github.com/black-forest-labs/flux-action/tree/e2dd1d8dbc5977b54315d61f7548c63c043d6d4f
