"""Exact BF16 SO101 FLUX inference with staged encoders and block CPU offload.

This is a simulator/offline adapter, never a hardware driver. Original model,
processor statistics, sampler and embodiment remain unchanged. CPU offload
changes placement only. The accelerated DROID backend is deliberately unused.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

REVISION = "2d6be04357eef9fb8c8468adbc69e9d0c7093bec"
SOURCE_REVISION = "e2dd1d8dbc5977b54315d61f7548c63c043d6d4f"
ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_staged_policy(prompts, *, resident_gib=12.0, progress=None):
    """Return official policy plus placement report; cache all allowed prompts.

    Each offloaded transformer block moves to CUDA immediately before its own
    forward and returns to CPU immediately after. Activations stay on CUDA.
    Don't call policy.to() afterward, because that would undo the staging.
    """
    import torch
    from flux_action.inference import so101
    from flux_action.models.text_encoder import load_text_encoder
    from flux_action.models.video_vae import load_video_vae
    from flux_action.models.transformer import ModeBlock, SingleStreamBlock

    def event(phase, **kwargs):
        if progress:
            progress({"phase": phase, **kwargs})

    base = ROOT / "models/flux-action/base"
    package = ROOT / "models/flux-action/so101"
    metadata = {
        "model_revision": REVISION,
        "source_revision": SOURCE_REVISION,
        "precision": "bfloat16 (original weights)",
        "strategy": "staged text encoder, original BF16 immutable CPU-reference block offload",
        "sampling_overrides": {},
        "processor_overrides": {},
        "resident_budget_bytes": int(resident_gib * 2**30),
    }
    event("hash_package")
    metadata["sha256"] = {
        path.name: sha256(path) for path in sorted(package.iterdir())
        if path.is_file() and path.suffix in (".json", ".safetensors", ".md")
    }
    event("load_cpu_encoders")
    encoder = load_text_encoder(str(base / "text_encoder"), compile_model=False)
    vae = load_video_vae(str(base / "video_vae.safetensors"), compile_model=False)
    event("load_cpu_policy")
    real_constructor = so101.FluxActionPolicy
    # Inject the same already downloaded pinned shared encoders. Saved config
    # remains original; strict loader verifies all saved processor statistics.
    with patch.object(so101, "FluxActionPolicy", lambda config, **kw:
                      real_constructor(config, video_vae=vae, text_encoder=encoder, **kw)):
        policy = so101.load_policy(package)
    metadata["config"] = policy.config.to_dict()
    metadata["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                           "gpu": torch.cuda.get_device_name()}
    free, total = torch.cuda.mem_get_info()
    metadata["free_before_cuda_bytes"] = free
    if free < 11 * 2**30:
        raise RuntimeError("Need at least 11 GiB free before staged Qwen text encoding")
    torch.cuda.reset_peak_memory_stats()
    policy.set_text_encoder_offload(True)
    event("preencode_text")
    started = time.perf_counter()
    # Official context API applies SO101 fixed-length 320 token encoding and
    # stores (caption, 320) cache keys, including encoded padding tokens.
    for prompt in sorted(set(prompts) | {""}):
        policy._context(prompt, torch.device("cuda"))
    torch.cuda.synchronize()
    metadata["text_encoding_seconds"] = time.perf_counter() - started
    metadata["peak_text_bytes"] = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    free_after_text, _ = torch.cuda.mem_get_info()
    metadata["free_after_text_bytes"] = free_after_text
    metadata["activation_headroom_bytes"] = int(1.5 * 2**30)
    if free_after_text < metadata["resident_budget_bytes"] + metadata["activation_headroom_bytes"]:
        raise RuntimeError("Insufficient free CUDA memory for requested resident budget plus 1.5 GiB activation headroom")
    event("stage_cuda_modules")
    # Keep small inputs/heads/modulations plus as many complete transformer
    # blocks as fit the requested permanent GPU weight budget.
    blocks = [(name, m) for name, m in policy.dit.named_modules()
              if isinstance(m, (ModeBlock, SingleStreamBlock))]
    owned_parameters = set()
    for name, module in blocks:
        for parameter in module.parameters():
            if id(parameter) in owned_parameters:
                raise RuntimeError(f"Overlapping offload block parameter ownership at {name}")
            owned_parameters.add(id(parameter))
    block_names = {name for name, _ in blocks}
    vae.module.to("cuda")
    resident_bytes = sum(p.numel() * p.element_size() for p in vae.module.parameters())
    for name, module in policy.dit.named_children():
        if not any(b == name or b.startswith(name + ".") for b in block_names):
            module.to("cuda")
            resident_bytes += sum(p.numel() * p.element_size() for p in module.parameters())
    resident, offloaded, handles = [], [], []
    retained_cpu = {}

    def upload(module, inputs):
        module.to("cuda")

    def unload(module, inputs, output):
        # Weights are immutable under inference. Restore retained original CPU
        # storage instead of copying identical CUDA weights back to CPU. CUDA
        # allocator tracks outstanding kernels on the current stream before
        # reusing the released GPU allocations.
        parameters, buffers = retained_cpu[id(module)]
        for parameter, original in parameters:
            parameter.data = original
        for parent, key, original in buffers:
            parent._buffers[key] = original

    for name, module in blocks:
        nbytes = sum(p.numel() * p.element_size() for p in module.parameters())
        if resident_bytes + nbytes <= metadata["resident_budget_bytes"]:
            module.to("cuda")
            resident_bytes += nbytes
            resident.append(name)
        else:
            parameters = [(p, p.detach()) for p in module.parameters()]
            buffers = [(child, key, tensor) for child in module.modules()
                       for key, tensor in child._buffers.items() if tensor is not None]
            retained_cpu[id(module)] = (parameters, buffers)
            handles.append(module.register_forward_pre_hook(upload))
            handles.append(module.register_forward_hook(unload, always_call=True))
            offloaded.append(name)
    policy._local_offload_handles = handles
    policy._local_retained_cpu = retained_cpu
    metadata.update(resident_blocks=resident, offloaded_blocks=offloaded,
                    resident_weight_bytes=resident_bytes,
                    retained_cpu_bytes=sum(t.numel()*t.element_size()
                                           for parameters, buffers in retained_cpu.values()
                                           for t in ([t for _, t in parameters] + [t for _, _, t in buffers])),
                    model_parameter_count=sum(p.numel() for p in policy.dit.parameters()))
    # DIT first parameter is resident embeddings; policy device is CUDA.
    if policy.device.type != "cuda":
        raise RuntimeError("Policy first parameter did not land on CUDA")
    policy.eval()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    metadata["allocated_after_staging_bytes"] = torch.cuda.memory_allocated()
    event("ready", offloaded_blocks=len(offloaded), resident_blocks=len(resident),
          allocated_bytes=metadata["allocated_after_staging_bytes"])
    return policy, metadata


def history_batch(policy, *, scene, wrist, states, commands, prompt):
    """Validate measured eight-tick window; no units/calibration guessed here.

    Images are RGB uint8 (8,H,W,3). State and command arrays are (8,6) in the
    checkpoint's calibrated dataset units. Root simulator owns conversions.
    """
    import numpy as np
    import torch
    expected = (policy.config.n_obs_steps, policy.config.action_dim)
    batch = {"task": [prompt]}
    for key, array in (("state", states), ("command_history", commands)):
        array = np.asarray(array)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"Invalid {key}: need {expected} finite calibrated values")
        batch[key] = torch.from_numpy(array.astype(np.float32)).unsqueeze(0).to("cuda")
    for key, array in (("images.scene", scene), ("images.wrist", wrist)):
        array = np.asarray(array)
        if array.dtype != np.uint8 or array.ndim != 4 or array.shape[0] != expected[0] or array.shape[-1] != 3:
            raise ValueError(f"Invalid {key}: need uint8 RGB eight-frame window")
        batch[key] = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).unsqueeze(0).to("cuda")
    return batch


def predict(policy, batch):
    import torch
    for prompt in batch["task"]:
        if (prompt, policy.config.text_fixed_length) not in policy._ctx_cache:
            raise ValueError("Prompt was not preencoded; reset_cached_policy preserves staged contexts")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    begun = time.perf_counter()
    with torch.inference_mode():
        actions = policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begun
    expected = (1, policy.config.chunk_size, policy.config.action_dim)
    if tuple(actions.shape) != expected or not torch.isfinite(actions).all():
        raise RuntimeError("Invalid FLUX action chunk")
    return actions.float().cpu().numpy()[0], {
        "inference_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def reset_cached_policy(policy):
    """Reset between episodes while retaining immutable preencoded instructions."""
    contexts = dict(policy._ctx_cache)
    policy.reset()
    policy._ctx_cache.update(contexts)
