"""Train the official LeRobot ACT policy on local, simulated SO-101 demonstrations.

This is an ACT experiment, not a FLUX 3 Action fine-tune. Demonstration NPZs contain
images_scene/images_wrist (RGB uint8), states/actions (absolute simulator units).
Episode-level train/validation splitting prevents adjacent-frame leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

ROOT = Path(__file__).resolve().parents[1]
REVISION = "6e1fa4faf2a42927d463591aebaa0f62c2e654b7"


def read_episode(path: Path, size: int) -> dict:
    with np.load(path) as src:
        aliases = {
            "states": ("states", "state"),
            "actions": ("actions", "action"),
            "scene": ("images_scene", "images_front", "images.scene", "scene"),
            "wrist": ("images_wrist", "images.wrist", "wrist"),
        }
        episode = {}
        for name, choices in aliases.items():
            key = next((key for key in choices if key in src), None)
            if key is None:
                raise ValueError(f"{path}: missing {name}; keys={src.files}")
            values = src[key]
            if name in ("scene", "wrist"):
                assert values.ndim == 4 and values.shape[-1] == 3
                values = np.stack([np.asarray(Image.fromarray(frame).resize((size, size), Image.Resampling.BILINEAR)) for frame in values])
                values = values.transpose(0, 3, 1, 2).copy()
            else:
                values = values.astype(np.float32)
                assert values.ndim == 2 and values.shape[1] == 6
                assert np.isfinite(values).all()
            episode[name] = torch.from_numpy(values)
    assert len({len(values) for values in episode.values()}) == 1
    episode["path"] = str(path)
    episode["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return episode


class EpisodeSampler:
    def __init__(self, episodes: list[dict], chunk: int, stats: dict, device: str):
        self.episodes, self.chunk, self.stats, self.device = episodes, chunk, stats, device
        self.index = [(e, t) for e, episode in enumerate(episodes) for t in range(len(episode["states"]))]

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        samples = [self.index[index] for index in np.random.randint(len(self.index), size=batch_size)]
        states, actions, pads, scene, wrist = [], [], [], [], []
        for e, t in samples:
            episode = self.episodes[e]
            length = len(episode["states"])
            indices = torch.arange(t, t + self.chunk)
            pads.append(indices >= length)
            actions.append(episode["actions"][indices.clamp_max(length - 1)])
            states.append(episode["states"][t])
            scene.append(episode["scene"][t])
            wrist.append(episode["wrist"][t])
        batch = {"observation.state": torch.stack(states), "action": torch.stack(actions), "action_is_pad": torch.stack(pads)}
        for key in ("observation.state", "action"):
            stat_key = "states" if key == "observation.state" else "actions"
            batch[key] = (batch[key] - self.stats[stat_key]["mean"]) / self.stats[stat_key]["std"]
        batch["observation.images.scene"] = torch.stack(scene).float().div(255).sub(0.5).div(0.5)
        batch["observation.images.wrist"] = torch.stack(wrist).float().div(255).sub(0.5).div(0.5)
        return {key: value.to(self.device) for key, value in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/lerobot-teaser/train")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--action-steps", type=int, default=8)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--lr-decay", action="store_true", help="Cosine-decay learning rates to10% for a fresh run")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    assert torch.cuda.is_available(), "Training requires CUDA; simulation is separately CPU physics"
    candidates = sorted(args.dataset.resolve().glob("**/episode*.npz"))
    files, excluded = [], []
    for path in candidates:
        source_report = path.parent / "report.json"
        if source_report.exists() and not json.loads(source_report.read_text()).get("success", False):
            excluded.append(str(path))
        else:
            files.append(path)
    assert len(files) >= 3, "Need at least 3 separate successful demonstrations"
    random.shuffle(files)
    count = max(1, round(len(files) * 0.15))
    val_files, train_files = files[:count], files[count:]
    started = time.perf_counter()
    train = [read_episode(path, args.image_size) for path in train_files]
    validation = [read_episode(path, args.image_size) for path in val_files]
    stats = {}
    for key in ("states", "actions"):
        values = torch.cat([episode[key] for episode in train])
        stats[key] = {"mean": values.mean(0), "std": values.std(0).clamp_min(1e-3)}
    if args.resume:
        prior_stats = json.loads((args.resume.parent / "normalization.json").read_text())
        prior_report = json.loads((args.resume.parent / "training-report.json").read_text())
        previous = {row["sha256"] for key in ("train_episodes", "validation_episodes") for row in prior_report[key]}
        current = {episode["sha256"] for episode in train + validation}
        assert previous == current, "Resume requires exactly the original dataset; use a new training run for changed data"
        for key, row in stats.items():
            for name, value in row.items():
                assert np.allclose(value.numpy(), prior_stats[key][name], atol=1e-6), "Resume normalization must match"
        for name in ("image_size", "chunk_size", "action_steps", "pretrained_backbone"):
            assert getattr(args, name) == prior_report["arguments"][name], f"Resume changed architecture argument {name}"
    config = ACTConfig(
        device="cuda", push_to_hub=False,
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
                        **{f"observation.images.{name}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, args.image_size, args.image_size)) for name in ("scene", "wrist")}},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        chunk_size=args.chunk_size, n_action_steps=args.action_steps,
        dim_model=128, n_heads=4, dim_feedforward=512, n_encoder_layers=2, n_decoder_layers=1,
        use_vae=False, dropout=0.0,
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1" if args.pretrained_backbone else None,
        optimizer_lr=3e-4, optimizer_lr_backbone=1e-4,
    )
    policy = ACTPolicy(config).cuda()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=config.optimizer_lr, weight_decay=1e-4)
    assert not (args.resume and args.lr_decay), "Resume cosine scheduling requires a complete scheduler checkpoint; use a fresh run"
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.1 + 0.9 * (1 + np.cos(np.pi * min(step, args.steps) / args.steps)) / 2) if args.lr_decay else None
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cuda", weights_only=False)
        policy.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
    training_sampler = EpisodeSampler(train, args.chunk_size, stats, "cuda")
    validation_sampler = EpisodeSampler(validation, args.chunk_size, stats, "cuda")
    metadata = {"policy": "Official LeRobot ACT, compact deterministic configuration (VAE disabled)",
                "lerobot_revision": REVISION, "input": "Two rendered RGB cameras and 6 joint states; no object coordinates", "action": "Six absolute joint targets in radians, including gripper actuator angle",
                "image_normalization": "RGB uint8 / 255, then channel mean0.5 and standard deviation0.5",
                "state_action_normalization": "Per-joint mean/std computed on training episodes only, floor1e-3",
                "train_episodes": [{"file": e["path"], "sha256": e["sha256"]} for e in train],
                "validation_episodes": [{"file": e["path"], "sha256": e["sha256"]} for e in validation],
                "excluded_unsuccessful_demonstrations": excluded,
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "parameters": sum(p.numel() for p in policy.parameters()), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "history": []}
    normalization = {key: {name: value.tolist() for name, value in row.items()} for key, row in stats.items()}
    (out / "normalization.json").write_text(json.dumps(normalization, indent=2))
    config.save_pretrained(out)
    for step in range(start_step + 1, args.steps + 1):
        policy.train()
        batch = training_sampler.sample(args.batch_size)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, details = policy(batch)
        assert torch.isfinite(loss), f"Nonfinite loss at step {step}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            policy.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                validation_loss = float(policy(validation_sampler.sample(args.batch_size))[0])
            row = {"step": step, "train_loss": float(loss), "validation_loss": validation_loss, "elapsed_seconds": time.perf_counter() - started, "learning_rates": [group["lr"] for group in optimizer.param_groups]}
            metadata["history"].append(row)
            print(json.dumps(row), flush=True)
            metadata["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
            (out / "training-report.json").write_text(json.dumps(metadata, indent=2))
        if step % 1000 == 0 or step == args.steps:
            policy.save_pretrained(out)
            torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, out / "training-state.pt")
    metadata["training_complete"] = True
    metadata["closed_loop_success"] = "Not evaluated by training; see evaluation report"
    metadata["model_file_bytes"] = (out / "model.safetensors").stat().st_size
    (out / "training-report.json").write_text(json.dumps(metadata, indent=2))
    (out / "report.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
