"""Verify ACT imports, gradient computation and portable checkpoint serialization.

Uses zero tensors only for a runtime check. This is not a robotics task test.
"""
import json
from pathlib import Path
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

torch.set_num_threads(6)
config = ACTConfig(
    device="cuda", push_to_hub=False,
    input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
                    **{f"observation.images.{name}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)) for name in ("scene", "wrist")}},
    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    chunk_size=16, n_action_steps=8, dim_model=128, n_heads=4, dim_feedforward=512,
    n_encoder_layers=2, n_decoder_layers=1, use_vae=False, dropout=0.0,
    pretrained_backbone_weights=None,
)
policy = ACTPolicy(config).cuda()
batch = {"observation.state": torch.zeros(2, 6, device="cuda"),
         "action": torch.zeros(2, 16, 6, device="cuda"),
         "action_is_pad": torch.zeros(2, 16, device="cuda", dtype=torch.bool),
         **{f"observation.images.{name}": torch.zeros(2, 3, 96, 96, device="cuda") for name in ("scene", "wrist")}}
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss, details = policy(batch)
loss.backward()
assert torch.isfinite(loss)
out = Path(__file__).resolve().parents[1] / "runs/lerobot-teaser/runtime-check"
out.mkdir(parents=True, exist_ok=True)
policy.save_pretrained(out)
loaded = ACTPolicy.from_pretrained(out, config=ACTConfig.from_pretrained(out)).cuda().eval()
actions = loaded.predict_action_chunk(batch)
assert actions.shape == (2, 16, 6) and torch.isfinite(actions).all()
report = {"smoke_check_only": True, "inputs": "Zero tensors for runtime/config/serialization verification; not task evaluation",
          "loss": float(loss.detach()), "parameters": sum(value.numel() for value in policy.parameters()),
          "model_bytes": (out / "model.safetensors").stat().st_size,
          "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
          "shape": list(actions.shape), "torch": torch.__version__}
(out / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
