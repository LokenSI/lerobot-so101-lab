"""Compare teacher-forced ACT predictions with held-out demonstration actions.

This diagnoses supervised fit around approach and closing. It is not a closed-loop
task-success test, and recorded future actions enter only the error calculation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    report = json.loads((args.checkpoint / "training-report.json").read_text())
    config = ACTConfig.from_pretrained(args.checkpoint)
    config.device = "cuda"
    policy = ACTPolicy.from_pretrained(args.checkpoint, config=config).cuda().eval()
    stats = json.loads((args.checkpoint / "normalization.json").read_text())
    rows = []
    for episode in report["validation_episodes"]:
        with np.load(episode["file"]) as data:
            for t in (150, 165, 180, 195, 210, 225, 240):
                state = ((data["states"][t] - stats["states"]["mean"]) / stats["states"]["std"]).astype(np.float32)
                batch = {"observation.state": torch.from_numpy(state).unsqueeze(0).cuda()}
                for name, key in (("scene", "images_front"), ("wrist", "images_wrist")):
                    frame = data[key][t]
                    normalized = (frame.transpose(2, 0, 1).astype(np.float32) / 255 - 0.5) / 0.5
                    batch[f"observation.images.{name}"] = torch.from_numpy(normalized).unsqueeze(0).cuda()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = policy.predict_action_chunk(batch).float().cpu().numpy()[0]
                predicted = prediction * stats["actions"]["std"] + stats["actions"]["mean"]
                truth = data["actions"][t:t + config.chunk_size]
                rows.append({"episode": episode["file"], "frame": t,
                             "gripper_target_prediction_first_last_min_rad": [float(predicted[0, 5]), float(predicted[-1, 5]), float(predicted[:, 5].min())],
                             "gripper_truth_first_last_min_rad": [float(truth[0, 5]), float(truth[-1, 5]), float(truth[:, 5].min())],
                             "absolute_joint_L1_rad": float(np.abs(predicted - truth).mean())})
    result = {"scope": "Teacher-forced demonstration fit; no closed-loop success claim", "checkpoint": str(args.checkpoint), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
