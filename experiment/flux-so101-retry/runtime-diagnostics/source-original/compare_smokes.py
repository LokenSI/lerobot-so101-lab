"""Compare baseline and optimized BF16 memory-placement smoke outputs."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original_dir = ROOT / "smoke-raw-v1"
    fast_dir = ROOT / "smoke-raw-fast-v1"
    original = json.loads((original_dir / "report.json").read_text())
    fast = json.loads((fast_dir / "report.json").read_text())
    if original["status"] != "complete" or fast["status"] != "complete":
        raise RuntimeError("Both predictions must complete before comparison")
    a = np.load(original_dir / "actions.npy", allow_pickle=False)
    b = np.load(fast_dir / "actions.npy", allow_pickle=False)
    same_input = original["observation_sha256"] == fast["observation_sha256"]
    same_model = original["loader"]["sha256"] == fast["loader"]["sha256"]
    same_contract = original["loader"]["config"] == fast["loader"]["config"]
    report = {
        "scope": "Offline unvalidated-calibration runtime diagnostic, no task execution",
        "same_observation": same_input,
        "same_prompt": original["prompt"] == fast["prompt"],
        "same_model_and_processor_hashes": same_model,
        "same_effective_contract_and_sampler": same_contract,
        "same_output_shape": a.shape == b.shape == (42, 6),
        "bit_identical_float32_actions": np.array_equal(a, b),
        "max_absolute_action_difference": float(np.max(np.abs(a-b))),
        "baseline_prediction_seconds": original["metrics"]["inference_seconds"],
        "optimized_prediction_seconds": fast["metrics"]["inference_seconds"],
        "prediction_speedup_ratio": original["metrics"]["inference_seconds"] / fast["metrics"]["inference_seconds"],
        "optimized_metrics": fast["metrics"],
        "optimized_whole_device_peak_mib": max(s.get("used_mib", 0) for s in fast["gpu_samples"]),
        "optimized_resident_budget_gib": fast["loader"]["resident_budget_bytes"] / 2**30,
        "baseline_report_sha256": digest(original_dir / "report.json"),
        "optimized_report_sha256": digest(fast_dir / "report.json"),
        "source_sha256": {name: digest(ROOT / name) for name in ("local_runtime.py", "local_runtime_fast.py", "smoke.py", "smoke_fast.py")},
    }
    (ROOT / "offload-comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not all((same_input, same_model, same_contract, report["same_prompt"], report["same_output_shape"], report["bit_identical_float32_actions"])):
        raise RuntimeError("Optimized inference failed exact comparison")


if __name__ == "__main__":
    main()
