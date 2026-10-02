"""Smoke real SO101 FLUX action inference, without executing any commands."""
import argparse
import json
import subprocess
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from local_runtime import history_batch, load_staged_policy, predict, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Grasp the red block and place it in the gray bin.")
    parser.add_argument("--resident-gib", type=float, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "phase": "starting", "execution": "offline only",
              "task_success_measured": False, "prompt": args.prompt,
              "observation_sha256": sha256(args.observation)}
    begun, samples, stop = time.monotonic(), [], threading.Event()

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2))

    def event(data):
        report.update(data)
        save()
        print(json.dumps(data), flush=True)

    def monitor():
        while not stop.is_set():
            try:
                gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits"], text=True, timeout=8)
                used, free, util = map(int, gpu.strip().splitlines()[0].split(","))
                sample = {"elapsed_seconds": time.monotonic()-begun, "used_mib": used,
                          "free_mib": free, "util_percent": util, "phase": report["phase"]}
                samples.append(sample)
                (args.output / "heartbeat.json").write_text(json.dumps(sample))
            except Exception as error:
                samples.append({"error": str(error)})
            stop.wait(1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        policy, metadata = load_staged_policy([args.prompt], resident_gib=args.resident_gib, progress=event)
        report["loader"] = metadata
        with np.load(args.observation, allow_pickle=False) as observation:
            batch = history_batch(policy, **{key: observation[key] for key in ("scene", "wrist", "states", "commands")}, prompt=args.prompt)
        event({"phase": "predict"})
        actions, metrics = predict(policy, batch)
        np.save(args.output / "actions.npy", actions, allow_pickle=False)
        report.update(status="complete", metrics=metrics, actions_shape=list(actions.shape),
                      actions_min=actions.min(axis=0).tolist(), actions_max=actions.max(axis=0).tolist(),
                      actions_sha256=sha256(args.output / "actions.npy"), phase="finished")
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        stop.set()
        thread.join(timeout=10)
        report["gpu_samples"] = samples
        report["wall_seconds"] = time.monotonic()-begun
        save()
        print(json.dumps({key:report.get(key) for key in ("status", "phase", "metrics", "error")}), flush=True)


if __name__ == "__main__":
    main()
