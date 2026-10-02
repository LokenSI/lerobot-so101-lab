"""Run exactly one anonymous public SO-101 FLUX/MuJoCo rollout, without uploads."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import traceback
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/flux-action-study/hosted-so101"
HOST = "https://multimodalart-flux-3-action-so101-sim.hf.space"
SPACE_REVISION = "d7da38e032da0e136d6de21c218dc616982a8cc9"
PARAMETERS = {
    "prompt": "Grasp the red block and place it in the gray bin",
    "scene": "Single cube",
    "n_chunks": 8,
    "seed": 0,
    "make_video": True,
}
NORMALIZATION = {
    "state": {"q01": [-37.969, -99.316, -45.78, 22.987, -68.712, 0.45], "q99": [35.288, 43.076, 90.308, 95.704, 17.753, 40.324]},
    "action": {"q01": [-1.652, -3.235, -3.147, -2.198, -1.708, 0.0], "q99": [1.701, 3.487, 3.307, 2.072, 1.699, 40.733]},
}


def save_json(name: str, data) -> None:
    (OUT / name).write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def find_files(value):
    if isinstance(value, dict):
        if value.get("url") and value.get("path"):
            yield value
        for child in value.values():
            yield from find_files(child)
    elif isinstance(value, list):
        for child in value:
            yield from find_files(child)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Refuse a second submission. Inspect the saved result instead.
    if (OUT / "submission.json").exists():
        raise RuntimeError("This experiment was already submitted; no automatic reruns are allowed")
    report = {
        "status": "not_submitted",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hosted": True,
        "local_gpu_used_for_policy": False,
        "authentication": "anonymous; no credentials or billing supplied",
        "uploads": [],
        "request": PARAMETERS,
        "space": "multimodalart/flux-3-action-so101-sim",
        "space_source_revision": SPACE_REVISION,
        "model": "black-forest-labs/flux-3-action-so101",
        "policy_precision": "BF16, as configured by published Space",
        "normalization_override": NORMALIZATION,
        "normalization_note": "The public Space overrides the released checkpoint normalization with median quantiles from new-calibration SO-10x training datasets.",
        "control": "FLUX predicts 42 commands; first 32 execute, then replan using updated simulated cameras/state and command history",
        "simulation_frequency_hz": 30,
        "planned_frames": 256,
        "planned_simulated_seconds": 256 / 30,
        "realtime_claim": False,
        "training_domain": "real SO-100/101 teleoperation; this simulator is out of distribution",
        "cost": "Public ZeroGPU anonymous free quota; no paid credentials supplied",
        "quota_duration_budget_seconds": 49,
        "quota_source": "https://huggingface.co/docs/hub/spaces-zerogpu",
        "task_success": "not_evaluated",
        "notes": ["One submission only; do not retry quota or rollout errors", "Published source uses physical contacts and free objects, with no scripted grasp weld"],
    }
    started = time.perf_counter()
    try:
        info = get_json("https://huggingface.co/api/spaces/multimodalart/flux-3-action-so101-sim")
        save_json("space-info.json", info)
        report["space_current_hub_revision"] = info.get("sha")
        report["runtime"] = info.get("runtime")
        model_info = get_json("https://huggingface.co/api/models/black-forest-labs/flux-3-action-so101")
        save_json("model-info.json", model_info)
        report["model_current_hub_revision_at_request"] = model_info.get("sha")
        report["model_runtime_revision_confirmed"] = False
        report["model_revision_note"] = "Space loads the model from main at startup; Hub revision above does not independently prove the running worker's model revision."
        save_json("request.json", {"url": HOST + "/gradio_api/call/v2/rollout", "parameters": PARAMETERS})
        save_json("submission.json", {"submitted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "attempt": 1})
        request = urllib.request.Request(
            HOST + "/gradio_api/call/v2/rollout",
            data=json.dumps(PARAMETERS).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            submission = json.load(response)
        save_json("submission-response.json", submission)
        event_id = submission["event_id"]
        report["event_id"] = event_id
        report["status"] = "streaming"
        save_json("report.json", report)
        print(f"Submitted one anonymous rollout: {event_id}", flush=True)
        events = []
        final = None
        with urllib.request.urlopen(HOST + "/gradio_api/call/rollout/" + event_id, timeout=60) as response:
            with (OUT / "response.sse").open("w", encoding="utf-8") as raw:
                event_type = None
                data_lines = []
                for encoded in response:
                    line = encoded.decode("utf-8")
                    raw.write(line)
                    raw.flush()
                    text = line.rstrip("\r\n")
                    if text.startswith("event:"):
                        event_type = text[6:].strip()
                    elif text.startswith("data:"):
                        data_lines.append(text[5:].strip())
                    elif not text and event_type:
                        data_text = "\n".join(data_lines)
                        try:
                            data = json.loads(data_text)
                        except json.JSONDecodeError:
                            data = data_text
                        events.append({"event": event_type, "elapsed_seconds": time.perf_counter() - started, "data": data})
                        save_json("events.json", events)
                        print(f"SSE {event_type} at {time.perf_counter() - started:.1f} s", flush=True)
                        if event_type == "error":
                            report["status"] = "hosted_error"
                            report["error"] = data
                            break
                        if event_type == "complete":
                            final = data
                            break
                        event_type, data_lines = None, []
        if final is not None:
            save_json("final-response.json", final)
            files = []
            for item in find_files(final):
                url = item["url"]
                if url.startswith("/"):
                    url = HOST + url
                if not url.startswith(HOST + "/"):
                    raise RuntimeError("Unexpected hosted file origin")
                suffix = Path(item["path"]).suffix.lower()
                filename = "rollout.mp4" if suffix == ".mp4" else f"model-input-{len(files)}{suffix}"
                path = OUT / filename
                with urllib.request.urlopen(url, timeout=60) as response:
                    content = response.read()
                path.write_bytes(content)
                files.append({"filename": filename, "url": url, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
            report["artifacts"] = files
            report["status"] = "completed"
            report["timing_log"] = final[-1] if isinstance(final, list) else None
            if isinstance(final, list) and final and isinstance(final[0], dict):
                save_json("all-poses.json", final[0])
                report["pose_frame_count"] = len(final[0].get("poses", []))
            print("Hosted rollout completed; saved response, poses and files", flush=True)
        elif report["status"] != "hosted_error":
            report["status"] = "stream_ended_without_result"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        (OUT / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Hosted attempt failed (will not retry): {exc}", flush=True)
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save_json("report.json", report)


if __name__ == "__main__":
    main()
