"""Verify saved case results, overlay provenance and dynamic prediction timing."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/lerobot-cases"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = read(OUT / "integrity-before.json")
    after = {name: sha(ROOT / name) for name in before}
    assert before == after, "Frozen evaluation or policy changed"
    (OUT / "integrity-after.json").write_text(json.dumps(after, indent=2))
    suite = read(OUT / "report.json")
    assert suite["complete"] and len(suite["episodes"]) == 12
    assert sum(ep["success"] for ep in suite["episodes"]) == 10
    for case, successes in [("nominal", 3), ("blue_target", 3), ("dim_light", 2), ("visual_distractor", 2)]:
        eps = [ep for ep in suite["episodes"] if ep["case"] == case]
        assert sorted(ep["seed"] for ep in eps) == [400, 401, 402]
        assert sum(ep["success"] for ep in eps) == successes
    assert read(OUT / "telemetry-validation.json")["checks_passed"]
    overlay_root = OUT / "overlays"
    manifest = read(overlay_root / "report.json")
    assert manifest["complete"] and len(manifest["clips"]) == 5
    tcp_validation = read(overlay_root / "tcp-consistency-validation.json")
    assert tcp_validation["complete"] and tcp_validation["overlay_frames_checked"] == 2630
    checks = []
    for clip in manifest["clips"]:
        folder = (overlay_root / clip["provenance"]).parent
        provenance = read(folder / "provenance.json")
        for key, path in [
            ("source_report_sha256", ROOT / provenance["source_report"]),
            ("trajectory_sha256", ROOT / provenance["source_trajectory"]),
            ("video_sha256", folder / "overlay.mp4"),
            ("overlay_record_sha256", folder / "overlay-record.json"),
        ]:
            assert sha(path) == provenance[key], str(path)
        for name, digest in provenance["snapshots"].items():
            assert sha(folder / name) == digest
        assert provenance["projection_validation"]["max_error_px"] < 1.5
        records = read(folder / "overlay-record.json")
        trajectory = np.load(ROOT / provenance["source_trajectory"])
        fk_check = next(item for item in tcp_validation["results"]
                        if item["episodepath"] == clip["episodepath"])
        assert fk_check["trajectory_sha256"] == provenance["trajectory_sha256"]
        assert fk_check["overlay"]["overlay_record_sha256"] == provenance["overlay_record_sha256"]
        assert fk_check["overlay"]["observed_vs_fresh_saved_qpos_fk_max_error_m"] < 1e-12
        assert len(records) == len(trajectory["qpos_before"]) + 1 == 526
        updates = 0
        for i, record in enumerate(records[:-1]):
            ri = int(trajectory["frame_replan_index"][i])
            offset = int(trajectory["frame_chunk_offset"][i])
            assert record["replan_index"] == ri and record["chunk_offset"] == offset
            assert np.allclose(record["future_tcp_world_m"], trajectory["chunks_tcp_target_m"][ri], atol=2e-6)
            # npz TCP values preserve mj_step's one-substep-old cached transforms.
            # Independent MuJoCo validation above checks the trail against fresh
            # FK of authoritative qpos, including the actual terminal state.
            assert record["rgb_source_tick"] == i
            assert not record["terminal_state"]
            assert record["replan_changed"] == (offset == 0)
            updates += int(record["replan_changed"])
        assert updates == 66 and records[-1]["terminal_state"]
        assert np.max(np.abs(trajectory["chunks_tcp_target_m"][1] - trajectory["chunks_tcp_target_m"][0])) > 1e-5
        checks.append({"case": clip["case"], "seed": clip["seed"], "replans": updates,
                       "source_hashes_match": True, "observed_and_predicted_points_match": True,
                       "prediction_updates_are_dynamic": True})
    result = {"passed": True, "frozen_assets_unchanged": True, "fresh_trials": 12,
              "successful": 10, "shared_initial_positions": 3, "annotated_clips": checks}
    (OUT / "artifact-verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
