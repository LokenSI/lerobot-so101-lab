# Independent review of the SO-101 teaser experiment

Reviewed the training/evaluation scripts, pinned LeRobot ACT implementation and SO-101 environment independently from the implementation agents. This is an experiment-integrity review, not a hardware validation or a learned-policy success result.

## Findings

- **Observation timing:** inspected the actual seed-000 demonstration. It contains 525 aligned samples, two uint8 RGB arrays of shape `525 × 96 × 96 × 3`, six measured joints and six commands. Every next stored state equals the preceding telemetry’s measured joints exactly; every stored action equals the corresponding issued command exactly. Images and state are captured before that action.
- **Policy inputs:** training and evaluation pass only scene RGB, wrist RGB and measured robot joints. Simulator object coordinates, contacts, expert waypoints, task phases and grading history do not enter the model. Demonstration generation uses privileged object pose, disclosed as scripted IK; it is distinct from learned-policy inference.
- **Camera alignment:** non-rendered training demonstrations use native square 96-pixel cameras, matching policy evaluation. The separate overview renderer is for video only. Do not mix 640 × 480 video-generation observations resized to square into this dataset; their camera projection differs.
- **Normalization/checkpoints:** the manual RGB transform and train-only state/action means and standard deviations are identical during training and evaluation. The pinned ACT implementation has no extra internal normalization in this direct call path. Saved config, model and `normalization.json` form the checkpoint bundle; this is a custom simulator harness, not a tested physical LeRobot deployment. Resume checks reject changed dataset hashes, normalization and exposed architecture arguments.
- **Separation:** train/validation split is by entire episode, not adjacent frames. Evaluation asserts that its seeds overlap neither training nor validation demonstration seeds. If evaluation seeds are reused to tune a checkpoint, use fresh seeds for the final reported result.
- **Physical task grading:** the grader requires the cube to exceed 9 cm height, fit wholly inside the tray using rotated cube extents, lose all finger contacts, and remain there for the last 45 control frames. Rest requires less than 1 mm position spread, less than 1° orientation change and linear speed below 2 mm/s. The object is a free body with genuine contacts; no attachment or object-pose writes occur during the rollout. The corrected finger collision hull is disclosed in reports.
- **Negative controls:** nine synthetic grader cases passed: a valid lift/release/rest passes; no lift, insufficient history, still gripped, wall crossing, floating above the tray, slow drift, rotation and excessive linear speed fail. These test grader logic, not learned robot capability. See `grader-review-checks.json`.

Camera-aspect mismatch, unguarded resume normalization and an overly weak per-frame rest check were raised during review and corrected before the learned evaluation. Baseline reports now call the same strict grader.

## Reviewed snapshots

| Script | SHA-256 |
|---|---|
| `scripts/train_lerobot_teaser.py` | `4f4e4d8cb30f17b95912dd15eddcba5fb0dcbe76a974018ee6e482d751e5eb59` |
| `scripts/evaluate_lerobot_teaser.py` | `500f45569f8538838509151dbb440bb9532ad39ea8c07f0bf6e54fd64dd96882` |
| `scripts/so101_pick_place_baseline.py` | `30705476f46075b495d92be8f6a15b766f2032dd7a963573c51ef98a88e73188` |

Any later script or environment changes need comparison with these reviewed snapshots. Learned success rates must come from the completed closed-loop evaluation report, with all evaluated seeds included. Simulation success does not establish real-camera, physical-arm, Orin, Thor, ROS 2 or offshore performance.

## Follow-up review: longer ACT chunks and cosine learning rate

Reviewed the subsequent changes independently. `--lr-decay` affects the optimizer learning rate only, decaying to 10% through a cosine schedule. Resuming a cosine run is explicitly refused without a complete scheduler checkpoint. A 64-action prediction chunk with 32 executed actions is an ACT configuration change, not an expert-control substitution. Evaluation may explicitly override execution horizon; the effective value is recorded in its architecture report.

Policy inputs, manual normalization, free-object dynamics and the strict task grader remain unchanged. Added target-limit measurements count clipping already performed by the simulator; they do not introduce waypoints, phases, object-pose queries into the policy or manual replacement commands. Clipping is part of this simulation control pipeline and must not be confused with unconstrained raw-policy output or physical control validation.

| Updated script | SHA-256 |
|---|---|
| `scripts/train_lerobot_teaser.py` | `c3b3ce2138ba576bf841304ef2bfdc6a1c2ad344af7a1cad101f9bac2cf8e936` |
| `scripts/evaluate_lerobot_teaser.py` | `3625b05a0d1707ea17796acbb07d4c08888e3a2c53e12e11fdeffee8da654bebb` |
| Strict environment/grader, unchanged | `30705476f46075b495d92be8f6a15b766f2032dd7a963573c51ef98a88e73188` |

Verified completed report roles:

- `eval-4000`: original 4,000-step ACT checkpoint, **0/5** on seeds 100–104. Retain this unsuccessful result.
- `dev-8000`: 8,000-step checkpoint, **1/1** on seed 100. This is a development check on a reused seed and is not part of the final success-rate sample.
- `eval-8000`: that same frozen 8,000-step checkpoint, **3/5** on fresh seeds 200–204. Successful seeds 201, 203 and 204; failures 200 and 202. The recorded model SHA-256 matches the actual checkpoint, and the strict environment/grader hash matches the reviewed unchanged source.

The 64/32 cosine-run checkpoint completed 6,000 training steps, then failed its development probe. It was not selected for the final assessment; its lower supervised loss did not establish task success. Keep different checkpoints and development sets separate and preserve failures in reported denominators.

Checkpoint selection was declared before evaluating seeds 300–304: choose the 64/32 model if its development probe on seed 100 succeeds; otherwise retain the 8,000-step model. Seeds 300–304 are reserved for the selected model’s final assessment. RGB input availability alone does not establish causal reliance on vision; no camera ablation was performed. These fixed-task, same-simulator tests vary initial cube position by approximately ±15 mm and do not establish broad scene generalization.

## Completed final assessment

The longer 64/32 candidate failed its development probe on seed 100. The predeclared rule therefore selected the frozen 8,000-step 16/8 model before running the final sample. `experiment-roles.json` preserves the selection and earlier assessments.

The completed canonical `runs/lerobot-teaser/eval/report.json` records **2/5 successes on untouched seeds 300–304**: seeds 301 and 302 pass; seeds 300, 303 and 304 fail. All five episodes remain in the denominator. The report’s checkpoint SHA-256 `c38fc8fc714a09a3f7a5f920df0f353ef0cc5fd96a916358b491cb43e4a30974` matches the actual frozen model file, and the environment/grader hash remains `30705476…`. The evaluation source remains the reviewed `3625b05a…` snapshot.

Final fresh-chunk inference median is 13.42 ms, p95 25.59 ms on the desktop GPU; measured whole-device peak is 1,209 MiB. These are desktop measurements, not Orin or Thor benchmarks. Raw-policy targets were clipped by the existing simulator limits on 65 and 66 steps of the two successful episodes; the limiter is disclosed, not an expert controller.

The earlier **3/5** result on seeds 200–204 is correctly retained as a prior assessment subsequently used for development. It must not replace the final **2/5** rate or be combined with development probes to produce a stronger headline. The successful episodes support a teaser showing learned grasp/transfer/release in this simulation; the final failure rate prevents claiming reliable general-purpose manipulation or readiness for drillfloor deployment.
