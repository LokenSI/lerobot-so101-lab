# Same learned policy, four visual conditions

The frozen local LeRobot ACT checkpoint completed **10 of 12 exploratory MuJoCo rollouts**. This is **three paired cube starts (seeds 400–402) across four conditions**, not 12 independent starts. No model training or tuning occurred during this suite. The prior untouched benchmark remains **2/5** and is not superseded by these exploratory cases.

| Condition | Passed | Change |
|---|---:|---|
| Original scene | 3/3 | New initial cube positions |
| Blue target | 3/3 | Target colour only; same layout |
| Lower light | 2/3 | Diffuse/ambient simulator light scaled to 0.75; not calibrated lux |
| Visual distractor | 2/3 | Extra blue cube rendered without physical collisions |

All cases use the same strict grader: lift the cube above 9 cm, contain it wholly inside the tray, release it and remain stable through the last 45 control frames. The scene and wrist RGB inputs are 96 × 96, with six measured joints; object coordinates and task phases do not enter the learned policy. Existing simulator actuator limits remain active.

**Lower light, seed 401:** the cube lifts but remains held and moving; the complete containment/release/rest window fails. **Visual distractor, seed 402:** the cube ends in the tray, but fails the required final 1.5-second window. A successful-looking final frame is not enough to pass.

The annotated replay overlays amber forward-kinematics projections of the 16 predicted ACT joint targets and the cyan actual TCP trail. The prediction refreshes every eight executed ticks (0.267 simulated seconds). It is a kinematic projection, not a validated dynamic forecast or predicted object trajectory. Insets show the actual logged policy RGB inputs; annotated videos replay simulator telemetry rather than a live hardware stream. Some episodes have raw video only. Seed 301 is a previous successful reference, excluded from the 12-rollout count.

This fixed-task exploration identifies useful physical tests after the cameras arrive: lighting changes, colour changes, scene distraction and reliable release confirmation. It does not establish general visual understanding, physical transfer or production reliability. Orin, Thor, ROS 2 and Omniverse are untested here.

Evidence: [interactive viewer](index.html), [suite report](report.json), [telemetry validation](telemetry-validation.json), [overlay manifest](overlays/report.json), [artifact verification](artifact-verification.json), [TCP consistency validation](overlays/tcp-consistency-validation.json), [earlier experiment roles](../lerobot-teaser/experiment-roles.json).
