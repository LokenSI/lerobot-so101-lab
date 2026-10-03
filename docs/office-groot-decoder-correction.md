# GR00T native decoder bounds: correction and scope

This correction supplements the unchanged historical reports and contracts. Their shorthand suggesting all GR00T action groups use percentile bounds was too broad.

The actual loaded checkpoint processor uses **raw relative-action min/max for the five arm joints**. The **absolute gripper uses q01/q99 percentile bounds**. Native MIN_MAX unnormalization clips normalized values to [-1, 1] before decoding. Relative arm commands are then added to the observed current joint state; their absolute commands can still exceed physical joint limits.

The independent CPU-loaded processor audit found exact equality between the active relative-arm lower bound and cached raw relative minimum (maximum discrepancy 0). The discrepancy from q01 was 0.3123409152030945 radians. Both processor statistics and training statistics matched in all 26 audited checks. This is native decoder behavior, not a new clipping intervention or evidence that the neural action head itself is safe.

Command-limit events therefore refer to **policy-decoded commands before the common actuator supervisor**. ACT and SmolVLA native MEAN_STD outputs are unbounded. Their counts should not be interpreted as directly comparable raw neural-head compliance. Physical task success remains independently graded.

Evidence: `common-training-v1/groot-native-normalization-audit-v2.json`, especially `active_norm_parameters` and `relative_action_stats`. Pinned Isaac-GR00T source 51d4c89f72fda44cbf77285c6a8114b52676b8a1: `gr00t/data/state_action/state_action_processor.py` installs relative-action statistics after its percentile configuration, and `unapply_action` uses the resulting min/max cache; `gr00t/data/utils.py` performs normalized clipping. Original files and scores were not rewritten.
