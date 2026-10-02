FLUX successfully completed the simulated cube-to-tray task in two of three fresh starts. The third run genuinely grasped, lifted and carried the cube, but kept holding it instead of releasing it within the test's 24 action chunks plus 45 settling ticks. All three runs passed the independent evidence integrity audit. The earlier successful development run is reported separately.

It uses two RGB camera views, the arm's measured joint positions, its previous movement commands and one fixed instruction: "Grasp the red block and place it in the gray bin." It predicts 42 movement targets; the simulation executes 32 before asking again. Object coordinates from the simulator are used for grading, never supplied to FLUX.

Our successful setup used additional task training. We supplied 20 simulated demonstrations: 17 for training and three for validation. A rank-8 LoRA adapter and the action heads received 256 optimizer updates on the actual pretrained FLUX model; the large base stayed frozen. This is our measured recipe, not a proven minimum number of examples or training steps.

The full BF16 base runs on the 16 GB desktop GPU by keeping some immutable weights in CPU memory. No quantization was used. Inference peaked at 12.7 GiB of Torch allocation, with actual whole-device usage around 14 GiB including driver and desktop. Training peaked around 10.03 GiB of Torch allocation. Across the three fresh trials, warm predictions took a median 4.58 seconds for commands representing 1.07 seconds of simulated motion: approximately 4.3 times slower than real time. The simulator pauses while the model thinks. Physical real-time control has not been demonstrated.

The fresh results are:

- Seed 920: pass; warm prediction median 4.59 seconds.
- Seed 921: pass; warm prediction median 4.59 seconds.
- Seed 922: failed to release the cube; warm prediction median 4.57 seconds.

These are three unseen random starts for the same task, fixed instruction, camera setup, scene and tray. Cube X and Y vary within +/-15 mm of the standard start. They are not three different tasks, tests of new instructions or evidence of broad business reliability. No physical SO-101, Orin Nano Super, Thor or industrial deployment has been tested.

For business, the useful result is a concrete proof of concept: task demonstrations adapted a pretrained visual robot controller, and testing exposed a real completion failure that a polished video could hide. Physical transfer, more varied conditions, failure recovery and response time are the next questions. The JSON companion records exact measurements and source hashes.
