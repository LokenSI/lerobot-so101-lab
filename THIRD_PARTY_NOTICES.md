# Third-party source, meshes and fonts

This repository contains the experiment's scripts and rendered evidence. It does not vendor external runtime source, simulator XML/STL assets, virtualenvs or font binaries. `scripts/bootstrap_runtime.py` retrieves the pinned official inputs when requested and checks `experiment/upstream-source-lock.json`.

## Official LeRobot

LeRobot is developed by Hugging Face and its contributors. The measured source revision is `6e1fa4faf2a42927d463591aebaa0f62c2e654b7`, from [huggingface/lerobot](https://github.com/huggingface/lerobot/tree/6e1fa4faf2a42927d463591aebaa0f62c2e654b7). Its exact upstream LICENSE was retrieved and checked against the source used in the experiment: **Apache License 2.0**. A complete copy is retained in [docs/licenses/LEROBOT-LICENSE.txt](docs/licenses/LEROBOT-LICENSE.txt). The upstream dependency declaration and lockfile retain their original notices and bytes. The local ACT model was trained in this experiment; it is not an upstream pretrained policy release.

## SO-101 robot model and meshes

The pinned [FLUX 3 Action SO-101 simulator Space](https://huggingface.co/spaces/multimodalart/flux-3-action-so101-sim/tree/d7da38e032da0e136d6de21c218dc616982a8cc9) attributes its robot model and decimated meshes to [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100), identifies `so101_new_calib_camera.xml`, and states Apache-2.0. The official robot repository's complete **Apache License 2.0** text was also retrieved, at license-evidence revision `5f6d2b876a53a4872e405b991dd925556c9e38a4`, and is retained in [docs/licenses/SO-ARM100-LICENSE.txt](docs/licenses/SO-ARM100-LICENSE.txt). The pinned Space does not identify the exact original robot mesh commit; the actual XML and all 18 downloaded source/asset files are instead locked by their measured hashes and Space revision.

At runtime, the experiment replaces one fixed-finger collision hull with two overlapping convex hulls derived from the same original vertices. This happens in memory in `so101_pick_place_baseline.py`; the visible robot meshes, inertial properties, actuators and pinned upstream files remain unchanged. Robot imagery in the viewer and videos is rendered from this attributed model.

## Space runtime code license gap

At revision `d7da38e032da0e136d6de21c218dc616982a8cc9`, the public Space has no LICENSE/NOTICE file and no license value in its repository metadata. Its model/mesh attribution does not establish a general license for `sim.py` or the other application code. That code is **not redistributed here**. Bootstrap downloads the required files directly from the official pinned Space into ignored local `runtime/` paths. No FLUX model weights or hosted-service implementation is packaged. The license inspection URLs, revisions and hashes are recorded in [docs/licenses/license-evidence.json](docs/licenses/license-evidence.json), checked 2026-10-02.

## Rendering dependencies and fonts

MuJoCo, NumPy, Pillow, ImageIO, PyOpenGL, PyTorch and other installed dependencies retain their own upstream licenses. They are installed into local environments rather than redistributed in this source package. No external photos, logos, textures or screenshots from unrelated projects are included by the source packaging step. Scene imagery comes from the simulator and its attributed robot assets.

Original presentation scripts use Segoe UI supplied by the Windows host; no Microsoft font files are bundled. The new overlay renderer can use the Linux installation's DejaVu Sans fallback. Font binaries remain part of the user's licensed operating-system/packages rather than this repository. License evidence for upstream inputs is separate from any license chosen for the experiment's own code.
