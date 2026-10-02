"""Visual robustness cases around the frozen contact-based SO-101 environment.

Changes affect RGB rendering only. The target, robot, tray, contacts and strict
grader are identical to the established pick-and-place experiment.
"""
from __future__ import annotations

import numpy as np
import mujoco
import so101_pick_place_baseline as baseline

ORIGINAL_MAKE_ENV = baseline.make_env
CASES = {
    "nominal": {"label": "New object starts", "seeds": [400, 401, 402]},
    "blue_target": {"label": "Blue target cube", "seeds": [400, 401, 402], "target_rgba": [.10, .25, .75, 1.]},
    "dim_light": {"label": "25% dimmer illumination", "seeds": [400, 401, 402], "light_scale": .75},
    "visual_distractor": {"label": "Visual-only blue distractor", "seeds": [400, 401, 402], "distractor_xyz_m": [.13, .12, .016], "distractor_size_m": [.015, .015, .015]},
}


def decorate_scene(env, renderer):
    """Add a render-only cube; it has no body, mass, joint or collision shape."""
    spec = env.case_config
    if "distractor_xyz_m" not in spec:
        return
    scene = renderer.scene
    assert scene.ngeom < scene.maxgeom
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                       np.asarray(spec["distractor_size_m"]), np.asarray(spec["distractor_xyz_m"]),
                       np.eye(3).ravel(), np.array([.10, .25, .75, 1.], dtype=np.float32))
    scene.ngeom += 1


def make_env(case="nominal", seed=400, width=96, height=96, offscreen=True, **kwargs):
    env = ORIGINAL_MAKE_ENV(seed, width, height, offscreen, **kwargs)
    env.case_name = case
    env.case_config = dict(CASES[case])
    if "target_rgba" in env.case_config:
        target_id = env.model.body("obj0").id
        env.model.geom_rgba[env.model.geom_bodyid == target_id] = env.case_config["target_rgba"]
    if "light_scale" in env.case_config:
        scale = env.case_config["light_scale"]
        env.model.light_diffuse[:] *= scale
        env.model.light_ambient[:] *= scale
        env.model.vis.headlight.diffuse[:] *= scale
        env.model.vis.headlight.ambient[:] *= scale
    def render(camera):
        env.renderer.update_scene(env.data, camera=camera)
        decorate_scene(env, env.renderer)
        return env.renderer.render().copy()
    env.render = render
    return env


task_success = baseline.task_success
