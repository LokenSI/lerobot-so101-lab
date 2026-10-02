"""Saved-state FLUX replay; run only after the inference GPU owner releases it.

No policy is imported or run here. Raw predicted joint-target forward kinematics
is a diagnostic, never a guarantee of future physical motion or a world model
video prediction. MuJoCo mj_forward replays saved states without stepping them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TCP_LOCAL = np.array([.0071, -.000218121, -.088])
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
WIDTH, HEIGHT = 1440, 1080
INK, CYAN, AMBER = (9, 20, 31), (80, 232, 232), (255, 188, 67)
MUTED = (181, 197, 213)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def font(size, bold=False):
    for path in [Path('/mnt/c/Windows/Fonts') / ('segoeuib.ttf' if bold else 'segoeui.ttf'),
                 Path('C:/Windows/Fonts') / ('segoeuib.ttf' if bold else 'segoeui.ttf'),
                 Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def project(scene, points):
    if scene.enabletransform or scene.stereo != mujoco.mjtStereo.mjSTEREO_NONE:
        raise ValueError('Projection requires an untransformed mono scene')
    camera = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
    forward = np.asarray(camera.forward, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, camera.up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = np.asarray(points, dtype=float).reshape(-1, 3) - camera.pos
    depth = relative @ forward
    x, y = relative @ right, relative @ up
    if not camera.orthographic:
        scale = camera.frustum_near / np.maximum(depth, 1e-12)
        x, y = x * scale, y * scale
    half = float(camera.frustum_width) or WIDTH / HEIGHT * (camera.frustum_top - camera.frustum_bottom) / 2
    pixels = np.c_[WIDTH * (x - camera.frustum_center + half) / (2 * half),
                   HEIGHT * (1 - (y - camera.frustum_bottom) / (camera.frustum_top - camera.frustum_bottom))]
    visible = (depth > camera.frustum_near) & (depth < camera.frustum_far)
    visible &= np.all((pixels >= [0, 0]) & (pixels < [WIDTH, HEIGHT]), axis=1)
    return pixels, visible


def tcp(model, state):
    body = model.body('gripper').id
    return state.xpos[body] + state.xmat[body].reshape(3, 3) @ TCP_LOCAL


def inverse_policy(values, report, specification):
    if report['mode'] == 'hosted_median':
        return np.asarray(values, dtype=float).copy()
    if report['mode'] != 'legacy_candidate':
        raise ValueError('Unknown adapter; add its recorded conversion explicitly')
    affine = specification['legacy_candidate_affine']
    scale, offset = np.asarray(affine['scale']), np.asarray(affine['offset'])
    if scale.shape != (6,) or np.any(scale == 0) or offset.shape != (6,):
        raise ValueError('Invalid frozen affine conversion')
    return (np.asarray(values, dtype=float) - offset) / scale


def command_radians(commands):
    result = np.deg2rad(np.asarray(commands, dtype=float))
    # Preserve raw, unbounded gripper outputs for honest clipping diagnostics.
    result[..., -1] = -.17453 + np.asarray(commands)[..., -1] / 100 * (1.74533 + .17453)
    return result


def checked_targets(episode, report, spec, arrays, model):
    count = len(report['inference'])
    executed = spec['execute_per_chunk']
    if len(arrays['qpos_after']) != count * executed + report['completed_settling_hold_ticks']:
        raise ValueError('Saved trajectory does not match predicted and settling ticks')
    qadr = np.array([model.joint(name).qposadr[0] for name in JOINTS])
    targets, hashes = [], {}
    kin = mujoco.MjData(model)
    for chunk in range(count):
        path = episode / f'actions-policy-{chunk:03d}.npy'
        values = np.load(path, allow_pickle=False)
        if values.shape != (42, 6) or not np.isfinite(values).all():
            raise ValueError('Need 42 finite recorded FLUX predictions')
        raw = inverse_policy(values, report, spec)
        first = chunk * executed
        np.testing.assert_allclose(raw[:executed], arrays['raw_commands_sim_units'][first:first + executed], atol=1e-8, rtol=0)
        radians = command_radians(raw)
        applied = np.clip(radians[:executed], *model.actuator_ctrlrange.T)
        np.testing.assert_allclose(applied, arrays['controls'][first:first + executed], atol=1e-8, rtol=0)
        points = []
        kin.qpos[:] = arrays['qpos_before'][first]
        for command in radians:
            kin.qpos[qadr] = command
            mujoco.mj_forward(model, kin)
            points.append(tcp(model, kin))
        targets.append(points)
        hashes[path.name] = sha(path)
    return np.asarray(targets), hashes


def load_scene(path, mesh_assets):
    tree = ET.parse(path)
    compiler = tree.getroot().find('compiler')
    if compiler is None:
        compiler = ET.SubElement(tree.getroot(), 'compiler')
    compiler.set('meshdir', str(mesh_assets))
    # Relocate only this in-memory derivative; preserve measured XML bytes/hash.
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding='unicode'))
    model.vis.global_.offwidth, model.vis.global_.offheight = WIDTH, HEIGHT
    return model


def projection_check(renderer, state):
    renderer.update_scene(state, camera='overview')
    camera = mujoco.mjv_averageCamera(renderer.scene.camera[0], renderer.scene.camera[1])
    point = camera.pos + np.asarray(camera.forward) * .5
    pixels, _ = project(renderer.scene, [point])
    renderer.scene.ngeom, renderer.scene.nlight = 1, 0
    mujoco.mjv_initGeom(renderer.scene.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
                       np.full(3, .0025), point, np.eye(3).ravel(), np.array([1., 0., 1., 1.]))
    renderer.scene.geoms[0].emission = 1
    rgb = renderer.render()
    yy, xx = np.where((rgb[..., 0] > 180) & (rgb[..., 1] < 70) & (rgb[..., 2] > 180))
    if not len(xx):
        raise ValueError('Projection fiducial was not rendered')
    error = float(np.linalg.norm([xx.mean() + .5 - pixels[0, 0], yy.mean() + .5 - pixels[0, 1]]))
    if error >= 1.5:
        raise ValueError(f'Overlay projection failed: {error:.3f} px')
    return error


def render(episode, output, mesh_assets, specification):
    report_path = episode / 'report.json'
    report = json.loads(report_path.read_text())
    if not report.get('complete'):
        raise ValueError('Refuse to present an incomplete run as a completed episode')
    spec = json.loads(specification.read_text())
    with np.load(episode / 'trajectory.npz', allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    for name, value in arrays.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise ValueError(f'Nonfinite saved values in {name}')
    count = len(arrays['qpos_after'])
    np.testing.assert_allclose(arrays['qpos_after'][:-1], arrays['qpos_before'][1:], atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.diff(arrays['time_s']), 1 / spec['fps'], atol=1e-10, rtol=0)
    for key in ['scene_rgb', 'wrist_rgb']:
        if arrays[key].shape != (count, 240, 320, 3) or arrays[key].dtype != np.uint8:
            raise ValueError('Expected the native saved 320 by 240 camera images')
    model = load_scene(episode / 'cell.xml', mesh_assets)
    targets, prediction_hashes = checked_targets(episode, report, spec, arrays, model)
    state = mujoco.MjData(model)
    for tick in range(count):
        state.qpos[:] = arrays['qpos_after'][tick]
        mujoco.mj_forward(model, state)
        np.testing.assert_allclose(tcp(model, state), arrays['tcp_m'][tick], atol=2e-7, rtol=0)
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    projection_error = projection_check(renderer, state)
    output.mkdir(parents=True, exist_ok=False)
    title, medium, small = font(30, True), font(21), font(17)
    video = output / 'flux-replay.mp4'
    sources, snapshots = [], []
    physical = len(report['inference']) * spec['execute_per_chunk']
    try:
        with imageio.get_writer(str(video), fps=30, codec='libx264', quality=9,
                               macro_block_size=1, ffmpeg_params=['-movflags', '+faststart']) as writer:
            for tick in range(count):
                state.qpos[:] = arrays['qpos_after'][tick]
                state.qvel[:] = arrays['qvel_after'][tick]
                state.ctrl[:] = arrays['controls'][tick]
                state.time = arrays['time_s'][tick]
                mujoco.mj_forward(model, state)
                renderer.update_scene(state, camera='overview')
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                image = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(image)
                chunk = min(tick // spec['execute_per_chunk'], len(targets) - 1)
                offset = tick % spec['execute_per_chunk']
                trail = arrays['tcp_m'][max(0, tick - 180):tick + 1]
                pixels, valid = project(renderer.scene, trail)
                for a, b, va, vb in zip(pixels[:-1], pixels[1:], valid[:-1], valid[1:]):
                    if va and vb:
                        draw.line([tuple(a), tuple(b)], fill=CYAN, width=4)
                if tick < physical:
                    points, visible = project(renderer.scene, targets[chunk])
                    for index, (point, vis) in enumerate(zip(points, visible)):
                        if not vis:
                            continue
                        x, y = point
                        radius = 5 if index < 32 else 3
                        colour = AMBER if index < 32 else (149, 119, 75)
                        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=colour, width=2)
                        if index == offset:
                            draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline=AMBER, width=3)
                draw.rectangle((0, 0, WIDTH, 120), fill=INK)
                draw.text((24, 12), 'FLUX 3 Action / SO-101 / actual simulation replay', font=title, fill='white')
                draw.text((24, 56), f"{report['mode']} / seed {report['seed']} / {state.time:.2f} simulated s", font=medium, fill=AMBER)
                draw.text((24, 90), 'Pretrained BF16 policy / CPU offload / no additional training / inference pauses physics', font=small, fill=MUTED)
                draw.rounded_rectangle((1070, 140, 1430, 845), radius=9, fill=INK)
                for name, key, y in [('SCENE', 'scene_rgb', 190), ('WRIST', 'wrist_rgb', 495)]:
                    draw.text((1090, y - 31), name + ' / actual saved RGB / 320 x 240', font=small, fill='white')
                    image.paste(Image.fromarray(arrays[key][tick]), (1090, y))
                draw.text((1090, 765), 'These images feed the policy.', font=small, fill=CYAN)
                draw.text((1090, 795), 'Scene left / wrist right at encoding.', font=small, fill=MUTED)
                draw.rectangle((0, 875, WIDTH, HEIGHT), fill=INK)
                phase = f"Prediction {chunk + 1}/{len(targets)} / issued command {offset + 1}/32" if tick < physical else 'Settling hold / last FLUX-issued command retained'
                draw.text((24, 888), phase, font=medium, fill='white')
                draw.text((24, 927), 'Amber: RAW predicted joint-target FK (42 points / first 32 executed). Cyan: observed tool path.', font=small, fill=AMBER)
                draw.text((24, 957), 'Targets update every 32 control ticks. They do not predict actual future motion or generated camera video.', font=small, fill=MUTED)
                draw.text((24, 987), 'Recorded playback / simulated 30 Hz / no scripted pickup / free cube / diagnostic truth never fed to FLUX.', font=small, fill=MUTED)
                terminal = tick == count - 1
                status = ('FINAL STRICT GRADER: PASS' if report['success'] else 'FINAL STRICT GRADER: FAIL') if terminal else 'Strict task outcome is evaluated at the end.'
                draw.text((24, 1020), status, font=medium, fill=(146, 234, 155) if terminal and report['success'] else AMBER)
                writer.append_data(np.asarray(image))
                if tick in [0, count // 2, count - 1]:
                    image.save(output / f'frame-{tick:04d}.png')
                    snapshots.append(image.copy())
                    if tick == count // 2:
                        image.save(output / 'poster.png')
                sources.append({'video_frame': tick, 'source_tick': tick, 'simulation_time_s': float(state.time),
                                'prediction_index': chunk, 'command_offset': offset if tick < physical else None,
                                'settling_hold': tick >= physical})
    finally:
        renderer.close()
    sheet = Image.new('RGB', (2160, 540), INK)
    for i, image in enumerate(snapshots):
        sheet.paste(image.resize((720, 540)), (i * 720, 0))
    sheet.save(output / 'contact-sheet.png')
    np.savez_compressed(output / 'raw-predicted-tcp.npz', tcp_m=targets)
    (output / 'frame-provenance.json').write_text(json.dumps(sources))
    record = {'renderer_sha256': sha(__file__), 'source_report_sha256': sha(report_path),
              'source_trajectory_sha256': sha(episode / 'trajectory.npz'), 'source_xml_sha256': sha(episode / 'cell.xml'),
              'frozen_specification_sha256': sha(specification), 'prediction_file_sha256': prediction_hashes,
              'video_sha256': sha(video), 'width': WIDTH, 'height': HEIGHT, 'fps': 30, 'frames': count,
              'success': bool(report['success']), 'mode': report['mode'], 'seed': report['seed'],
              'rendering': 'Saved states only; mj_forward, no dynamics steps or policy inference',
              'overlays': 'Raw joint-target FK amber; independently verified observed TCP cyan',
              'projection_check_error_px': projection_error, 'observed_tcp_fk_checked': True,
              'raw_prediction_conversion_checked': True, 'applied_control_clipping_checked': True,
              'state_continuity_checked': True, 'model_weights_included': False}
    (output / 'provenance.json').write_text(json.dumps(record, indent=2))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mesh-assets', required=True, type=Path)
    parser.add_argument('--specification', type=Path)
    args = parser.parse_args()
    episode = args.episode.resolve()
    specification = args.specification or episode.parents[1] / 'frozen-specification.json'
    print(json.dumps(render(episode, args.output.resolve(), args.mesh_assets.resolve(), specification.resolve()), indent=2))
