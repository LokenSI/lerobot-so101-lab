"""CPU-only FLUX replay using saved RGB plus independently computed target FK.

No GPU context, simulator render, model load or inference is started. Run with
MUJOCO_GL=disable. Actual processed 256x256 model input is enlarged for readability; camera projection remains physical320x240 before PIL resizing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from render_flux_adapted_helpers import (AMBER, CYAN, HEIGHT, INK, MUTED, WIDTH,
                               checked_targets, font, load_scene, sha, tcp)


def project_native(model, state, points):
    cid = model.camera('scene').id
    relative = np.asarray(points, dtype=float).reshape(-1, 3) - state.cam_xpos[cid]
    optical = relative @ state.cam_xmat[cid].reshape(3, 3)
    depth = -optical[:, 2]
    focal = 120 / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2)
    pixels = np.c_[160 + focal * optical[:, 0] / np.maximum(depth, 1e-12),
                   120 - focal * optical[:, 1] / np.maximum(depth, 1e-12)]
    valid = (depth > .005) & np.all((pixels >= [0, 0]) & (pixels < [320, 240]), axis=1)
    # The saved model pixels are an anisotropic PIL resize of physical320x240.
    # Project in the original optical image first, then scale coordinates.
    return pixels * np.array([256/320,256/240]), valid


def render(episode, output, mesh_assets, specification):
    report_path = episode / 'report.json'
    report = json.loads(report_path.read_text())
    if not report.get('complete'):
        raise ValueError('CPU replay requires a completed measured episode')
    spec = json.loads(specification.read_text())
    if report['mode']!='native_adapted' or spec['modes']!=['native_adapted']:
        raise ValueError('This renderer is reserved for explicitly trained native_adapted rollouts')
    audit=json.loads((episode/'independent-cpu-audit.json').read_text())
    if not audit.get('passed'):
        raise ValueError('Independent CPU evidence audit must pass before trained-result presentation')
    with np.load(episode / 'trajectory.npz', allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    count = len(arrays['qpos_after'])
    for name, value in arrays.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise ValueError(f'Nonfinite recorded {name}')
    for key in ['scene_rgb', 'wrist_rgb']:
        if arrays[key].shape != (count, 256, 256, 3) or arrays[key].dtype != np.uint8:
            raise ValueError('Need exact saved PIL-processed256x256 model RGB')
    np.testing.assert_allclose(arrays['qpos_after'][:-1], arrays['qpos_before'][1:], atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.diff(arrays['time_s']), 1 / spec['fps'], atol=1e-10, rtol=0)
    model = load_scene(episode / 'cell.xml', mesh_assets)
    targets, prediction_hashes = checked_targets(episode, report, spec, arrays, model)
    state = mujoco.MjData(model)
    observed_error = 0.
    rgb_checks = []
    for tick in range(count):
        state.qpos[:] = arrays['qpos_after'][tick]
        mujoco.mj_forward(model, state)
        observed_error = max(observed_error, float(np.linalg.norm(tcp(model, state) - arrays['tcp_m'][tick])))
        # Diagnostic-only independent pixel consistency check against actual RGB.
        # Cube pose stays outside policy input and is never used for corrections.
        if tick < 8:
            pixel, valid = project_native(model, state, [arrays['cube_truth_m'][tick]])
            rgb = arrays['scene_rgb'][tick]
            red = (rgb[..., 0] > 130) & (rgb[..., 1] < 90) & (rgb[..., 2] < 90)
            yy, xx = np.where(red)
            if valid[0] and len(xx):
                distance = float(np.sqrt((xx + .5 - pixel[0, 0])**2 + (yy + .5 - pixel[0, 1])**2).min())
                rgb_checks.append(distance)
    if observed_error > 2e-7:
        raise ValueError('Recorded TCP fails independent FK')
    if not rgb_checks or max(rgb_checks) > 6:
        raise ValueError('Initial scene camera projection disagrees with saved cube RGB')
    output.mkdir(parents=True, exist_ok=False)
    title, medium, small = font(28, True), font(21), font(17)
    physical = len(report['inference']) * spec['execute_per_chunk']
    clipped = np.any(np.abs(arrays['raw_commands_sim_units'][:physical] - arrays['applied_commands_sim_units'][:physical]) > 1e-8, axis=1)
    if int(clipped.sum()) != report['clipped_commands']:
        raise ValueError('Saved clipping count differs from report')
    video = output / 'flux-replay.mp4'
    snapshots, sources = [], []
    with imageio.get_writer(str(video), fps=30, codec='libx264', quality=9,
                           macro_block_size=1, ffmpeg_params=['-movflags', '+faststart']) as writer:
        for tick in range(count):
            state.qpos[:] = arrays['qpos_after'][tick]
            mujoco.mj_forward(model, state)
            image = Image.new('RGB', (WIDTH, HEIGHT), INK)
            image.paste(Image.fromarray(arrays['scene_rgb'][tick]).resize((960, 720), Image.Resampling.BICUBIC), (20, 145))
            draw = ImageDraw.Draw(image)
            chunk = min(tick // spec['execute_per_chunk'], len(targets) - 1)
            offset = tick % spec['execute_per_chunk']
            pixels, visible = project_native(model, state, arrays['tcp_m'][max(0, tick - 180):tick + 1])
            pixels = pixels * np.array([960/256,720/256]) + [20, 145]
            for a, b, va, vb in zip(pixels[:-1], pixels[1:], visible[:-1], visible[1:]):
                if va and vb:
                    draw.line([tuple(a), tuple(b)], fill=CYAN, width=4)
            if tick < physical:
                pixels, visible = project_native(model, state, targets[chunk])
                pixels = pixels * np.array([960/256,720/256]) + [20, 145]
                for index, (point, vis) in enumerate(zip(pixels, visible)):
                    if not vis:
                        continue
                    x, y = point
                    radius = 5 if index < 32 else 3
                    colour = AMBER if index < 32 else (149, 119, 75)
                    draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=colour, width=2)
                    if index == offset:
                        draw.ellipse((x-10, y-10, x+10, y+10), outline=AMBER, width=3)
            draw.text((24, 14), 'Trained FLUX 3 Action / SO-101 / actual model RGB', font=title, fill='white')
            draw.text((24, 57), f"{report['mode']} / seed {report['seed']} / {(tick+1)/30:.2f} simulated s", font=medium, fill=AMBER)
            draw.text((24, 95), 'BF16 base + trained FP32 LoRA/heads / measured processed RGB / inference pauses physics', font=small, fill=MUTED)
            for name, key, y in [('SCENE', 'scene_rgb', 190), ('WRIST', 'wrist_rgb', 495)]:
                draw.text((1090, y-31), name + ' / saved model256 x 256', font=small, fill='white')
                image.paste(Image.fromarray(arrays[key][tick]), (1090, y))
            inference = report['inference'][chunk]['inference_seconds']
            draw.text((1090, 765), f'Model inference: {inference:.2f} s', font=small, fill=AMBER)
            draw.text((1090, 795), 'Physics waits during inference.', font=small, fill=MUTED)
            draw.text((1090, 825), 'This is recorded playback.', font=small, fill=MUTED)
            phase = f'Prediction {chunk+1}/{len(targets)} / command {offset+1}/32' if tick < physical else 'Settling hold / last FLUX command retained'
            draw.text((24, 886), phase, font=medium, fill='white')
            cumulative = int(clipped[:min(tick+1, physical)].sum())
            draw.text((24, 925), f'Amber: RAW joint-target FK, 42 points / first 32 executed. Cyan: observed TCP. Clipped commands: {cumulative}.', font=small, fill=AMBER)
            draw.text((24, 957), 'Targets refresh every 32 simulated ticks. They are not a guaranteed future tool path or predicted camera video.', font=small, fill=MUTED)
            draw.text((24, 989), 'Actual contact dynamics / free cube / no scripted pickup / diagnostic object coordinates never enter FLUX.', font=small, fill=MUTED)
            terminal = tick == count-1
            status = ('FINAL STRICT GRADER: PASS' if report['success'] else 'FINAL STRICT GRADER: FAIL') if terminal else 'Final strict task grader pending.'
            draw.text((24, 1023), status, font=medium, fill=(146, 234, 155) if terminal and report['success'] else AMBER)
            writer.append_data(np.asarray(image))
            if tick in [0, count//2, count-1]:
                image.save(output / f'frame-{tick:04d}.png')
                snapshots.append(image.copy())
                if tick == count//2:
                    image.save(output / 'poster.png')
            sources.append({'video_frame': tick, 'source_tick': tick, 'simulation_time_s': float(arrays['time_s'][tick]),
                            'prediction_index': chunk, 'command_offset': offset if tick < physical else None,
                            'settling_hold': tick >= physical})
    sheet = Image.new('RGB', (2160, 540), INK)
    for index, image in enumerate(snapshots):
        sheet.paste(image.resize((720, 540)), (index*720, 0))
    sheet.save(output / 'contact-sheet.png')
    np.savez_compressed(output / 'raw-predicted-tcp.npz', tcp_m=targets)
    (output / 'frame-provenance.json').write_text(json.dumps(sources))
    proof = {'renderer_sha256': sha(__file__), 'renderer_helper_sha256': sha(Path(__file__).with_name('render_flux_adapted_helpers.py')),
             'source_report_sha256': sha(report_path), 'source_trajectory_sha256': sha(episode / 'trajectory.npz'),
             'source_xml_sha256': sha(episode / 'cell.xml'), 'frozen_specification_sha256': sha(specification),
             'prediction_file_sha256': prediction_hashes, 'video_sha256': sha(video),
             'width': WIDTH, 'height': HEIGHT, 'fps': 30, 'frames': count,
             'success': bool(report['success']), 'mode': report['mode'], 'seed': report['seed'],
             'rendering': 'CPU-only saved RGB compositor. No GPU context, model inference or dynamics steps.',
             'main_image': 'Actual saved256x256 uint8 PIL-processed model RGB enlarged to960x720; native camera projection320x240',
             'observed_tcp_fk_max_error_m': observed_error,
             'initial_cube_projected_center_distance_to_red_pixel_px': rgb_checks,
             'projection_validation': 'Native optical projection checked against initial RGB; no GL fiducial check performed',
             'raw_prediction_conversion_checked': True, 'applied_control_clipping_checked': True,
             'state_continuity_checked': True, 'model_weights_included': False,
             'independent_episode_audit_sha256':sha(episode/'independent-cpu-audit.json'),
             'trained_adapter_sha256':spec['adapter_sha256'],
             'adapter_loader_sha256':spec['adapter_loader_sha256'],
             'calibration':'Native simulator degrees and gripper percentage; identity',
             'preprocessing':'Physical320x240 projection -> uint8 PIL bilinear256x256',
             'precision':'Original BF16 base + FP32 LoRA/full heads under BF16 autocast'}
    (output / 'provenance.json').write_text(json.dumps(proof, indent=2))
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mesh-assets', type=Path, required=True)
    parser.add_argument('--specification', type=Path)
    args = parser.parse_args()
    episode = args.episode.resolve()
    spec = (args.specification or episode.parents[1] / 'frozen-specification.json').resolve()
    print(json.dumps(render(episode, args.output.resolve(), args.mesh_assets.resolve(), spec), indent=2))
