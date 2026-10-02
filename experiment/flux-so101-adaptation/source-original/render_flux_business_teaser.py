"""Make a simple square presentation from an audited FLUX replay, on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import imageio_ffmpeg


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--replay-folder', type=Path, required=True)
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--font', type=Path, default=Path('C:/Windows/Fonts/arial.ttf'))
    a = p.parse_args()
    report = json.loads((a.episode / 'report.json').read_text())
    audit = json.loads((a.episode / 'independent-cpu-audit.json').read_text())
    suite = json.loads((a.suite / 'summary.json').read_text())
    suite_audit = json.loads((a.suite / 'independent-cpu-audit.json').read_text())
    source = a.replay_folder / 'flux-replay.mp4'
    proof = json.loads((a.replay_folder / 'provenance.json').read_text())
    assert report['complete'] and report['success'] and audit['passed']
    assert suite['complete'] and suite_audit['passed']
    assert proof['source_report_sha256'] == sha(a.episode / 'report.json')
    assert proof['source_trajectory_sha256'] == report['trajectory_sha256']
    assert proof['video_sha256'] == sha(source)
    assert proof['width'] == 1440 and proof['height'] == 1080
    a.output.mkdir(parents=True, exist_ok=False)
    font = a.font.resolve().as_posix().replace(':', r'\:')
    labels = [
        ('FLUX 3 Action / SO-101', 42, 24, 20, 'white'),
        (report['prompt'], 27, 24, 82, 'white'),
        ('Predicted movement', 25, 24, 946, '0xffbd50'),
        ('Actual movement', 25, 560, 946, '0x60e5e1'),
        (f"Simulation | 17 training demonstrations | {suite['successful']}/{suite['total_completed']} fresh placements", 24, 24, 988, 'white'),
        ('Playback omits model inference pauses', 24, 24, 1030, '0xb7c9d7'),
    ]
    filters = ['crop=960:720:20:145', 'scale=1080:810', 'pad=1080:1080:0:125:color=0x08141e']
    for i, (label, size, x, y, color) in enumerate(labels):
        path = (a.output / f'label-{i}.txt').resolve()
        path.write_text(label, encoding='utf-8')
        text_path = path.as_posix().replace(':', r'\:')
        filters.append(f"drawtext=fontfile='{font}':textfile='{text_path}':fontsize={size}:fontcolor={color}:x={x}:y={y}")
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-i', str(source),
               '-vf', ','.join(filters), '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(a.output / 'flux-teaser.mp4')]
    subprocess.run(command, check=True)
    subprocess.run([command[0], '-hide_banner', '-loglevel', 'error', '-ss', '13.57', '-i',
                    str(a.output / 'flux-teaser.mp4'), '-frames:v', '1', str(a.output / 'poster.png')], check=True)
    provenance = {'renderer_sha256': sha(Path(__file__)), 'source_video_sha256': sha(source),
                  'source_report_sha256': sha(a.episode / 'report.json'),
                  'source_trajectory_sha256': report['trajectory_sha256'],
                  'source_audit_sha256': sha(a.episode / 'independent-cpu-audit.json'),
                  'suite_summary_sha256': sha(a.suite / 'summary.json'),
                  'video_sha256': sha(a.output / 'flux-teaser.mp4'), 'seed': report['seed'],
                  'fresh_task_successes': suite['successful'], 'fresh_task_trials': suite['total_completed'],
                  'presentation': 'CPU FFmpeg crop of actual saved model RGB and original prediction/measured-path overlays, resized to a square presentation with plain labels. No new model inference or physics.',
                  'source_crop_xywh': [20, 145, 960, 720], 'width': 1080, 'height': 1080,
                  'simulation_only': True, 'inference_pauses_omitted': True}
    (a.output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance))


if __name__ == '__main__':
    main()
