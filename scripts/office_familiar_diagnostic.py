"""Inspect retained familiar-layout observations/commands without changing the policy."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];OFFICE=ROOT/'runtime/action-training-phase/office'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    source=OFFICE/'dataset/train/seed-1000/red-left/episode.npz';episode=OFFICE/'evaluation-smol-union18-trainscene-control/seed-1000-red-left';output=OFFICE/'comparison-v6/familiar-feedback-diagnostic.json'
    if output.exists():raise FileExistsError('Retain diagnostics')
    with np.load(source) as data:expert_state=data['states'].copy();expert_action=data['actions'].copy();expert_scene=data['images_scene'][[0,8,16]].copy();expert_wrist=data['images_wrist'][[0,8,16]].copy()
    with np.load(episode/'trajectory.npz') as data:states=data['states'].copy();actions=data['actions'].copy();applied=data['applied_actions'].copy()
    with np.load(episode/'predictions.npz') as data:actual_chunks=data['chunks'][:3].copy()
    with np.load(episode/'first-inputs.npz') as data:scene=data['scene'].copy();wrist=data['wrist'].copy();input_states=data['states'].copy()
    with np.load(OFFICE/'comparison-v6/teacher-smolvla/full-chunks.npz') as data:teacher_chunks=data['chunks'][:3].copy()
    report={'scope':'familiar TRAINING-scene control diagnostic; no held-out success or causal proof','episode_sha256':sha(source),'trajectory_sha256':sha(episode/'trajectory.npz'),
      'same_reset_rng':510000,'decoded_commands':'before unchanged common supervisor','complete_control_outcome':json.loads((episode/'episode.json').read_text()),'first16':[], 'anchors':[]}
    for index in range(16):report['first16'].append({'tick':index,'observed_joints_radians':states[index].tolist(),'expert_observed_joints_radians':expert_state[index].tolist(),'decoded_command_radians':actions[index].tolist(),'expert_command_radians':expert_action[index].tolist(),'applied_actuator_radians':applied[index].tolist()})
    canvas=Image.new('RGB',(1024,3*300),(15,25,30));draw=ImageDraw.Draw(canvas)
    for row,index in enumerate([0,8,16]):
        report['anchors'].append({'tick':index,'observed_state_max_abs_error_radians':float(np.abs(input_states[row]-expert_state[index]).max()),
          'scene_rgb_mae_255':float(np.abs(scene[row].astype(float)-expert_scene[row]).mean()),'wrist_rgb_mae_255':float(np.abs(wrist[row].astype(float)-expert_wrist[row]).mean()),
          'actual_first8_expert_mae_radians':float(np.abs(actual_chunks[row,:8]-expert_action[index:index+8]).mean()),
          'teacher_first8_expert_mae_radians':float(np.abs(teacher_chunks[row,:8]-expert_action[index:index+8]).mean()),
          'actual_first50_expert_mae_radians':float(np.abs(actual_chunks[row]-expert_action[index:index+len(actual_chunks[row])]).mean()),
          'actual_first_command':actual_chunks[row,0].tolist(),'teacher_first_command':teacher_chunks[row,0].tolist(),'expert_first_command':expert_action[index].tolist()})
        for column,image in enumerate([expert_scene[row],scene[row],expert_wrist[row],wrist[row]]):canvas.paste(Image.fromarray(image),(column*256,row*300+34))
        draw.text((8,row*300+8),f't{index}: Expert scene | Actual scene | Expert wrist | Actual wrist. Training-scene diagnostic.',fill='white')
    canvas.save(OFFICE/'comparison-v6/familiar-observation-comparison.png');output.write_text(json.dumps(report,indent=2));print(json.dumps({'anchors':report['anchors']}))

if __name__=='__main__':main()
