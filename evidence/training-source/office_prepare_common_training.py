"""Prepare, never launch, matched task-breadth stock training after timed diagnostics."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    output=ROOT/'runtime/action-training-phase/office/common-training-v1';output.mkdir(parents=True,exist_ok=False)
    source=ROOT/'runtime/action-training-phase/cloud-evidence/evidence-20261003-1128/data/runs'
    snapshots={model:json.loads((source/f'{old}-main-01-launch-manifest.json').read_text()) for model,old in [('smolvla','smol'),('groot','groot')]}
    recipes={}
    for model,previous in snapshots.items():
        recipes[model]={'source_pin':previous['source_revision'],'base_model_snapshot':previous['base_model'],
          'dataset_format':'v3.0' if model=='smolvla' else 'v2.1','dataset_task_ids':['red-left','red-right'],
          'scene_seeds':[1000,1001,1002],'episodes':6,'frames':3150,
          'smoke':{'updates':20,'save_every':20,'max_wall_seconds':600,'gate':'native saved fresh-process reload finite[H,6], no robot quality claim'},
          'main':{'updates':2000,'save_every':2000,'max_wall_seconds':3600,'start':'same official base snapshot, new optimizer/scheduler; smoke is not silently resumed','checkpoint_barrier':False,'save_total_limit':1},
          'batch_size':16 if model=='smolvla' else 8,
          'trainable':'action expert and state projector; vision frozen' if model=='smolvla' else 'projector and diffusion action model; language/visual frozen',
          'native_decoder':'MEAN_STD absolute commands' if model=='smolvla' else 'MIN_MAX q01/q99 native clipping, relative-current-state arm decode and absolute gripper',
          'post_training':'quiesce trainers, hash direct full native checkpoint files then same-A100 matched development pair with unchanged grader; no sealed test until development selection'}
    recipe={'scope':'approved planned stock retry with matched two-task union; no training launched by this preparation',
      'trigger':'only after timed common8/common16 comparisons complete; retain every original failure',
      'dataset_archive_sha256':'63586275f3e7c817d8dd64f67908c40f03384bb437ded07cb5b4dffceb9762f2',
      'dataset_source':'runtime/action-training-phase/act/common-comparison-data/dataset-v1',
      'models':recipes,'guards':{'whole_cloud_gpu_mib':71680,'whole_cloud_ram_mib':102400,'external_hard_stop_utc':'2026-10-03T14:52:00Z'},
      'parallelism':'serial by default; parallel main only after measured global GPU/RAM headroom and parent coordination; no comparison overlaps trainers',
      'recovery':{'method':'direct native-file manifest/SCP, every file SHA verified; no duplicate local TAR','local_available_gb_reported_by_parent':69,'estimated_final_full_groot_gb':25.55,'stock_full_smoke':'keep remote smoke until reload evidence saved; no duplicate local full smoke recovery needed'},
      'claim_limits':'Matched task breadth/data does not equal architecture, batch size, objective, pretraining, or total compute. ACT remains separate exact-task router baseline. Corrective off-expert data is a separate future intervention.'}
    target=output/'recipe.json';target.write_text(json.dumps(recipe,indent=2));(output/'recipe.sha256').write_text(hashlib.sha256(target.read_bytes()).hexdigest()+'  recipe.json\n')
    print(json.dumps({'recipe':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))

if __name__=='__main__':main()
