"""Summarize matched development rollouts with model-native decoder caveats."""
import argparse,hashlib,json,statistics
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--comparison',type=Path,required=True);a=p.parse_args();models={};first_inputs={}
    contract_path=a.comparison/'frozen-contract.json';contract=json.loads(contract_path.read_text())
    for name in contract['order']:
        folder=a.comparison/f'rollout-{name}';path=folder/'report.json';report=json.loads(path.read_text())
        if not report['complete']:raise ValueError('Incomplete comparison')
        rows=[]
        for episode in report['episodes']:
            source=folder/f'seed-{episode["seed"]}-{episode["task"]["id"]}'
            predictions=json.loads((source/'prediction-inputs.json').read_text());first_inputs.setdefault(episode['task']['id'],[]).append(predictions[0]['input_sha256'])
            rows.append({key:episode.get(key) for key in ['seed','task_success','success','command_limit_events','steps','server_synchronized_inference_p50_s','server_synchronized_inference_p95_s','fresh_inference_p50_s','fresh_inference_p95_s','trajectory_sha256','prediction_trace_sha256']})
            rows[-1]['task']=episode['task']['id'];rows[-1]['target_grader']=episode['grader'];rows[-1]['actual_requests']=len(predictions)
        guard=json.loads((folder/'guard-report.json').read_text())
        models[name]={'episodes':rows,'task_successes':report['task_successes'],'strict_successes':report['successes'],
          'native_decoder':contract['models'][name]['decoder'],'training_scope':contract['models'][name]['training_scope'],
          'learned_language':contract['models'][name]['learned_language'],'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'admission':guard['admission'],'whole_host_gpu_peak_mib':guard['peak_gpu_mib'],'whole_host_ram_peak_mib':guard['peak_host_mib']}
    if any(not all(row==rows[0] for row in rows) for rows in first_inputs.values()):raise ValueError('Initial actual policy inputs differ across models')
    prior=contract.get('expected_previous_first_actual_inputs')
    if prior and any(rows[0]!=prior[task] for task,rows in first_inputs.items()):raise ValueError('Actual initial inputs differ from frozen previous comparison scenes')
    summary={'scope':contract['scope'],'contract_sha256':hashlib.sha256(contract_path.read_bytes()).hexdigest(),
      'same_actual_first_scene_state_text_hashes':True,'first_inputs':{task:rows[0] for task,rows in first_inputs.items()},
      'chunk_execution':contract['development']['chunk_execution'],'models':models,'sealed_tests_executed':0,
      'timing_scope':'server CUDA-synchronized native policy processing separated from end-to-end PNG/SSH; simulation pauses',
      'safety_scope':'policy-decoded commands before common supervisor; native decoder clipping differs; zero events do not prove raw neural head safety',
      'ranking_limit':'training scopes/objectives differ; ACT uses exact external task routing, not learned language'}
    destination=a.comparison/'comparison-summary.json'
    if destination.exists():raise FileExistsError('Do not overwrite original summary')
    destination.write_text(json.dumps(summary,indent=2));print(json.dumps({'models':{name:{key:value for key,value in row.items() if key not in ['episodes','native_decoder','training_scope']} for name,row in models.items()}}))

if __name__=='__main__':main()
