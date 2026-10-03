"""Freeze a prospective common16 ablation only after the matched common8 failures."""
import argparse,datetime,hashlib,json,shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    evidence={}
    for model in ['smolvla','groot','act']:
        path=a.previous/f'rollout-{model}'/'report.json';report=json.loads(path.read_text())
        if not report['complete'] or len(report['episodes'])!=2 or report['task_successes']!=0 or report['chunk_execution']!=8:
            raise ValueError('Conditional ablation requires complete matched8 pair failures for every model')
        evidence[model]={'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'task_successes':0,'episodes':2}
    a.output.mkdir(parents=True,exist_ok=False);original=a.previous/'frozen-contract.json';contract=json.loads(original.read_text())
    contract['scope']='prospective common16 temporal ablation after retained common8 development failures; no sealed test'
    contract['development']['chunk_execution']=16
    contract['ablation']={'single_changed_control':'execute16 instead of8 commands per fresh prediction','previous_contract_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
      'triggering_evidence':evidence,'native_horizon_note':'16 is within every unchanged native model chunk horizon','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (a.output/'frozen-contract.json').write_text(json.dumps(contract,indent=2))
    for name in ['act-native-selector.json','execution-config.private.json']:shutil.copy2(a.previous/name,a.output/name)
    digest=hashlib.sha256((a.output/'frozen-contract.json').read_bytes()).hexdigest()
    (a.output/'frozen-contract.sha256').write_text(digest+'  frozen-contract.json\n');print(json.dumps({'contract':str(a.output/'frozen-contract.json'),'sha256':digest}))

if __name__=='__main__':main()
