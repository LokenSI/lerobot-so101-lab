"""CPU-only overlays for complete matched comparisons, retaining failed cases."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    contract=json.loads((a.comparison/'frozen-contract.json').read_text());a.output.mkdir(parents=True,exist_ok=True)
    for model in contract['order']:
        folder=a.comparison/f'rollout-{model}';report=json.loads((folder/'report.json').read_text())
        if not report['complete']:raise ValueError('Incomplete comparison')
        for row in report['episodes']:
            episode=folder/f'seed-{row["seed"]}-{row["task"]["id"]}'
            output=a.output/f'{a.comparison.name}-{model}-{row["task"]["id"]}.mp4'
            if output.exists():raise FileExistsError('Do not overwrite previous videos')
            metadata=contract['models'][model];steps=metadata.get('updates',metadata.get('updates_per_task'))
            subprocess.run([sys.executable,str(Path(__file__).with_name('office_overlay_rollout.py')),'--episode',str(episode),'--output',str(output),'--cumulative-step',str(steps),'--chunk-execution',str(contract['development']['chunk_execution'])],check=True)
    print(json.dumps({'scope':'presentation only, actual unchanged failed/successful outcomes','comparison':str(a.comparison),'clips':6}))

if __name__=='__main__':main()
