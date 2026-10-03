"""CPU-only visual replay variant preserving original cell poses and report provenance."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);source=a.source_review/'review-contract.json';contract=json.loads(source.read_text());replays=[]
    for cell in contract['cells']:
        replay=a.output/f'cell-{cell["cell"]:03d}.usda'
        subprocess.run([sys.executable,str(Path(__file__).with_name('office_export_isaac_replay.py')),'--episode',cell['episode'],'--output',str(replay),'--stride','4','--presentation-legibility'],check=True,stdout=subprocess.DEVNULL)
        replays.append(replay)
    def matrix(selected,destination,columns):
        subprocess.run([sys.executable,str(Path(__file__).with_name('office_isaac_matrix.py')),'--replays',*[str(path) for path in selected],'--output',str(destination),'--columns',str(columns)],check=True)
    matrix(replays,a.output/'matrix.usda',4);matrix(replays[:4],a.output/'subset-four.usda',2)
    matrix([replays[-1]],a.output/'closeup-last-failure.usda',1)
    manifest={'scope':'visual-only legibility variant of measured MuJoCo poses; no new physics/policy success',
      'source_contract_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cells':contract['cells'],
      'modifications':['remove duplicated visual walls2,3','resolve source material colors','light white/orange robot body palette','double-sided mesh'],
      'poses_modified':False,'requested_capture_resolution':[1920,1080],'native_capture_status':'NOT_EXECUTED; GPU handoff required',
      'subset_scope':'first four source cells only; full overview retains all source failures',
      'files':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in a.output.iterdir() if path.is_file()}}
    (a.output/'legibility-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'output':str(a.output),'cells':len(replays),'native_capture':'NOT_EXECUTED'}))

if __name__=='__main__':main()
