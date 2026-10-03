"""Save deterministic scene-level training plans and a sealed 120-case test contract."""
import argparse,json
from pathlib import Path
import office_environment as office
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=office.OUTPUT/'suites');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    plans={'pilot':{'train':list(range(1000,1003)),'validation':[10000]},
           'increment':{'train':list(range(1000,1008)),'validation':[10000,10001]},
           'target50plus10':{'train':list(range(1000,1050)),'validation':list(range(10000,10010))}}
    (a.output/'scene-split-plan.json').write_text(json.dumps({'seeds':plans,'cases_per_seed':12,'episode_counts':{'pilot':[36,12],'increment':[96,24],'target50plus10':[600,120]},'status':'pilot rendered/audited; other entries are plans, not generated data','sealed_test_used_for_checkpoint_selection':False},indent=2))
    tests=[{'seed':s,'scene':office.scene_spec(s),'tasks':office.instructions(s,'validation')} for s in range(20000,20010)]
    (a.output/'sealed-test-120.json').write_text(json.dumps({'scope':'held-out deterministic evaluation contract; no rollout outcomes yet','split':'test','seeds':list(range(20000,20010)),'episodes':120,'scenes':tests,'checkpoint_selection_allowed':False,'generation_source_hashes':office.source_hashes()},indent=2))
    print(json.dumps({'output':str(a.output),'sealed_cases':120,'evaluated_cases':0}))
if __name__=='__main__':main()
