"""Record the private master suite mapping without promoting surrogate pilot tests."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    master_path=ROOT/'runtime/action-demo-suite-v1/suite.json';master=json.loads(master_path.read_text())
    office=[row for row in master['cases'] if row['group']=='office']
    mappings={
        'target-selection':('six color/destination cube tasks','Surrogate color-cube objects; master cap/coupling/bolt embodiment not implemented; learned object selection not evaluated'),
        'destination-swap':('red-left/red-right same-scene pair','Development paired rollouts only; none of five master sealed starts evaluated; ACT routes separate task specialists explicitly'),
        'same-scene-instruction-swap':('all twelve expert instructions share initial scene','Learned evaluations change red destination only, not requested object; object counterfactual remains unexecuted'),
        'two-step-sort':('two red/blue ordered expert sequences','Expert dataset surrogate only; no learned sequence rollout'),
        'ordered-sequence':('two reversed red/blue expert sequences','Expert dataset surrogate only; no learned ordering test'),
        'relative-position':('leftmost/rightmost object selection experts','Master challenge asks for relative destination behind coupling; object-relative surrogate does not implement it'),
        'attribute-conjunction':(None,'No varied size/material object attributes or conjunction task'),
        'exclusion':('two protected-distractor expert instructions','Independent distractor grader exists; no learned exclusion task rollout'),
        'paraphrase':('six validation paraphrases','Smol development paraphrase failures retained; master sealed paraphrase cases unexecuted'),
        'ambiguous-reference':(None,'Clarification supervisor and no-motion ambiguous requests not implemented'),
        'shifted-start':('small object position/yaw jitter','Robot starts at the same rest state; explicit robot shifted-start challenge not implemented'),
        'low-light':(None,'Lighting variation not implemented'),
        'partial-occlusion':(None,'Occlusion variation not implemented'),
        'visual-distractor':('three visible color cubes','Existing distractor grading is not the master added visual-distractor challenge; no varied extra distractor condition'),
    }
    rows=[]
    for challenge in sorted(set(row['challenge'] for row in office)):
        surrogate,reason=mappings[challenge];cases=[row for row in office if row['challenge']==challenge]
        rows.append({'master_challenge':challenge,'master_case_count':len(cases),'master_status':'not_executed',
                     'master_completed_cases':0,'pilot_surrogate':surrogate,'reason':reason})
    output={'scope':'private master-office mapping; separate from the office-only sealed120 contract',
            'master_source':str(master_path),'master_source_sha256':hashlib.sha256(master_path.read_bytes()).hexdigest(),
            'master_total_cases':len(master['cases']),'master_office_cases':len(office),'master_pipe_cases':len(master['cases'])-len(office),
            'office_only_sealed_contract':'suites/sealed-test-120.json: twelve office pilot tasks x ten seeds; planned only',
            'master_office_completed_cases':0,'master_office_executed_cases':0,'mapping':rows,
            'actual_development_scope':'paired red-left/red-right at seed10000, plus retained Smol familiar seed1000 controls; experts48 separate',
            'conditional_ACT_sealed_protocol':{'status':'not_authorized_to_execute_until_development_selection_passes',
                'seeds':[20000,20001,20002,20003,20004],'tasks':['red-left','red-right'],'prospective_rollouts':10,
                'interpretation':'narrow task-specialist destination pair; does not complete the master70 office matrix or establish general language grounding',
                'checkpoint_selection':'use development only; no adjustment after looking at sealed outcomes'}}
    target=ROOT/'runtime/action-training-phase/office/suites/master-office-coverage-status.json'
    target.write_text(json.dumps(output,indent=2));print(json.dumps({'path':str(target),'master_office_completed_cases':0,'master_office_cases':len(office)}))


if __name__=='__main__':main()
