"""CPU-load actual native processors and reconcile checkpoint statistics with its data."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def serial(value):
    if isinstance(value,dict):return {str(key):serial(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [serial(item) for item in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if hasattr(value,'detach'):return value.detach().cpu().numpy().tolist()
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['smolvla','groot'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve normalization evidence')
    stats=json.loads((a.dataset/'meta/stats.json').read_text());report={'scope':'actual CPU native processor loading; no model or GPU allocation','model':a.model,'complete':False,'passed':False,'dataset_stats_sha256':sha(a.dataset/'meta/stats.json'),'checks':[],'processor_files':{}}
    try:
        if a.model=='smolvla':
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            from lerobot.policies.factory import make_pre_post_processors
            config=SmolVLAConfig.from_pretrained(str(a.checkpoint));config.device='cpu'
            pre,post=make_pre_post_processors(config,pretrained_path=str(a.checkpoint),preprocessor_overrides={'device_processor':{'device':'cpu'}})
            for pipeline,name in [(pre,'pre'),(post,'post')]:
                for step in pipeline.steps:
                    tensors=getattr(step,'_tensor_stats',{})
                    for feature in ['observation.state','action']:
                        for kind in ['mean','std']:
                            if feature not in tensors or kind not in tensors[feature]:continue
                            error=float(np.max(np.abs(np.asarray(serial(tensors[feature][kind]))-np.asarray(stats[feature][kind]))))
                            report['checks'].append({'pipeline':name,'feature':feature,'stat':kind,'max_abs_error':error,'passed':error<1e-6})
            for path in a.checkpoint.glob('policy_*processor*'):report['processor_files'][path.name]=sha(path)
        else:
            import gr00t.policy.gr00t_policy
            # The stock model constructor normally imports/registers this class;
            # a processor-only CPU audit must register it explicitly.
            import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7
            from transformers import AutoProcessor
            processor=AutoProcessor.from_pretrained(a.checkpoint);processor.eval()
            active=serial(processor.state_action_processor.statistics)['new_embodiment'];saved=json.loads((a.checkpoint/'statistics.json').read_text())['new_embodiment'];native=json.loads((a.checkpoint/'experiment_cfg/dataset_statistics.json').read_text())['new_embodiment']
            report['checks'] += [{'check':'native loaded NEW_EMBODIMENT statistics exactly match saved statistics.json','passed':active==saved},{'check':'saved statistics exactly match native training dataset_statistics.json','passed':saved==native}]
            for feature,key,start,end in [('observation.state','single_arm',0,5),('observation.state','gripper',5,6),('action','single_arm',0,5),('action','gripper',5,6)]:
                for kind in ['mean','std','min','max','q01','q99']:
                    if kind not in stats[feature]:continue
                    error=float(np.max(np.abs(np.asarray(active['state' if feature=='observation.state' else 'action'][key][kind])-np.asarray(stats[feature][kind])[start:end])))
                    report['checks'].append({'feature':feature,'group':key,'stat':kind,'max_abs_error':error,'passed':error<1e-6})
            report['active_norm_parameters']=serial(processor.state_action_processor.norm_params['new_embodiment'])
            report['relative_action_stats']=active.get('relative_action');report['native_flags']={key:getattr(processor,key,None) for key in ['use_percentiles','use_mean_std','use_relative_action','clip_outliers']}
            for name in ['processor_config.json','statistics.json','embodiment_id.json']:report['processor_files'][name]=sha(a.checkpoint/name)
        assert report['checks'] and all(row['passed'] for row in report['checks']),'Native processor stats mismatch'
        report.update(complete=True,passed=True)
    except Exception as error:report['error']=str(error);raise
    finally:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps({'model':a.model,'passed':report['passed'],'checks':len(report['checks'])}))

if __name__=='__main__':main()
