"""Bounded same-A100 policy RPC with native decoders and synchronized server timing."""
import argparse,base64,contextlib,hashlib,io,json,os,subprocess,sys,threading,time
from pathlib import Path


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def make_act(selector,device):
    # Explicit, auditable absolute-path constructor. Reset/predict remain the
    # original ACT adapter methods and checkpoint-owned native processors.
    from office_act_adapter import Policy
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    import torch
    document=json.loads(Path(selector).read_text())
    allowed={task:f'/data/action-training/act/runs/{task}-v2/main2000/checkpoints/002000/pretrained_model' for task in ['red-left','red-right']}
    if document['checkpoints']!=allowed or document.get('learned_language') is not False:raise ValueError('ACT selector differs from the explicitly verified native two-path allowlist')
    policy=Policy.__new__(Policy);policy.torch=torch;policy.models={};policy.selected=None;hashes={}
    for task,location in allowed.items():
        root=Path(location)
        for row in document['files'][task]:
            if sha(root/row['path'])!=row['sha256']:raise ValueError('ACT native model/processor file SHA mismatch')
        config=ACTConfig.from_pretrained(location);config.device=device
        model=ACTPolicy.from_pretrained(location,config=config).to(device).eval()
        pre,post=make_pre_post_processors(config,pretrained_path=location,preprocessor_overrides={'device_processor':{'device':device}})
        policy.models[task]=(model,pre,post);hashes[task]=sha(root/'model.safetensors')
    policy.metadata={'model':'ACT task-specific pair','learned_policy':True,'learned_language':False,
       'office_updates_by_task':document['office_updates_by_task'],'external_task_selector':document['selector'],
       'action_units':'radians','checkpoint_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
       'native_adapter_sha256':sha(sys.modules['office_act_adapter'].__file__)}
    return policy


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['smolvla','groot','act'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--max-seconds',type=int,default=1500);a=p.parse_args()
    a.report.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic();stop=threading.Event()
    status={'scope':'one comparison inference worker on existing A100; no simulator truth or training',
      'model':a.model,'checkpoint':str(a.checkpoint),'gpu_limit_mib':71680,'ram_limit_mib':102400,
      'gpu_peak_mib':0,'host_peak_mib':0,'requests':0,'complete':False}
    def save():
        temporary=a.report.with_suffix('.writing.json');temporary.write_text(json.dumps(status,indent=2));temporary.replace(a.report)
    def sample():
        gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=3).splitlines()[0])
        memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
        ram=(memory['MemTotal']-memory['MemAvailable'])/1024
        status['gpu_peak_mib']=max(status['gpu_peak_mib'],gpu);status['host_peak_mib']=max(status['host_peak_mib'],ram)
        if gpu>=71680 or ram>=102400:raise MemoryError('Cloud whole-host comparison admission threshold exceeded')
        if time.monotonic()-started>a.max_seconds:raise TimeoutError('Cloud comparison wall-clock bound exceeded')
    def reply(value):sys.stdout.write(json.dumps(value,separators=(',',':'))+'\n');sys.stdout.flush()
    sample();save()
    def monitor():
        while not stop.wait(.5):
            try:sample();save()
            except Exception as exc:status['error']=str(exc);save();os._exit(77)
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import numpy as np,torch,transformers
            from PIL import Image
            torch.cuda.reset_peak_memory_stats()
            if a.model=='act':policy=make_act(a.checkpoint,'cuda')
            else:
                from cloud_train_policy import CloudPolicy
                policy=CloudPolicy(a.model,str(a.checkpoint),'cuda')
        if a.model=='act':
            checkpoint_hash=policy.metadata['checkpoint_sha256'];config_hash=sha(a.checkpoint)
            dtype=str(next(policy.models['red-left'][0].parameters()).dtype)
        else:
            digest=hashlib.sha256()
            # Hash model weights, excluding separately saved processor tensors.
            for shard in sorted(a.checkpoint.glob('model*.safetensors')):digest.update((shard.name+':'+sha(shard)+'\n').encode())
            checkpoint_hash=digest.hexdigest();config_hash=sha(a.checkpoint/'config.json')
            dtype=str(next(policy.policy.parameters()).dtype) if a.model=='smolvla' else str(policy.policy.model.dtype)
        ready={'ready':True,'model':{'smolvla':'SmolVLA','groot':'GR00T-N1.7-3B','act':'ACT task-specific pair'}[a.model],
               'checkpoint_sha256':checkpoint_hash,'checkpoint_hash_scope':'model weight filename/SHA aggregate (ACT task/model hash map)',
               'model_config_sha256':config_hash,'server_sha256':sha(__file__),'inference_device':torch.cuda.get_device_name(0),
               'deployment_dtype':dtype,'torch_version':torch.__version__,'transformers_version':transformers.__version__,
               'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               'decoder_bound_semantics':'native q01/q99 MIN_MAX clips normalized values to[-1,1]' if a.model=='groot' else 'native MEAN_STD decoded commands unbounded',
               'learned_language':a.model!='act','timing_scope':'CUDA synchronized before and after native policy processing; excludes SSH and PNG decode',
               'owned_cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
        if a.model=='act':ready['external_task_selector']=policy.metadata['external_task_selector']
        reply(ready)
        for line in sys.stdin:
            request=json.loads(line)
            if request['kind']=='close':break
            with contextlib.redirect_stdout(sys.stderr):
                if request['kind']=='reset':
                    torch.manual_seed(int(request['seed']));policy.reset();result={'reset':True,'rng_seed':int(request['seed'])}
                elif request['kind']=='predict':
                    def decode(value):return np.asarray(Image.open(io.BytesIO(base64.b64decode(value))).convert('RGB')).copy()
                    scene,wrist=decode(request['scene_png']),decode(request['wrist_png']);state=np.asarray(request['state'],dtype=np.float32)
                    hashes={'scene':hashlib.sha256(scene.tobytes()).hexdigest(),'wrist':hashlib.sha256(wrist.tobytes()).hexdigest(),
                            'state_float32':hashlib.sha256(state.tobytes()).hexdigest(),'instruction_utf8':hashlib.sha256(request['instruction'].encode()).hexdigest()}
                    torch.cuda.synchronize();before=time.perf_counter()
                    if a.model=='act':actions=policy.predict({'scene':scene,'wrist':wrist,'state':state},request['instruction'])
                    else:actions=policy.predict(scene,wrist,state,request['instruction'])
                    torch.cuda.synchronize();latency=time.perf_counter()-before
                    result={'actions':actions.tolist(),'cloud_inference_seconds':latency,'input_sha256':hashes}
                    status['requests']+=1;status['owned_cuda_peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
                else:raise ValueError('Unknown comparison request')
            reply({'id':request['id'],**result})
        status['complete']=True
    except Exception as exc:status['error']=str(exc);reply({'error':str(exc)});raise
    finally:stop.set();worker.join(timeout=4);status['wall_seconds']=time.monotonic()-started;save()


if __name__=='__main__':main()
