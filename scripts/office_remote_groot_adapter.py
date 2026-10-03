"""Observations-only RPC adapter; real policy inference occurs on the existing task VM."""
import base64,hashlib,io,json,os,selectors,shlex,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image
import office_ssh_transport as brev_remote

ROOT=Path(__file__).resolve().parents[1]
CLOUD_SOURCE='/home/ubuntu/office-action-training/sources/Isaac-GR00T'
CLOUD_BASE='/home/ubuntu/workspace/action-training'


def windows_path(value):
    if value.startswith('/mnt/'):
        return value[5].upper()+':'+value[6:].replace('/','\\')
    return value


class RemoteGroot:
    def __init__(self,checkpoint,device):
        if device!='cpu':raise ValueError('Remote inference requires explicit local --device cpu')
        self.seed=510000;self.counter=0;self.cloud_times=[]
        local=Path(checkpoint).resolve()
        name=os.environ.get('OFFICE_REMOTE_SESSION','groot-2000-rpc-'+str(int(time.time())))
        self.session=ROOT/'runtime/action-training-phase/office/remote-sessions'/name
        self.session.mkdir(parents=True,exist_ok=False)
        self.stderr=(self.session/'cloud-stderr.log').open('w')
        destination,port=brev_remote.connection()
        options=[]
        for value in brev_remote.options():
            if value.startswith(('UserKnownHostsFile=','-oUserKnownHostsFile=')):
                value=value.split('=',1)[0]+'='+windows_path(value.split('=',1)[1])
            else:value=windows_path(value)
            options.append(value)
        ssh='/mnt/c/Windows/System32/OpenSSH/ssh.exe' if os.name=='posix' else 'ssh'
        step=int(os.environ.get('OFFICE_REMOTE_CHECKPOINT_STEP','2000'))
        if step not in (2000,4000):raise ValueError('Require an explicitly authorized saved main-run checkpoint step')
        if local.name!=f'checkpoint-{step}':raise ValueError('Local provenance reference and explicit remote checkpoint step differ')
        cloud_checkpoint=CLOUD_BASE+f'/data/runs/groot-main-01/checkpoint-{step}'
        report=CLOUD_BASE+'/groot-eval/'+name+'-report.json'
        command='cd '+shlex.quote(CLOUD_SOURCE)+' && env PYTHONPATH='+CLOUD_BASE+'/package/scripts HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python '+CLOUD_BASE+'/groot-eval/office_remote_policy_server.py --checkpoint '+shlex.quote(cloud_checkpoint)+' --report '+shlex.quote(report)+' --max-seconds 1500'
        self.child=subprocess.Popen([ssh,'-T',*options,'-p',str(port),destination,command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,bufsize=1)
        self.selector=selectors.DefaultSelector();self.selector.register(self.child.stdout,selectors.EVENT_READ)
        try:ready=self.read_reply(300)
        except Exception:
            self.child.terminate();self.child.wait(timeout=10);self.stderr.close();self.selector.close();raise
        if not ready.get('ready'):raise RuntimeError('Remote policy failed its cold-load handshake')
        self.metadata={'model':ready['model'],'learned_policy':True,'action_units':'radians','checkpoint':str(local),
                       'checkpoint_sha256':ready['checkpoint_sha256'],'checkpoint_hash_scope':ready['checkpoint_hash_scope'],
                       'model_config_sha256':ready['model_config_sha256'],'deployment_dtype':ready['deployment_dtype'],
                       'inference_device':ready['inference_device'],'inference_scope':'existing cloud GPU via SSH stdin RPC; local CPU OSMesa physics/rendering',
                       'remote_report':report,'session':str(self.session),'control':'identical cloud observations-only decoder; no IK/manual correction',
                       'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self.metadata['server_sha256']=ready['server_sha256']
        self.metadata['cloud_checkpoint']=cloud_checkpoint
        self.metadata['checkpoint_step']=step
        self.metadata['local_checkpoint_recovered']=local.is_dir()
        if ready['server_sha256']!=hashlib.sha256(Path(__file__).with_name('office_remote_policy_server.py').read_bytes()).hexdigest():
            self.close();raise ValueError('Remote server source hash differs from the prepared local file')
        (self.session/'handshake.json').write_text(json.dumps(self.metadata,indent=2))
    def read_reply(self,timeout=120):
        if not self.selector.select(timeout):raise TimeoutError('Bounded remote inference reply timed out')
        line=self.child.stdout.readline()
        if not line:raise RuntimeError('Owned remote inference connection exited')
        row=json.loads(line)
        if row.get('error'):raise RuntimeError(row['error'])
        return row
    def request(self,row):
        self.counter+=1;row['id']=self.counter
        self.child.stdin.write(json.dumps(row,separators=(',',':'))+'\n');self.child.stdin.flush()
        result=self.read_reply()
        if result['id']!=self.counter:raise RuntimeError('Remote inference request/reply mismatch')
        return result
    def set_rng_seed(self,seed):self.seed=seed
    def reset(self):
        result=self.request({'kind':'reset','seed':self.seed})
        if not result.get('reset') or result.get('rng_seed')!=self.seed:raise ValueError('Remote Torch RNG reset was not acknowledged')
        with (self.session/'rng-reset-acknowledgments.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
    def predict(self,observation,instruction):
        def encode(rgb):
            if rgb.dtype!=np.uint8 or rgb.shape!=(256,256,3):raise ValueError('Require native uint8 256px camera observations')
            data=io.BytesIO();Image.fromarray(rgb).save(data,format='PNG');return base64.b64encode(data.getvalue()).decode()
        result=self.request({'kind':'predict','scene_png':encode(observation['scene']),'wrist_png':encode(observation['wrist']),
                            'state':np.asarray(observation['state'],dtype=np.float32).tolist(),'instruction':instruction})
        self.cloud_times.append(result['cloud_inference_seconds']);return np.asarray(result['actions'],dtype=np.float32)
    def close(self):
        try:
            self.child.stdin.write(json.dumps({'kind':'close'})+'\n');self.child.stdin.flush();self.child.stdin.close()
            self.child.wait(timeout=15)
        except (OSError,subprocess.TimeoutExpired):self.child.terminate();self.child.wait(timeout=5)
        self.stderr.close();self.selector.close()
        (self.session/'transport-summary.json').write_text(json.dumps({'cloud_inference_seconds':self.cloud_times,'requests':len(self.cloud_times),'owned_ssh_returncode':self.child.returncode},indent=2))


def make_policy(checkpoint,device):return RemoteGroot(checkpoint,device)
