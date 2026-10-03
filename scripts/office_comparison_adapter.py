"""Frozen-contract SSH transport; all selected policies infer on the same A100."""
import base64,hashlib,io,json,os,selectors,shlex,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image
import office_ssh_transport as brev_remote
from office_remote_groot_adapter import RemoteGroot,windows_path

ROOT=Path(__file__).resolve().parents[1]
BASE=os.environ.get('OFFICE_REMOTE_BASE','/tmp/office-comparison')


class ComparisonPolicy(RemoteGroot):
    def __init__(self,checkpoint,device):
        if device!='cpu':raise ValueError('Comparison local transport/simulator requires CPU; model inference is remote')
        contract_path=Path(checkpoint).resolve();contract=json.loads(contract_path.read_text())
        self.kind=os.environ['OFFICE_COMPARISON_MODEL'];model=contract['models'][self.kind]
        runtime=json.loads(contract_path.with_name('execution-config.private.json').read_text())[self.kind]
        self.seed=510000;self.counter=0;self.cloud_times=[];self.last_server_inference_seconds=0.;self.last_input_sha256={}
        name=os.environ.get('OFFICE_REMOTE_SESSION',self.kind+'-comparison-'+str(int(time.time())))
        self.session=ROOT/'runtime/action-training-phase/office/remote-sessions'/name;self.session.mkdir(parents=True,exist_ok=False)
        self.stderr=(self.session/'cloud-stderr.log').open('w');destination,port=brev_remote.connection()
        options=[]
        for value in brev_remote.options():
            if value.startswith(('UserKnownHostsFile=','-oUserKnownHostsFile=')):value=value.split('=',1)[0]+'='+windows_path(value.split('=',1)[1])
            else:value=windows_path(value)
            options.append(value)
        ssh='/mnt/c/Windows/System32/OpenSSH/ssh.exe' if os.name=='posix' else 'ssh'
        report=BASE+'/'+name+'-report.json'
        env={'PYTHONPATH':'/home/ubuntu/workspace/action-training/package/scripts:'+BASE+'/scripts',
             'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'0','NO_ALBUMENTATIONS_UPDATE':'1',
             'TORCH_HOME':'/data/action-training/act/torch-cache'}
        command='cd '+shlex.quote(runtime['cwd'])+' && env '+' '.join(key+'='+shlex.quote(value) for key,value in env.items())+' '+shlex.quote(runtime['python'])+' '+BASE+'/scripts/office_comparison_server.py --model '+self.kind+' --checkpoint '+shlex.quote(runtime['checkpoint'])+' --report '+shlex.quote(report)+' --max-seconds 1500'
        self.child=subprocess.Popen([ssh,'-T',*options,'-p',str(port),destination,command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,bufsize=1)
        self.selector=selectors.DefaultSelector();self.selector.register(self.child.stdout,selectors.EVENT_READ)
        try:
            ready=self.read_reply(300)
            if not ready.get('ready') or ready['checkpoint_sha256']!=model['expected_weight_aggregate']:raise ValueError('Remote checkpoint differs from frozen comparison contract')
            if ready['source_revision']!=model['source_pin']:raise ValueError('Remote source pin differs from frozen comparison contract')
            if ready['server_sha256']!=hashlib.sha256(Path(__file__).with_name('office_comparison_server.py').read_bytes()).hexdigest():raise ValueError('Remote comparison server source differs')
        except Exception:self.child.terminate();self.child.wait(timeout=10);self.stderr.close();self.selector.close();raise
        self.metadata={**ready,'learned_policy':True,'action_units':'radians','checkpoint':runtime['checkpoint'],
            'frozen_contract_sha256':hashlib.sha256(contract_path.read_bytes()).hexdigest(),'session':str(self.session),'remote_report':report,
            'inference_scope':'same existing A100 sequential inference; local CPU OSMesa physics/rendering',
            'training_scope':model['training_scope'],'control':'native policy-decoded commands before common simulator supervisor',
            'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (self.session/'handshake.json').write_text(json.dumps(self.metadata,indent=2))
        for name in ['office_comparison_adapter.py','office_comparison_server.py']:
            (self.session/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    def predict(self,observation,instruction):
        state=np.asarray(observation['state'],dtype=np.float32)
        def encode(rgb):
            if rgb.dtype!=np.uint8 or rgb.shape!=(256,256,3):raise ValueError('Expected unchanged native256uint8 camera image')
            target=io.BytesIO();Image.fromarray(rgb).save(target,format='PNG');return base64.b64encode(target.getvalue()).decode()
        expected={'scene':hashlib.sha256(observation['scene'].tobytes()).hexdigest(),'wrist':hashlib.sha256(observation['wrist'].tobytes()).hexdigest(),
                  'state_float32':hashlib.sha256(state.tobytes()).hexdigest(),'instruction_utf8':hashlib.sha256(instruction.encode()).hexdigest()}
        result=self.request({'kind':'predict','scene_png':encode(observation['scene']),'wrist_png':encode(observation['wrist']),
                             'state':state.tolist(),'instruction':instruction})
        if result['input_sha256']!=expected:raise ValueError('Actual remote policy inputs differ from local pixel/state/text hashes')
        self.last_input_sha256=expected;self.last_server_inference_seconds=result['cloud_inference_seconds'];self.cloud_times.append(self.last_server_inference_seconds)
        return np.asarray(result['actions'],dtype=np.float32)


def make_policy(checkpoint,device):return ComparisonPolicy(checkpoint,device)
