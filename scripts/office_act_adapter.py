"""Two independent learned ACT controllers, selected by an explicit external router.

ACT does not interpret language. Only two exact known instructions are accepted;
unknown instructions raise a clarification request rather than guessing a task.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from office_act_train import TASKS

def select_task(instruction):
    matches=[task for task,text in TASKS.items() if instruction==text]
    if len(matches)!=1:
        raise ValueError('Unknown ACT task. Clarify which supported instruction to run: '+ ' | '.join(TASKS.values()))
    return matches[0]

class Policy:
    def __init__(self,checkpoint,device='cuda'):
        root=Path(checkpoint).resolve();selection=json.loads((root/'selector.json').read_text())
        if selection.get('learned_language') is not False or set(selection['checkpoints'])!=set(TASKS):raise ValueError('Explicit two-task external selector required')
        updates=selection.get('office_updates_by_task',{})
        if set(updates)!=set(TASKS) or any(type(v) is not int or v<0 for v in updates.values()):raise ValueError('Explicit actual office-update counters required')
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors
        import torch
        self.torch=torch;self.models={};hashes={}
        for task,relative in selection['checkpoints'].items():
            path=(root/relative).resolve()
            if not path.is_relative_to(root):raise ValueError('Checkpoint path escapes ACT bundle')
            config=ACTConfig.from_pretrained(str(path));config.device=device
            policy=ACTPolicy.from_pretrained(str(path),config=config).to(device).eval()
            pre,post=make_pre_post_processors(config,pretrained_path=str(path),preprocessor_overrides={'device_processor':{'device':device}})
            self.models[task]=(policy,pre,post)
            value=hashlib.sha256()
            with (path/'model.safetensors').open('rb') as stream:
                for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
            hashes[task]=value.hexdigest()
        self.metadata={'model':'ACT task-specific pair' if min(updates.values())>0 else 'ACT zero-office-update initialization',
            'learned_policy':min(updates.values())>0,'learned_language':False,'office_updates_by_task':updates,
            'external_task_selector':'Exact original instruction maps to independently trained left/right policy; no neural language understanding',
            'action_units':'radians','checkpoint':str(root),'model_sha256_by_task':hashes,
            'checkpoint_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
            'observation':'Current scene/wrist RGB and measured joints only; text used solely by declared external selector'}
        self.selected=None
    def reset(self):
        for policy,_,_ in self.models.values():policy.reset()
        self.selected=None
    def predict(self,observation,instruction):
        task=select_task(instruction)
        if self.selected is not None and self.selected!=task:raise ValueError('Task changed mid-episode; reset before switching ACT controller')
        self.selected=task;policy,pre,post=self.models[task];t=self.torch
        state=np.asarray(observation['state'],dtype=np.float32)
        if state.shape!=(6,) or not np.isfinite(state).all():raise ValueError('Finite six-radian measured state required')
        raw={'observation.state':t.from_numpy(state.copy())}
        for camera in ['scene','wrist']:
            rgb=observation[camera]
            if rgb.dtype!=np.uint8 or rgb.shape!=(256,256,3):raise ValueError('Expected native256square uint8 RGB')
            raw['observation.images.'+camera]=t.from_numpy(rgb.transpose(2,0,1).copy()).float()/255
        # No task/text tokens are passed to ACT and no actuator clipping is hidden.
        with t.inference_mode():action=post(policy.predict_action_chunk(pre(raw)))
        result=action.float().cpu().numpy().reshape(-1,6)
        if not np.isfinite(result).all():raise ValueError('Nonfinite learned ACT command')
        return result

def make_policy(checkpoint,device='cuda'):
    return Policy(checkpoint,device)
