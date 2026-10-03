"""Deployment contract shared by cloud reload probes and learned MuJoCo rollouts.

No simulator state access, IK, correction policy, or actuator clipping here.
Input=current RGB+measured state+instruction; output=absolute six joint radians.
"""
from pathlib import Path
import numpy as np

class CloudPolicy:
    def __init__(self,model,checkpoint,device='cuda'):
        self.model_type=model;self.checkpoint=Path(checkpoint)
        if model=='groot':
            from gr00t.policy.gr00t_policy import Gr00tPolicy
            self.policy=Gr00tPolicy('NEW_EMBODIMENT',str(checkpoint),device=device,strict=True)
        elif model=='smolvla':
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.factory import make_pre_post_processors
            self.policy=SmolVLAPolicy.from_pretrained(str(checkpoint)).to(device).eval()
            self.policy.config.device=device
            self.pre,self.post=make_pre_post_processors(self.policy.config,pretrained_path=str(checkpoint),
                preprocessor_overrides={'device_processor':{'device':device}})
        else:raise ValueError(model)
    def reset(self):
        self.policy.reset()
    def predict(self,scene,wrist,state,task):
        import torch
        state=np.asarray(state,dtype=np.float32)
        if state.shape!=(6,) or not np.isfinite(state).all():raise ValueError('Expected finite six joint radians')
        for rgb in (scene,wrist):
            if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3:raise ValueError('Expected uint8 HWC RGB')
        with torch.inference_mode():
            if self.model_type=='groot':
                obs={'video':{'scene':scene[None,None],'wrist':wrist[None,None]},
                    'state':{'single_arm':state[None,None,:5],'gripper':state[None,None,5:]},
                    'language':{'annotation.human.task_description':[[task]]}}
                action,_=self.policy.get_action(obs)
                result=np.concatenate([action['single_arm'][0],action['gripper'][0]],axis=-1)
            else:
                obs={'observation.state':torch.from_numpy(state.copy()),'task':task,
                    'observation.images.scene':torch.from_numpy(scene.transpose(2,0,1).copy()).float()/255,
                    'observation.images.wrist':torch.from_numpy(wrist.transpose(2,0,1).copy()).float()/255}
                batch=self.pre(obs)
                action=self.policy.predict_action_chunk(batch)
                result=self.post(action)[0].detach().cpu().numpy()
        if result.ndim!=2 or result.shape[-1]!=6 or not np.isfinite(result).all():raise ValueError('Nonfinite/malformed decoded commands')
        return result.astype(np.float32)
