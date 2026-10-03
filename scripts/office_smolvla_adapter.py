"""Official LeRobot SmolVLA adapter using checkpoint-owned pre/post processors."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np

class Policy:
    def __init__(self,checkpoint,device='cuda'):
        import torch
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        self.torch=torch;self.device=device;self.checkpoint=Path(checkpoint)
        config=SmolVLAConfig.from_pretrained(checkpoint);config.device=device
        self.policy=SmolVLAPolicy.from_pretrained(checkpoint,config=config).to(device).eval()
        self.pre,self.post=make_pre_post_processors(config,pretrained_path=checkpoint,
            preprocessor_overrides={'device_processor':{'device':device}})
        digest=hashlib.sha256()
        with (self.checkpoint/'model.safetensors').open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        self.metadata={'learned_policy':True,'language_conditioned':True,'model':'SmolVLA','checkpoint':str(checkpoint),
                       'checkpoint_sha256':digest.hexdigest(),'action_units':'radians',
                       'observation':'scene/wrist RGB, measured six joints, instruction; no simulator truth'}
    def reset(self):self.policy.reset()
    def predict(self,observation,instruction):
        t=self.torch
        raw={'observation.state':t.from_numpy(observation['state'].astype(np.float32)),'task':instruction}
        for c in ['scene','wrist']:raw[f'observation.images.{c}']=t.from_numpy(observation[c].transpose(2,0,1).copy()).float()/255
        with t.inference_mode(): action=self.post(self.policy.predict_action_chunk(self.pre(raw)))
        return action.float().cpu().numpy().reshape(-1,6)

def make_policy(checkpoint,device='cuda'):return Policy(checkpoint,device)
