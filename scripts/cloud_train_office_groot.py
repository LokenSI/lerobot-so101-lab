"""Observations-only learned GR00T adapter for scripts/office_evaluate.py."""
from pathlib import Path
import hashlib
from cloud_train_policy import CloudPolicy

class OfficeGroot:
    def __init__(self, checkpoint, device):
        root=Path(checkpoint).resolve()
        manifest=root/'export-ready.json'
        config=root/'config.json'
        if not config.is_file():raise ValueError('GR00T checkpoint lacks model config')
        if manifest.is_file():
            checkpoint_hash=hashlib.sha256(manifest.read_bytes()).hexdigest();hash_scope='complete export manifest'
        else:
            shards=sorted(root.glob('*.safetensors'))
            if not shards:raise ValueError('Checkpoint has no saved model weights')
            overall=hashlib.sha256()
            for shard in shards:
                one=hashlib.sha256()
                with shard.open('rb') as stream:
                    for block in iter(lambda:stream.read(4*1024*1024),b''):one.update(block)
                overall.update((shard.name+':'+one.hexdigest()+'\n').encode())
            checkpoint_hash=overall.hexdigest();hash_scope='sorted safetensors shard filename/SHA256 aggregate'
        self.inner=CloudPolicy('groot', checkpoint, device)
        self.metadata={'model':'GR00T-N1.7-3B','learned_policy':True,
                       'action_units':'radians','checkpoint':str(Path(checkpoint).resolve()),
                       'checkpoint_sha256':checkpoint_hash,'checkpoint_hash_scope':hash_scope,
                       'model_config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
                       'control':'Native decoded absolute six-joint actions, no scripted correction'}
    def reset(self):
        self.inner.reset()
    def predict(self, observation, instruction):
        return self.inner.predict(observation['scene'],observation['wrist'],observation['state'],instruction)

def make_policy(checkpoint, device):
    return OfficeGroot(checkpoint, device)
