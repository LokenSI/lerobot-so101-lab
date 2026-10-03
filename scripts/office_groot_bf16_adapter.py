"""Explicit optimized source module; shared stock observation/action decoder unchanged."""
import hashlib,importlib.util,json,sys
from pathlib import Path
from cloud_train_office_groot import OfficeGroot
from cloud_train_policy import CloudPolicy

ROOT=Path(__file__).resolve().parents[1]
PREPARED=ROOT/'runtime/action-training-phase/office/groot-bf16-load'


class OfficeGrootBF16(OfficeGroot):
    def __init__(self,checkpoint,device):
        root=Path(checkpoint).resolve();manifest=json.loads((PREPARED/'manifest.json').read_text())
        source=PREPARED/'gr00t_policy_bf16.py'
        if hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['patched_source_sha256']:raise ValueError('Prepared source differs from its explicit patch manifest')
        import gr00t.policy
        name='gr00t.policy.office_explicit_bf16_policy'
        spec=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)
        # Reuse the unmodified stock CloudPolicy reset/predict decoder; only its
        # explicitly selected policy constructor comes from the saved source patch.
        self.inner=CloudPolicy.__new__(CloudPolicy);self.inner.model_type='groot';self.inner.checkpoint=root
        self.inner.policy=module.Gr00tPolicy('NEW_EMBODIMENT',str(root),device=device,strict=True)
        export_path=root/'export-ready.json';export=json.loads(export_path.read_text())
        digest=hashlib.sha256()
        for shard in sorted((row for row in export['files'] if row['path'].endswith('.safetensors')),key=lambda r:r['path']):
            if (root/shard['path']).stat().st_size!=shard['bytes']:raise ValueError('Checkpoint shard size differs from verified export manifest')
            digest.update((shard['path']+':'+shard['sha256']+'\n').encode())
        self.metadata={'model':'GR00T-N1.7-3B','learned_policy':True,'action_units':'radians','checkpoint':str(root),
                       'checkpoint_sha256':digest.hexdigest(),'checkpoint_hash_scope':'sorted SHA-verified export-manifest shard filename/SHA256 aggregate; local sizes rechecked',
                       'export_manifest_sha256':hashlib.sha256(export_path.read_bytes()).hexdigest(),
                       'model_config_sha256':hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),
                       'load_optimization':'explicit constructor BF16 dtype; stock final BF16 deployment unchanged',
                       'prepared_source_manifest':manifest,'deployment_dtype':str(self.inner.policy.model.dtype),
                       'control':'shared unchanged stock CloudPolicy decoded absolute6-radian actions; no IK/manual correction'}


def make_policy(checkpoint,device):return OfficeGrootBF16(checkpoint,device)
