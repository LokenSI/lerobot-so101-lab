"""CPU source/function proof of the adapted expert/rollout RGB resize path."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
runner=ROOT/'scripts/run_flux_so101_adapted.py'
replay=ROOT/'runtime/flux-so101-adaptation/replay_native.py'
tree=ast.parse(runner.read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='policy_rgb')
namespace={'np':np,'Image':Image}
exec(compile(ast.Module(body=[function],type_ignores=[]),'extracted-policy-rgb','exec'),namespace)
parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
render_calls=[]
for node in ast.walk(tree):
    if isinstance(node,ast.Call) and ast.unparse(node.func)=='env.render':
        parent=parents[node]
        assert isinstance(parent,ast.Call) and ast.unparse(parent.func)=='policy_rgb'
        render_calls.append(node.lineno)
assert len(render_calls)==3 # initial, each poststep, final hold
replay_tree=ast.parse(replay.read_text())
expert_resizes=[n for n in ast.walk(replay_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='resize']
assert len(expert_resizes)==2
for node in expert_resizes:
    assert ast.literal_eval(node.args[0])==(256,256) and ast.unparse(node.args[1])=='Image.Resampling.BILINEAR'
rng=np.random.default_rng(882)
samples=[rng.integers(0,256,(240,320,3),dtype=np.uint8),
         np.broadcast_to(np.arange(320,dtype=np.uint16)[None,:,None]%256,(240,320,3)).astype(np.uint8),
         np.zeros((240,320,3),dtype=np.uint8)]
rows=[]
for i,rgb in enumerate(samples):
    actual=namespace['policy_rgb'](rgb)
    expected=np.asarray(Image.fromarray(rgb).resize((256,256),Image.Resampling.BILINEAR))
    assert actual.shape==(256,256,3) and actual.dtype==np.uint8 and np.array_equal(actual,expected)
    rows.append({'sample':i,'max_uint8_error':int(np.max(np.abs(actual.astype(int)-expected.astype(int))))})
out={'scope':'CPU-only AST/extracted-function proof, no physics/render/model loaded',
     'runner_sha256':hashlib.sha256(runner.read_bytes()).hexdigest(),'replay_sha256':hashlib.sha256(replay.read_bytes()).hexdigest(),
     'all_native_render_calls_wrapped':True,'render_call_lines':render_calls,'samples':rows,
     'camera_contract':'Native320x240 projection -> identical PIL uint8 BILINEAR256x256 -> official scene/wrist side-by-side canvas',
     'native_camera_projection_re_rendered':False}
(Path(__file__).parent/'rgb-preprocessing-cpu-audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
