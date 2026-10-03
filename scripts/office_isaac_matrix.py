"""Compose baked USD replay references into a native Isaac review matrix."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replays',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--columns',type=int,default=3);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    end_frame=max(json.loads(r.with_suffix('.provenance.json').read_text())['frames']-1 for r in a.replays)
    lines=['#usda 1.0','(','    defaultPrim = "OfficeMatrix"','    upAxis = "Z"','    metersPerUnit = 1','    timeCodesPerSecond = 30',f'    endTimeCode = {end_frame}',')','def Xform "OfficeMatrix"','{']
    manifest=[]
    for i,replay in enumerate(a.replays):
        provenance=json.loads(replay.with_suffix('.provenance.json').read_text())
        uri=Path(os.path.relpath(replay.resolve(),a.output.parent.resolve())).as_posix();x=(i%a.columns)*1.5;y=(i//a.columns)*1.5
        lines.extend([f'    def Xform "cell_{i}" ( prepend references = @{uri}@ )', '    {',f'        double3 xformOp:translate = ({x}, {y}, 0)',
            '        uniform token[] xformOpOrder = ["xformOp:translate"]','    }'])
        manifest.append({'cell':i,'replay':str(replay),'evidence':provenance})
    lines.extend(['    def DomeLight "Light"','    {','        float inputs:intensity = 600','    }','}'])
    a.output.write_text('\n'.join(lines));a.output.with_suffix('.matrix.json').write_text(json.dumps({'scope':'replay matrix; all outcomes inherit original reports; no Isaac policy evaluation','cells':manifest},indent=2))
if __name__=='__main__':main()
