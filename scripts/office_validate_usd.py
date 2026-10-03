"""Parse office replay layers using installed Isaac USD libraries, without starting Kit."""
import argparse,json,os,sys
from pathlib import Path
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sdk',type=Path,required=True);p.add_argument('--layers',type=Path,nargs='+',required=True);a=p.parse_args()
    libs=next((a.sdk/'extscache').glob('omni.usd.libs-*'));sys.path.insert(0,str(libs))
    handles=[os.add_dll_directory(str(folder)) for folder in [libs/'bin',a.sdk/'kit',a.sdk/'kit/python'] if folder.exists()]
    from pxr import Usd
    rows=[]
    for layer in a.layers:
        stage=Usd.Stage.Open(str(layer));assert stage and stage.GetDefaultPrim().IsValid()
        rows.append({'layer':str(layer),'prims':sum(1 for _ in stage.Traverse()),'default_prim':str(stage.GetDefaultPrim().GetPath()),'end_time':stage.GetEndTimeCode()})
    print(json.dumps({'scope':'CPU USD parse only; no Kit, physics, or renderer launch','passed':True,'layers':rows}))
if __name__=='__main__':main()
