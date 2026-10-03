"""Encode already rendered native frames with honest evidence labels (CPU only)."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import imageio.v2 as imageio
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--review-contract',type=Path);a=p.parse_args();report=json.loads((a.frames/'native-render-report.json').read_text())
    if not report['complete']:raise ValueError('Incomplete native capture; retain failure report rather than make success video')
    a.output.parent.mkdir(parents=True,exist_ok=True);cells=json.loads(a.review_contract.read_text())['cells'] if a.review_contract else []
    fps=30/report['pose_stride']
    with imageio.get_writer(str(a.output),fps=fps,codec='libx264',quality=8,macro_block_size=1) as writer:
        for path in sorted(a.frames.glob('frame-*.ppm')):
            original=Image.open(path).convert('RGB')
            legend_rows=(len(cells)+1)//2
            frame=Image.new('RGB',(original.width,original.height+48+14*legend_rows));frame.paste(original,(0,48));draw=ImageDraw.Draw(frame)
            draw.text((4,4),'ISAAC RTX REPLAY | measured MuJoCo poses',fill='white');draw.text((4,20),'No new policy evaluation or domain-transfer result',fill='yellow')
            if not report.get('covers_final_source_pose'):draw.text((frame.width-110,4),'TEASER CROP',fill='orange')
            for i,cell in enumerate(cells):
                policy=cell.get('policy') or {};tag='EXPERT' if cell['expert'] else policy.get('model','CHECKPOINT')
                checkpoint=policy.get('checkpoint','');numbers=re.findall(r'(?:checkpoint-|stage-|01-)(\d+)',checkpoint)
                steps=int(numbers[-1]) if numbers else None
                if 'smol-extension' in checkpoint and steps is not None:steps+=1000
                if steps is not None:tag+=f' step{steps}'
                condition=f" {cell['chunk_execution']}ticks" if cell.get('chunk_execution') else ''
                text=f"Cell{cell['cell']}: {tag}{condition} {cell['instruction']} | {'PASS' if cell['success'] else 'FAIL'}"
                draw.text((4+(i%2)*(frame.width//2),48+original.height+14*(i//2)),text[:max(52,frame.width//13)],fill='lime' if cell['success'] else 'orange')
            writer.append_data(np.asarray(frame))
    a.output.with_suffix('.provenance.json').write_text(json.dumps({'scope':report['scope'],'native_render_report':report,'review_contract':str(a.review_contract),
        'fps':fps,'video_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
