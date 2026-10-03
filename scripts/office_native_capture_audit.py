"""CPU audit of actual native visual replay captures and their source manifests."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OFFICE=ROOT/'runtime/action-training-phase/office'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    rows=[]
    for name in ['native-development-v3-frames','native-legible-overview-frames','native-legible-closeup-frames']:
        folder=OFFICE/name;report=json.loads((folder/'native-render-report.json').read_text());guard=json.loads((folder/'guard-report.json').read_text())
        frames=[{'name':path.name,'sha256':sha(path)} for path in sorted(folder.glob('frame-*.ppm'))]
        if not report['complete'] or not report['covers_final_source_pose'] or guard['admission']!='PASS' or guard['returncode']!=0:raise ValueError('Native capture evidence failed')
        if len(frames)!=report['rendered_frames'] or len({row['sha256'] for row in frames})<2:raise ValueError('Missing or entirely static captured frames')
        rows.append({'folder':name,'report_sha256':sha(folder/'native-render-report.json'),'guard_sha256':sha(folder/'guard-report.json'),
          'source_stage_sha256':report['stage_sha256'],'frames':frames,'distinct_frame_hashes':len({row['sha256'] for row in frames}),
          'first_pose':report['pose_frame_indices'][0],'last_pose':report['pose_frame_indices'][-1],'gpu_peak_mib':guard['peak_gpu_mib'],'ram_peak_mib':guard['peak_host_mib']})
    result={'scope':'CPU evidence audit of actual RTX visual replay, not physics/policy success','passed':True,'captures':rows,
      'legacy_preparation_status':'The legibility manifest originally recorded NOT_EXECUTED at CPU preparation. These later capture reports establish execution; original preparation file remains unchanged.',
      'visual_quality':'Original640 overview crowded. New1920 overview removes walls and colors robot, but arms remain small with blank upperframe; matched closeup shows the robot clearly. No cinematic-quality claim.',
      'local_gpu_released':True,'physics_executed':False}
    target=OFFICE/'native-capture-audit.json'
    if target.exists():raise FileExistsError('Retain capture audit')
    target.write_text(json.dumps(result,indent=2));print(json.dumps({'passed':True,'captures':len(rows),'frames':sum(len(row['frames']) for row in rows)}))

if __name__=='__main__':main()
