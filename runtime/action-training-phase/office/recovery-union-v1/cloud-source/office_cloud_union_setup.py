"""Verify and unpack the authorized shared dataset in the owned cloud task folder."""
import hashlib,json,shutil,tarfile
from pathlib import Path

ROOT=Path('/home/ubuntu/workspace/action-training/groot-eval')
def main():
    archive=ROOT/'union-data-v1.tar.gz';target=ROOT/'recovery-union-v1'
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest!='2589fc52c2319dee15362ff0d315d275a8af32ffc901d240b835b330ad848a3d':raise ValueError('Dataset transfer SHA mismatch')
    resumed=target.exists()
    if (target/'setup-report.json').exists():raise FileExistsError('Setup already completed')
    target.mkdir(exist_ok=True)
    with tarfile.open(archive) as package:
        for member in package.getmembers():
            resolved=(target/member.name).resolve()
            if not resolved.is_relative_to(target.resolve()) or member.issym() or member.islnk():raise ValueError('Unsafe archive member')
        if resumed:
            for member in package.getmembers():
                if member.isfile() and hashlib.sha256(package.extractfile(member).read()).hexdigest()!=hashlib.sha256((target/member.name).read_bytes()).hexdigest():raise ValueError('Existing extraction differs from archive')
        else:package.extractall(target,filter='data')
    shutil.copy2(ROOT/'union-training-recipe-v1.json',target/'recipe.json')
    report={'scope':'CPU data-transfer setup; no training','archive_sha256':digest,'files':sum(path.is_file() for path in target.rglob('*')),'datasets':{},
      'initial_setup_error':'Expected dataset-v1 prefix absent; archive contains verified direct format roots' if resumed else None,'existing_extraction_all_file_hashes_reverified':resumed}
    for name in ['lerobot-v3','groot-v2']:
        folder=target/name/'train';info=json.loads((folder/'meta/info.json').read_text())
        if info['total_episodes']!=18 or info['total_frames']!=8730:raise ValueError('Unexpected shared dataset scope')
        report['datasets'][name]={'episodes':info['total_episodes'],'frames':info['total_frames'],'info_sha256':hashlib.sha256((folder/'meta/info.json').read_bytes()).hexdigest()}
    (target/'setup-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
