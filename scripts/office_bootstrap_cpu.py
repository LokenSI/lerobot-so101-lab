"""Create an isolated CPU office tool environment; does not download policies/assets."""
import argparse,subprocess,sys,venv
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--environment',type=Path,default=Path('.venv-office-cpu'));a=p.parse_args()
    if a.environment.exists():raise FileExistsError('Choose a fresh environment')
    venv.EnvBuilder(with_pip=True).create(a.environment)
    python=a.environment/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    subprocess.run([str(python),'-m','pip','install','mujoco==3.6.0','numpy==2.5.3','Pillow==12.3.0','imageio==2.38.0','imageio-ffmpeg==0.6.0','PyOpenGL==3.1.10'],check=True)
    print('CPU tools prepared. Use existing repository SO-101 asset bootstrap; configure EGL or OSMesa explicitly for rendering. No model/checkpoint downloads were performed.')

if __name__=='__main__':main()
