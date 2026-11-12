"""Scene-independent IO and camera math. No prior room cache is imported."""
from pathlib import Path
import importlib.util,json,hashlib,re,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.append(str(ROOT/'fusion_next'))
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
prep=module('scene_prepare',ROOT/'four-prepare.py')
gs=module('scene_gaussians',ROOT/'four-fuse.py')

def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')

def discover(inputs,stations=None):
    inputs=Path(inputs).resolve();pairs=[]
    if not inputs.is_dir():raise ValueError('Input directory does not exist')
    files=list(inputs.iterdir());selected={s.casefold() for s in stations} if stations else None
    for ply in sorted([f for f in files if f.suffix.lower()=='.ply'],key=lambda p:[int(s) if s.isdigit() else s.casefold() for s in re.split('(\\d+)',p.stem)]):
        if selected is not None and ply.stem.casefold() not in selected:continue
        photos=[p for p in files if p.suffix.lower() in {'.jpg','.jpeg','.png'} and p.stem.casefold() in {ply.stem.casefold(),ply.stem.casefold()+'_7680'}]
        if len(photos)!=1:raise ValueError(f'{ply.name}: expected exactly one matching panorama, found {len(photos)}')
        photo=photos[0]
        pairs.append(dict(id=f'cam-{len(pairs)+1:04d}',label=ply.stem,photo=str(photo),ply=str(ply),
            photo_sha256=hashlib.sha256(photo.read_bytes()).hexdigest(),ply_sha256=hashlib.sha256(ply.read_bytes()).hexdigest()))
    if len(pairs)<2:raise ValueError('At least two panorama/PLY pairs are required')
    if selected is not None and selected!={p['label'].casefold() for p in pairs}:raise ValueError('Requested station labels not found')
    return dict(version=2,inputs=str(inputs),count=len(pairs),stations=pairs,normalization='robust-radius-5-v1')

def initialize(inputs,out,stations=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);new=discover(inputs,stations);path=out/'scene.json'
    if path.exists() and json.loads(path.read_text(encoding='utf8'))!=new:
        raise ValueError('Input files changed: choose a new output directory instead of reusing stale calibration')
    write_json(path,new);return new

def load(out):return json.loads((Path(out)/'scene.json').read_text(encoding='utf8'))

def viewmat(center,basis):
    v=np.eye(4);v[:3,:3]=basis.T;v[:3,3]=-basis.T@center;return v

def look(center,target,up):
    z=target-center;z=z/np.linalg.norm(z);x=np.cross(z,up);x/=np.linalg.norm(x)
    return np.stack([x,np.cross(z,x),z],axis=1)
