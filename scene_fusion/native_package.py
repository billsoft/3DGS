"""Package byte-identical source PLYs for N-camera native-view playback."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from core import load,write_json
from input_io import header

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def run(a):
    scene=load(a.scene);cams=np.load(a.scene/'aligned-cameras.npz')
    data=json.loads((a.scene/'data-512/dataset.json').read_text(encoding='utf8'))
    a.out.mkdir(parents=True,exist_ok=True);sources=[]
    for i,s in enumerate(scene['stations']):
        src=Path(s['ply']);sha=digest(src)
        if sha!=s['ply_sha256']:raise ValueError('Original PLY changed after calibration')
        norm=json.loads((a.scene/s['id']/'normalization.json').read_text(encoding='utf8'))
        n=np.eye(4);n[:3,:3]*=norm['scale'];n[:3,3]=-np.array(norm['origin'])*norm['scale']
        world=cams['matrices'][i]@n;scale=np.cbrt(np.linalg.det(world[:3,:3]));rotation=world[:3,:3]/scale
        if scale<=0 or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6):raise ValueError('Source transform must be a positive similarity')
        dst=a.out/(s['id']+'.ply')
        if not dst.exists() or digest(dst)!=sha:shutil.copyfile(src,dst)
        if digest(dst)!=sha:raise ValueError('Source copy changed bytes')
        with dst.open('rb') as f:_,count,_=header(f)
        sources.append(dict(id=s['id'],label=s['label'],file=dst.name,sha256=sha,points=count,world_from_raw=world.tolist(),local_capture=(np.linalg.inv(world)@np.r_[cams['C'][i],1])[:3].tolist()))
    base=json.loads((a.baseline/'fusion.json').read_text(encoding='utf8'))
    write_json(a.out/'native.json',dict(version=1,stations=sources,centers=cams['C'].tolist(),up=base['up'],views=data['views'],midpoint_edges=data['midpoint_edges'],
        baseline=dict(scene=a.baseline.name,path='/four-room/scenes/'+a.baseline.name+'/'+base['model_folder']+'/fused'+('.ply' if base.get('sh_degree') else '.splat'),sh_degree=base.get('sh_degree',0)),
        original_files_byte_identical=True,source_point_count=sum(s['points'] for s in sources),note='Native source playback at capture centers; screen-space transition to fused baseline. Not a single fused PLY or new mesh.'))
    print(json.dumps(dict(sources=len(sources),points=sum(s['points'] for s in sources),byte_identical=True)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
