"""Check whether collider depth agrees with each calibrated original photograph."""
import os
import numpy as np,json,cv2 as cv
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
from embreex.rtcore_scene import EmbreeScene
from embreex.mesh_construction import TriangleMesh
s=spec_from_file_location('f','four-fuse.py');f=module_from_spec(s);s.loader.exec_module(f)
room=Path('four-room');inputs=Path(os.environ.get('GS_INPUT_DIR', 'data/input'))
reg=json.loads((room/'registration.json').read_text());meta=json.loads((room/'fusion.json').read_text())
def read_mesh(path):
    with path.open('rb') as file:
        counts={}
        while (line:=file.readline()).strip()!=b'end_header':
            if not line:raise ValueError('Missing header')
            if line.startswith(b'element '):
                _,label,count=line.split();counts[label]=int(count)
        verts=np.fromfile(file,np.dtype([('p','<f8',3),('c','u1',3)]),counts[b'vertex'])
        faces=np.fromfile(file,np.dtype([('n','u1'),('f','<u4',3)]),counts[b'face'])
    if np.any(faces['n']!=3):raise ValueError('Expected triangles')
    return verts['p'],verts['c'],faces['f']
def scene_for(p,tri):
    scene=EmbreeScene()
    mesh=TriangleMesh(scene,np.ascontiguousarray(p,'f'),np.ascontiguousarray(tri,'i'))
    return scene,mesh
def depth_at(scene,center,basis,size):
    y,x=np.indices((size,size),dtype='f');coords=np.stack([(x+.5-size*.5)/(size*.5),(y+.5-size*.5)/(size*.5),np.ones_like(x)],-1)
    d=np.ascontiguousarray(coords.reshape(-1,3)@basis.T,'f')
    starts=np.ascontiguousarray(np.tile(center,(len(d),1)),'f')
    hit=scene.run(starts,d,query='DISTANCE').reshape(size,size)
    return hit
def audit():
    folder=room/'colliders';folder.mkdir(exist_ok=True)
    reports={}
    for n,m in zip(reg['stations'],np.array(reg['matrices'])):
        p,c,tri=read_mesh(inputs/f'{n}m.ply')
        scene,mesh=scene_for(p,tri)
        cam=np.load(room/n/'cache.npz');R,C=cam['R'],cam['C']
        results=[]
        for k,b in enumerate(f.prep.FACES):
            z=np.load(room/n/f'depth-{k}.npz')
            depth=depth_at(scene,C,R.T@b,640)
            valid=(depth<1e8)&(z['alpha']>.97)&(z['depth']>.1)
            err=abs(depth-z['depth'])/np.maximum(.1,z['depth'])
            results.append((int(valid.sum()),float(np.median(err[valid])) if valid.any() else None))
            np.save(folder/f'{n}-{k}.npy',depth)
        reports[n]=results
        print(n,results,flush=True)
    (folder/'audit.json').write_text(json.dumps(reports,indent=2))
if __name__=='__main__':audit()
