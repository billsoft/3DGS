"""Export a geometry-only oriented PLY from smoothed original-station depths.

This is a point cloud for surface reconstruction, not a Gaussian display PLY.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from planar_depth import backproject

def frame_points(depth,alpha,viewmat,step=4):
    p,n,_,center=backproject(depth,viewmat)
    valid=np.isfinite(p).all(2)&np.isfinite(n).all(2)&(depth>.1)&(depth<30)&(alpha>=.95)&(np.linalg.norm(n,axis=2)>.5)
    # Normals across a foreground/background discontinuity are not surfaces.
    for axis in (0,1):
        a=[slice(None)]*2;b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1);a,b=tuple(a),tuple(b)
        edge=abs(depth[a]-depth[b])>.05*np.minimum(depth[a],depth[b])
        valid[a]&=~edge;valid[b]&=~edge
    n*=np.where(((p-center)*n).sum(2)>0,-1.,1.)[:,:,None]
    ids=valid[::step,::step]
    return p[::step,::step][ids],n[::step,::step][ids]

def run(a):
    if a.out.exists():raise ValueError('Do not overwrite an existing geometry guide')
    meta=json.loads((a.source/'capture.json').read_text('utf8'));parts=[]
    for f in meta['frames']:
        if not f['original_camera']:continue
        z=np.load(a.source/f['depth']);p,n=frame_points(z['depth'],z['alpha'],np.array(f['viewmat']),a.step)
        parts.append(np.c_[p,n].astype('f'))
    points=np.concatenate(parts);data=np.empty(len(points),dtype=[(k,'<f4') for k in ('x','y','z','nx','ny','nz')])
    for i,k in enumerate(data.dtype.names):data[k]=points[:,i]
    a.out.parent.mkdir(exist_ok=True,parents=True)
    header='ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(data))+'\n'+''.join('property float '+k+'\n' for k in data.dtype.names)+'end_header\n'
    with a.out.open('wb') as stream:stream.write(header.encode('ascii'));data.tofile(stream)
    report=dict(points=len(data),original_stations_only=True,step=a.step,geometry_only=True,gaussian_ply=False,
        smoothing=meta.get('geometry_smoothing'),source_model_sha256=meta['model_sha256'],sha256=hashlib.sha256(a.out.read_bytes()).hexdigest())
    a.out.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--step',type=int,default=4)
    a=p.parse_args()
    if a.step<1:p.error('step must be positive')
    run(a)
