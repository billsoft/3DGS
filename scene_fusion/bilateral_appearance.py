"""Surface-aware 3D color graph filtering; geometry, opacity and SH-rest stay fixed.

Not atmospheric dehazing: overlapping translucent geometry is not corrected by
color smoothing. No screen-space processing or object-specific masks are used.
"""
import argparse,json,os,shutil
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from core import write_json
from gaussian_geometry import shortest_axis_normals


def filter_colors(base,sh,denoise=.25,sharpen=.5,k=32):
    if len(base)<2 or k<3 or not 0<=denoise<=1 or not 0<=sharpen<=1:
        raise ValueError('Require at least two points, k >= 3, and strengths in [0,1]')
    xyz=base[:,:3];color=sh[:,0]*.2820947918+.5
    normal=shortest_axis_normals(base);scales=np.sort(np.exp(base[:,7:10]),axis=1)
    reliable=scales[:,0]<scales[:,1]*.65
    tree=cKDTree(xyz);sample=xyz[::max(1,len(xyz)//10000)]
    probe=tree.query(sample,k=2,workers=4)[0][:,1];positive=probe[probe>0]
    if not len(positive):raise ValueError('No distinct Gaussian centers')
    cap=float(np.median(positive)*12);out=color.copy();supported=0
    for start in range(0,len(base),8192):
        ids=np.arange(start,min(start+8192,len(base)))
        dist,idx=tree.query(xyz[ids],k=min(k,len(base)),workers=4)
        rows=np.arange(len(ids));self_col=np.argmax(idx==ids[:,None],axis=1)
        idx[rows,self_col]=idx[:,0];dist[rows,self_col]=dist[:,0]
        idx[:,0]=ids;dist[:,0]=0
        delta=xyz[idx]-xyz[ids,None];radius=np.minimum(dist[:,-1],cap).clip(1e-10)
        cosine=np.abs(np.sum(normal[ids,None]*normal[idx],axis=-1))
        plane=np.maximum(abs(np.sum(delta*normal[ids,None],axis=-1)),abs(np.sum(delta*normal[idx],axis=-1)))
        chroma=np.linalg.norm(color[idx]-color[ids,None],axis=-1)
        viewdiff=np.sqrt(np.mean((sh[idx,1:]-sh[ids,None,1:])**2,axis=(2,3)))
        thickness=np.minimum(radius[:,None]*.25,2*np.minimum(scales[ids,None,0],scales[idx,0]))
        gate=(cosine>.85)&(plane<thickness)&(dist<=radius[:,None])&reliable[idx]&reliable[ids,None]&(viewdiff<.2)
        weight=np.exp(-.5*(dist/radius[:,None])**2-.5*(chroma/.06)**2-.5*(plane/(radius[:,None]*.12))**2)*gate
        weight[:,0]=1
        def average(w):return np.sum(w[:,:,None]*color[idx],axis=1)/w.sum(1)[:,None]
        broad=average(weight);near=weight.copy();near[:,max(2,k//3):]=0;fine=average(near)
        support=(weight[:,1:].sum(1)>3)&reliable[ids];supported+=int(support.sum())
        # Denoise first; sharpen the surviving middle-band detail, not raw noise.
        detail=fine-broad;amount=np.clip(np.linalg.norm(detail,axis=1)/.015,0,1)
        update=denoise*(fine-color[ids])+sharpen*detail*amount[:,None]
        update=np.clip(update,-.025,.025);out[ids]+=update*support[:,None]
    result=sh.copy();result[:,0]=(out.clip(0,1)-.5)/.2820947918
    # Unsupported/unchanged points retain bit-identical coefficients.
    same=np.all(out==color,axis=1);result[same,0]=sh[same,0]
    b=base.copy();b[:,3:6]=result[:,0]
    return b,result,dict(points=len(base),supported=supported,denoise=denoise,sharpen=sharpen,
                         max_rgb_change=float(abs(out-color).max()),radius_cap=cap,
                         geometry_opacity_and_directional_SH_unchanged=True)


def run(a):
    if a.out.exists():raise ValueError('Use a fresh output directory')
    z=np.load(a.reference/'optimized.npz');b,s,report=filter_colors(z['base'],z['sh'],a.denoise,a.sharpen)
    a.out.mkdir(parents=True);np.savez(a.out/'optimized.npz',base=b,sh=s);write_json(a.out/'filter.json',report)
    data=a.out/'data';data.mkdir();meta=json.loads((a.reference/'data/dataset.json').read_text('utf8'))
    meta['comparison_model']=str((a.reference/'optimized.npz').resolve());meta['live_teacher_required']=True
    for p in (a.reference/'data').glob('teacher-*.npy'):
        try:os.link(p,data/p.name)
        except OSError:shutil.copy2(p,data/p.name)
    write_json(data/'dataset.json',meta);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--denoise',type=float,default=.25);p.add_argument('--sharpen',type=float,default=.5)
    run(p.parse_args())
