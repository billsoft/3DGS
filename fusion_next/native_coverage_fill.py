"""Fill only source-visible screen holes with existing baseline Gaussians.

No native covariance is resized. This is a static candidate, not a promise
that the different source geometries become consistent.
"""
from common import *
from scipy.spatial import cKDTree

def run():
    source=ROOT/'four-room/fusion-native-fidelity'
    native=rm.f.prep.read_gs(source/'fused.ply')
    baseline=rm.f.prep.read_gs(ROOT/'four-room/baselines/2026-11-11-coverage-guard/fused.ply')
    nearest,_=cKDTree(native[:,:3]).query(baseline[:,:3]);votes=np.zeros(len(baseline),'u1')
    selected_source=np.full(len(baseline),-1,'i');best=np.full(len(baseline),np.inf)
    for i,station in enumerate(stations):
        cam=rm.Camera(station,np.linalg.inv(cams[i].inv));cam.maps=[]
        for k in range(6):
            d=np.load(rm.room/station/f'depth-{k}.npz');cam.maps.append((d['depth'],d['alpha']))
        for start in range(0,len(baseline),100000):
            ids=np.arange(start,min(start+100000,len(baseline)));p=baseline[ids,:3];q=project(p,i)
            err,conf,dep=cam.sample(cam.project(p));primary=np.argmax(abs(q),axis=1);face=2*primary+(q[np.arange(len(q)),primary]<0)
            for k,b in enumerate(rm.f.prep.FACES):
                ix=np.flatnonzero(face==k);r=q[ix]@b;xy=np.floor(192+192*r[:,:2]/np.maximum(r[:,2,None],1e-8)).astype('i').clip(0,383)
                original=np.load(source/f'{station}-native-{k}.npz')['alpha'];merged=np.load(source/f'{station}-native_owner-{k}.npz')['alpha']
                good=(original[xy[:,1],xy[:,0]]>.97)&(merged[xy[:,1],xy[:,0]]<.65)&(conf[ix]>.97)&(abs(err[ix])<.08+.02*dep[ix])&(nearest[ids[ix]]>.012)
                take=ix[good];votes[ids[take]]+=1
                score=abs(err[take])+.01*dep[take];replace=score<best[ids[take]];chosen=ids[take[replace]]
                best[chosen]=score[replace];selected_source[chosen]=i
        print('screen holes checked',station,flush=True)
    ids=np.flatnonzero(votes>0);fill=baseline[ids].copy();images=photos()
    for i in range(4):
        take=np.flatnonzero(selected_source[ids]==i)
        color=sample(images[i],uv(project(fill[take,:3],i)))[:,::-1]/255
        fill[take,3:6]=(color-.5)/.2820947918
    result=np.concatenate([native,fill]);rm.f.write(OUT/'fused.ply',result);rm.f.to_splat(OUT/'fused.splat',result)
    write_json('fill-report.json',dict(native_points=len(native),added_points=len(fill),two_view_supported_points=int((votes[ids]>=2).sum()),output_points=len(result),note='Only baseline Gaussians whose projections fall in a native-merge hole and match an observed source depth are added. No unseen backface synthesis. Global geometry conflicts remain possible.'))
    write_json('fusion.json',dict(centers=C.tolist(),up=rm.meta['up'],stations=stations))

if __name__=='__main__':run()
