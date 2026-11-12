"""Semantic-free edge footprint ablation on the accepted Gaussian model."""
from common import *
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import argparse,shutil

def run():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['recolor','split'],required=True);args=parser.parse_args()
    baseline=ROOT/'four-room/baselines/2026-11-11-coverage-guard'
    if OUT.resolve()==baseline.resolve():raise ValueError('Use isolated trial output')
    a=rm.f.prep.read_gs(baseline/'fused.ply')
    labels=np.load(ROOT/'four-room/baselines/2026-11-11-fusion-next/ownership.npz')
    _,face=cKDTree(labels['center']).query(a[:,:3]);owner=labels['owner'][face]
    images=photos();keep=np.ones(len(a),bool);parts=[];counts=[]
    for source in range(4):
        ids=np.flatnonzero(owner==source);total=0
        for start in range(0,len(ids),30000):
            index=ids[start:start+30000];point=a[index,:3]
            rotation=Rotation.from_quat(a[index][:,[11,12,13,10]]).as_matrix()
            x=rotation[:,:,1]*np.exp(a[index,8,None]);y=rotation[:,:,2]*np.exp(a[index,9,None])
            probes=np.stack([point,point+x,point-x,point+y,point-y],1)
            colors=sample(images[source],uv(project(probes.reshape(-1,3),source)))[:,::-1].reshape(-1,5,3)/255
            contrast=np.max(np.linalg.norm(colors-colors[:,:1],axis=2),axis=1)
            error,confidence,depth=cams[source].sample(cams[source].project(point))
            current=np.clip(a[index,3:6]*.2820947918+.5,0,1)
            good=(contrast>.18)&(confidence>.8)&(abs(error)<.05+.02*depth)&(np.linalg.norm(current-colors[:,0],axis=1)<.18)
            selected=index[good];total+=len(selected)
            if args.mode=='recolor':
                a[selected,3:6]=(colors[good,0]-.5)/.2820947918
            else:
                keep[selected]=False
                # Nine components preserve mean/covariance in the tangent plane.
                # Optical-depth mass is preserved approximately, not exact alpha.
                for ix,wx in [(-1,.25),(0,.5),(1,.25)]:
                    for iy,wy in [(-1,.25),(0,.5),(1,.25)]:
                        child=a[selected].copy();child[:,:3]+=ix*x[good]+iy*y[good]
                        child[:,8:10]+=.5*np.log(.5)
                        tau=np.logaddexp(0,child[:,6])*wx*wy/.5
                        alpha=np.clip(-np.expm1(-tau),1e-6,1-1e-6);child[:,6]=np.log(alpha/(1-alpha))
                        rgb=sample(images[source],uv(project(child[:,:3],source)))[:,::-1]/255
                        child[:,3:6]=(rgb-.5)/.2820947918;parts.append(child)
        counts.append(dict(source=stations[source],affected=total));print(args.mode,counts[-1],flush=True)
    result=np.concatenate([a[keep],*parts]) if parts else a
    rm.f.write(OUT/'fused.ply',result);rm.f.to_splat(OUT/'fused.splat',result)
    shutil.copy2(baseline/'fusion.json',OUT/'fusion.json')
    write_json('footprint-report.json',dict(mode=args.mode,original_points=len(a),output_points=len(result),sources=counts,semantic_rules=False,note='Cross-edge image contrast selects splats. Recolor keeps geometry intact; split preserves Gaussian mixture moments but not exact alpha-compositing coverage. Both are experiments requiring multi-view validation.'))

if __name__=='__main__':run()
