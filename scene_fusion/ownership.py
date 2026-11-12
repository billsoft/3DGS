"""N-source spatial ownership with supported fallback and coherent voxel labels.

This initial fusion preserves native Gaussian parameters rather than resampling
a room-specific mesh. A spatial cell has one source; smoothing never assigns a
source that has no actual Gaussian center in that cell. Footprints can cross a
cell boundary, so a cell partition is not a guarantee of zero optical overlap.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from core import load,gs,write_json


def choose_labels(scores,neighbors,iterations=4,penalty=.3):
    """Deterministic ICM with strict support, works for any N >= 2."""
    labels=scores.argmax(1);best=scores.max(1);supported=scores>0
    unary=-np.log(np.maximum(scores,1e-20)/np.maximum(best[:,None],1e-20))
    unary[~supported]=1e6
    history=[]
    for _ in range(iterations):
        costs=unary.copy()
        for source in range(scores.shape[1]):
            costs[:,source]+=penalty*np.mean(labels[neighbors]!=source,axis=1)
        new=costs.argmin(1);history.append(int((new!=labels).sum()))
        labels=new
        if history[-1]==0:break
    assert supported[np.arange(len(labels)),labels].all()
    return labels,history


def run(out,mode='voxel',warped=False):
    scene=load(out);cam=np.load(out/'aligned-cameras.npz');C,Q,M=cam['C'],cam['Q'],cam['matrices'];N=len(C)
    aligned=out/'aligned';aligned.mkdir(exist_ok=True);clouds=[];sources=[];quality=[]
    for i,item in enumerate(scene['stations']):
        if warped:a=np.load(out/'warped'/(item['id']+'.npy'))
        else:
            a=gs.transform(np.load(out/item['id']/'visible.npy'),M[i]);np.save(aligned/(item['id']+'.npy'),a)
        normal=Rotation.from_quat(a[:,[11,12,13,10]]).as_matrix()[np.arange(len(a)),:,a[:,7:10].argmin(1)]
        direction=C[i]-a[:,:3];distance=np.linalg.norm(direction,axis=1)
        angle=abs((normal*direction).sum(1))/np.maximum(distance,1e-6)
        score=(.25+.75*angle)/np.maximum(distance,.1)**1.6
        clouds.append(a);sources.append(np.full(len(a),i,dtype='i2'));quality.append(score.astype('f'))
    if mode=='visible-domains':
        views=[gs.View(out/'warped-depth'/item['id'],np.eye(4)) if warped else gs.View(out/item['id'],m) for item,m in zip(scene['stations'],M)]
        pieces=[];indices=[];reports=[]
        for i,a in enumerate(clouds):
            p=a[:,:3];scores=np.zeros((len(a),N),dtype='f');scores[:,i]=quality[i]
            normals=Rotation.from_quat(a[:,[11,12,13,10]]).as_matrix()[np.arange(len(a)),:,a[:,7:10].argmin(1)]
            conflict=np.zeros(len(a),'i');support=np.zeros(len(a),'i')
            for j,v in enumerate(views):
                if i==j:continue
                valid,delta,depth=v.observe(p);tol=.035+.025*depth
                compatible=valid&(abs(delta)<tol);conflict+=valid&(delta<-2*tol);support+=compatible
                direction=C[j]-p;distance=np.linalg.norm(direction,axis=1)
                incidence=abs((normals*direction).sum(1))/np.maximum(distance,1e-6)
                # A camera whose front layer hides this point cannot claim it.
                eligible=valid&(delta<tol)
                scores[:,j]=np.where(eligible,(.25+.75*incidence)/np.maximum(distance,.1)**1.6,0)
            winner=scores.argmax(1)
            keep=(winner==i)&~((conflict>=2)&(support==0))
            pieces.append(a[keep]);indices.append(np.full(keep.sum(),i,dtype='i2'))
            reports.append(dict(id=scene['stations'][i]['id'],input=len(a),retained=int(keep.sum()),
                multiple_free_space_conflicts=int(((conflict>=2)&(support==0)).sum())))
        fused=np.concatenate(pieces);np.save(out/'initial.npy',fused);np.save(out/'source.npy',np.concatenate(indices))
        gs.write(out/'initial.ply',fused);gs.to_splat(out/'initial.splat',fused)
        up=Q[:,:,2].mean(0);up/=np.linalg.norm(up)
        write_json(out/'fusion.json',dict(stations=[s['id'] for s in scene['stations']],centers=C.tolist(),up=up.tolist(),splats=len(fused),method='N-camera visibility domains; single winning source; free-space conflict guard'))
        write_json(out/'ownership-report.json',dict(mode=mode,stations=N,sources=reports,retained=len(fused)))
        print(json.dumps(reports),flush=True);return
    a=np.concatenate(clouds);source=np.concatenate(sources);quality=np.concatenate(quality)
    extent=np.median(np.linalg.norm(a[::10,:3]-C.mean(0),axis=1));cell=max(1e-4,extent*.006)
    origin=a[:,:3].min(0);keys=np.floor((a[:,:3]-origin)/cell).astype('i')
    unique,inverse=np.unique(keys,axis=0,return_inverse=True);K=len(unique)
    scores=np.zeros((K,N),dtype='f')
    np.maximum.at(scores,(inverse,source),quality)
    positions=unique.astype('f')*cell+origin+cell/2
    tree=cKDTree(positions);distance,neighbors=tree.query(positions,k=min(7,K),workers=4)
    neighbors=np.where(distance<cell*1.8,neighbors,np.arange(K)[:,None])
    labels,history=choose_labels(scores,neighbors)
    keep=source==labels[inverse]
    fused=a[keep];np.save(out/'initial.npy',fused);np.save(out/'source.npy',source[keep])
    np.savez_compressed(out/'ownership.npz',positions=positions,labels=labels,scores=scores,cell=cell)
    gs.write(out/'initial.ply',fused);gs.to_splat(out/'initial.splat',fused)
    up=Q[:,:,2].mean(0);up/=np.linalg.norm(up)
    write_json(out/'fusion.json',dict(stations=[s['id'] for s in scene['stations']],centers=C.tolist(),up=up.tolist(),
        splats=len(fused),method='N-source supported voxel ownership; native Gaussians; no room-specific mesh'))
    report=dict(stations=N,input_visible=len(a),retained=len(fused),cells=K,cell_size=cell,
        iterations_changed=history,retained_per_source=np.bincount(source[keep],minlength=N).tolist(),
        limitations=['Cell boundary footprints may overlap','Generated depth is not a metric scan','No unseen geometry recovery'])
    write_json(out/'ownership-report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['voxel','visible-domains'],default='visible-domains');p.add_argument('--warped',action='store_true');a=p.parse_args();run(a.out,a.mode,a.warped)
