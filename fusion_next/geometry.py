"""Normal-only robust deformation on the surface graph; held blocks select acceptance."""
from common import *
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import spsolve
def run():
    p,tri,meta=mesh();xyz=p[tri];fn=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);norm=np.zeros_like(p)
    for k in range(3):np.add.at(norm,tri[:,k],fn)
    norm/=np.maximum(1e-10,np.linalg.norm(norm,axis=1))[:,None]
    corr=np.load(OUT/'correspondences.npz');distance,idx=cKDTree(p).query(corr['initial']);target=((corr['target']-p[idx])*norm[idx]).sum(1)
    valid=(distance<.10)&(abs(target)<.15);train=valid&~corr['held'];held=valid&corr['held']
    e=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),1),axis=0);a,b=e.T
    weight=np.clip(((norm[a]*norm[b]).sum(1)-.75)/.25,0,1)**2
    W=coo_matrix((np.r_[weight,weight],(np.r_[a,b],np.r_[b,a])),shape=(len(p),len(p))).tocsr();L=diags(np.asarray(W.sum(1)).ravel())-W
    displacement=np.zeros(len(p));ids=idx[train];wanted=target[train];quality=corr['quality'][train]
    for _ in range(4):
        w=quality/np.maximum(1,abs(displacement[ids]-wanted)/.025)
        counts=np.bincount(ids,weights=w,minlength=len(p));rhs=np.bincount(ids,weights=w*wanted,minlength=len(p))
        displacement=np.clip(spsolve(diags(counts+.12)+L*.9,rhs),-.065,.065)
    frozen=np.zeros(len(p),bool)
    for _ in range(16):
        candidate=p+norm*displacement[:,None];newfn=np.cross(candidate[tri[:,1]]-candidate[tri[:,0]],candidate[tri[:,2]]-candidate[tri[:,0]])
        bad=((fn*newfn).sum(1)<.2*np.linalg.norm(fn,axis=1)*np.linalg.norm(newfn,axis=1))|(np.linalg.norm(newfn,axis=1)<.2*np.linalg.norm(fn,axis=1))
        if not bad.any():break
        frozen[np.unique(tri[bad])]=True;displacement[frozen]=0
    candidate=p+norm*displacement[:,None];newfn=np.cross(candidate[tri[:,1]]-candidate[tri[:,0]],candidate[tri[:,2]]-candidate[tri[:,0]])
    before=np.abs(target[held]);after=np.abs(target[held]-displacement[idx[held]])
    accepted=len(before)>50 and np.median(after)<np.median(before)*.97 and np.quantile(after,.9)<=np.quantile(before,.9)*1.02
    flips=((fn*newfn).sum(1)<0);accepted=bool(accepted and flips.sum()==0)
    report=dict(train=int(train.sum()),held=int(held.sum()),held_median_before=float(np.median(before)),held_median_after=float(np.median(after)),held_p90_before=float(np.quantile(before,.9)),held_p90_after=float(np.quantile(after,.9)),frozen_unstable_vertices=int(frozen.sum()),flipped_faces=int(flips.sum()),accepted=accepted,max_candidate_shift=float(abs(displacement).max()),note='Held correspondences are estimated from photos, not ground-truth depth. Rejected geometry retains the baseline.')
    np.savez_compressed(OUT/'geometry.npz',p=candidate if accepted else p,tri=tri,norm=norm,original=p,displacement=displacement if accepted else np.zeros_like(displacement))
    write_json('geometry-report.json',report);print(report,flush=True)
if __name__=='__main__':run()
