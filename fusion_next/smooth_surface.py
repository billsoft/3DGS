"""Conservative normal denoising, protected by image/mesh edges and held matches."""
from common import *
from edge_detail import gradient_map,sample_map
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

def run():
    geo=np.load(OUT/'geometry.npz');data={k:geo[k] for k in geo.files};p,tri=data['p'],data['tri'];original=p.copy()
    fn=np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]])
    norm=np.zeros_like(p)
    for k in range(3):np.add.at(norm,tri[:,k],fn)
    norm/=np.maximum(np.linalg.norm(norm,axis=1)[:,None],1e-9)
    edge,count=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),1),axis=0,return_counts=True)
    a,b=edge.T;dot=(norm[a]*norm[b]).sum(1)
    locked=np.zeros(len(p),bool);locked[np.unique(edge[(count!=2)|(dot<.85)])]=True
    peak=np.zeros(len(p));support=np.zeros(len(p),int)
    for i,im in enumerate(photos()):
        err,conf,dep=cams[i].sample(cams[i].project(p));valid=(conf>.25)&(abs(err)<.06+.015*dep)
        g=sample_map(gradient_map(im),uv(project(p,i)));magnitude=np.linalg.norm(g,axis=1)
        peak=np.maximum(peak,np.where(valid,magnitude,0));support+=valid
    locked|=(peak>.025)|(support<2)
    w=np.clip((dot-.85)/.15,0,1)**2
    W=coo_matrix((np.r_[w,w],(np.r_[a,b],np.r_[b,a])),shape=(len(p),len(p))).tocsr()
    degree=np.maximum(np.asarray(W.sum(1)).ravel(),1e-9)
    for _ in range(3):
        average=(W@p)/degree[:,None];step=((average-p)*norm).sum(1)*.18
        step[locked]=0;p=p+step[:,None]*norm
        displacement=np.clip(((p-original)*norm).sum(1),-.008,.008)
        p=original+displacement[:,None]*norm
    frozen=np.zeros(len(p),bool)
    for _ in range(24):
        nf=np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]])
        bad=((fn*nf).sum(1)<=0)|(np.linalg.norm(nf,axis=1)<.5*np.linalg.norm(fn,axis=1))
        if not bad.any():break
        frozen[np.unique(tri[bad])]=True;p[frozen]=original[frozen]
    corr=np.load(OUT/'correspondences.npz');d,idx=cKDTree(original).query(corr['initial'])
    mask=corr['held']&(d<.10);idx=idx[mask];target=corr['target'][mask]
    before=abs(((target-original[idx])*norm[idx]).sum(1));after=abs(((target-p[idx])*norm[idx]).sum(1))
    nf=np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]])
    bad=((fn*nf).sum(1)<=0)|(np.linalg.norm(nf,axis=1)<.5*np.linalg.norm(fn,axis=1))
    accepted=bool(len(before)>50 and not bad.any() and np.median(after)<=np.median(before)*1.01 and np.quantile(after,.9)<=np.quantile(before,.9)*1.01)
    if accepted:data['p']=p
    np.savez_compressed(OUT/'geometry.npz',**data)
    report=dict(accepted=accepted,locked_vertices=int(locked.sum()),frozen_unstable_vertices=int(frozen.sum()),moved_vertices=int((np.linalg.norm(p-original,axis=1)>1e-5).sum()),max_candidate_displacement=float(np.linalg.norm(p-original,axis=1).max()),held_median_before=float(np.median(before)),held_median_after=float(np.median(after)),held_p90_before=float(np.quantile(before,.9)),held_p90_after=float(np.quantile(after,.9)),invalid_faces=int(bad.sum()),note='Normal denoising only. Does not fix topology, invent surfaces, or repair unobserved backsides.')
    write_json('smooth-report.json',report);print(report,flush=True)

if __name__=='__main__':run()
