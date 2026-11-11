"""Ray-preserving common-surface anchoring of cropped original Gaussians."""
from common import *
from collider_audit import scene_for
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from territory import ray_plane_jacobian

def run():
    geo=np.load(OUT/'geometry.npz');p,tri=geo['p'],geo['tri'];labels=np.load(OUT/'ownership.npz');tree=cKDTree(labels['center'])
    prior=np.load(OUT/'prior.npz');meta=json.loads((BASE/'mesh.json').read_text())
    valid_texture=np.array([cv.imread(str(BASE/g['texture'])).std()>1 for g in meta['groups']])
    scene,scene_mesh=scene_for(p,tri[valid_texture[prior['group']]])
    normal=np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]])
    normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-9)
    parts=[];photo_parts=[];report=[];images=photos();painted_overlap=0
    offsets=np.load(OUT/'seam-offset.npy') if (OUT/'seam-offset.npy').exists() else np.zeros((len(tri),3))
    for source,station in enumerate(stations):
        a=rm.f.prep.read_gs(OUT/f'cropped-{station}.ply');old=a[:,:3].copy()
        ray=old-C[source];radial=np.linalg.norm(ray,axis=1);ray/=np.maximum(radial[:,None],1e-9)
        hit=scene.run(np.ascontiguousarray(np.tile(C[source],(len(a),1)),'f'),np.ascontiguousarray(ray,'f'),query='DISTANCE')
        plausible=np.isfinite(hit)&(hit>0)&(abs(hit-radial)<.20)
        desired=old.copy();desired[plausible]=C[source]+ray[plausible]*hit[plausible,None]
        _,face=tree.query(desired);n=normal[face];denominator=(n*ray).sum(1)
        valid=plausible&(abs(denominator)>.20)
        idx=np.flatnonzero(valid);ratio=hit[idx]/radial[idx]
        J=ray_plane_jacobian(ray[idx],n[idx],ratio)
        rot=Rotation.from_quat(a[idx][:,[11,12,13,10]]).as_matrix()
        cov=(rot*np.exp(2*a[idx,None,7:10]))@rot.transpose(0,2,1)
        mapped=J@cov@J.transpose(0,2,1)+.0015**2*n[idx,:,None]*n[idx,None,:]
        eig,axes=np.linalg.eigh(mapped);eig=np.maximum(eig,1e-10);axes[:,:,0]*=np.linalg.det(axes)[:,None]
        a[idx,:3]=desired[idx];a[idx,7:10]=.5*np.log(eig);a[idx,10:14]=Rotation.from_matrix(axes).as_quat()[:,[3,0,1,2]]
        _,final_face=tree.query(a[:,:3]);keep=labels['owner'][final_face]==source
        distance=labels['distance'][final_face];near=distance<.12
        a[near,7:10]=np.minimum(a[near,7:10],np.log(np.maximum((distance[near,None]+.055)/3,.003)))
        if not np.isfinite(a).all():raise ValueError('Invalid anchored Gaussian')
        parts.append(a[keep]);rm.f.write(OUT/f'anchored-{station}.ply',a[keep])
        secondary=(labels['partner'][final_face]==source)&(distance<.055)&(labels['owner'][final_face]<4)&valid
        paint_keep=keep|secondary;painted=a[paint_keep].copy();pf=final_face[paint_keep]
        color=correct_rgb(sample(images[source],uv(project(painted[:,:3],source)))[:,::-1]/255,source)
        own=labels['owner'][pf]==source
        counterpart=np.where(own,labels['partner'][pf],labels['owner'][pf])
        agreement=np.zeros(len(painted),bool)
        for other in range(4):
            selected=np.flatnonzero((counterpart==other)&(labels['distance'][pf]<.055))
            if not len(selected):continue
            alternate=correct_rgb(sample(images[other],uv(project(painted[selected,:3],other)))[:,::-1]/255,other)
            err,conf,dep=cams[other].sample(cams[other].project(painted[selected,:3]))
            agreement[selected]=(np.linalg.norm(color[selected]-alternate,axis=1)<.08)&(conf>.25)&(abs(err)<.06+.015*dep)
        t=np.clip(labels['distance'][pf]/.055,0,1)
        weight=np.where(own,np.where(agreement,.5+.5*t,1),np.where(agreement,.5-.5*t,0))
        valid_paint=weight>1e-6;painted_overlap+=int((valid_paint&~own).sum())
        # Blend optical depth rather than linearly halving an opaque Gaussian.
        alpha=1/(1+np.exp(-np.clip(painted[:,6],-30,30)))
        alpha=1-np.power(np.maximum(1-alpha,1e-8),weight)
        alpha=np.clip(alpha,1e-6,1-1e-6);painted[:,6]=np.log(alpha/(1-alpha))
        color=np.clip(color+offsets[pf],0,1);painted[:,3:6]=(color-.5)/.2820947918
        photo_parts.append(painted[valid_paint])
        report.append(dict(station=station,retained=int(keep.sum()),anchored=int((valid&keep).sum()),unchanged=int((~valid&keep).sum()),max_shift=float(np.linalg.norm(a[:,:3]-old,axis=1).max())))
        print('anchored',report[-1],flush=True)
    merged=np.concatenate(parts);rm.f.write(OUT/'anchored-core-fused.ply',merged);rm.f.to_splat(OUT/'anchored-core-fused.splat',merged)
    painted=np.concatenate(photo_parts);rm.f.write(OUT/'anchored-photo-fused.ply',painted);rm.f.to_splat(OUT/'anchored-photo-fused.splat',painted)
    write_json('anchor-report.json',dict(sources=report,points=len(merged),painted_points=len(painted),painted_secondary_points=painted_overlap,note='The core candidate preserves source colors/opacity. The painted candidate reprojects original photos and blends optical depth only in a depth/color-consistent narrow band. Ray-plane projection uses covariance Jacobians and accepted common geometry; wrong geometry remains wrong.'))

if __name__=='__main__':run()
