"""Crop original aligned Gaussian PLYs by source visibility and surface ownership."""
from common import *
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

def run():
    labels=np.load(OUT/'ownership.npz');tree=cKDTree(labels['center'])
    owner,partner,distance=labels['owner'],labels['partner'],labels['distance']
    primary=[];reports=[]
    for source,station in enumerate(stations):
        original=rm.f.prep.read_gs(rm.inputs/f'{station}.ply')
        a=rm.f.transform(original,np.linalg.inv(cams[source].inv))
        nearest,face=tree.query(a[:,:3]);core=owner[face]==source
        border=(partner[face]==source)&(distance[face]<.055)&(owner[face]<4)
        cam=rm.Camera(station,np.linalg.inv(cams[source].inv))
        cam.maps=[]
        for k in range(6):
            depth=np.load(rm.room/station/f'depth-{k}.npz')
            cam.maps.append((depth['depth'],depth['alpha']))
        error,confidence,radial=cam.sample(cam.project(a[:,:3]))
        ray=a[:,:3]-C[source];ray/=np.maximum(np.linalg.norm(ray,axis=1)[:,None],1e-9)
        rotations=Rotation.from_quat(a[:,[11,12,13,10]]).as_matrix()
        ray_local=np.einsum('ni,nij->nj',ray,rotations)
        radial_sigma=np.sqrt(np.sum(ray_local**2*np.exp(2*a[:,7:10]),axis=1))
        tolerance=.025+.010*radial+np.minimum(2*radial_sigma,.10)
        visible=(confidence>.90)&(abs(error)<tolerance)
        support=nearest<.14+np.minimum(np.exp(a[:,7:10]).max(1)*2,.10)
        keep=visible&support&(core|border)
        # The boundary is a surface approximation, not exact ellipsoid clipping.
        # Bound the 3-sigma footprint so very broad splats cannot invade far away.
        sigma_limit=(np.minimum(distance[face],1)+.055)/3
        sigma_limit=np.maximum(sigma_limit,.003)
        near_seam=distance[face]<.12
        clipped=int((keep&near_seam&(np.exp(a[:,7:10]).max(1)>sigma_limit)).sum())
        a[near_seam,7:10]=np.minimum(a[near_seam,7:10],np.log(sigma_limit[near_seam,None]))
        rm.f.write(OUT/f'cropped-{station}.ply',a[keep])
        rm.f.to_splat(OUT/f'cropped-{station}.splat',a[keep])
        primary.append(a[keep&core])
        reports.append(dict(station=station,input_points=len(a),kept_points=int(keep.sum()),core_points=int((keep&core).sum()),narrow_overlap_points=int((keep&border).sum()),footprints_limited=clipped,removed_fraction=float(1-keep.mean())))
        print('cropped',reports[-1],flush=True)
    merged=np.concatenate(primary)
    rm.f.write(OUT/'raw-core-fused.ply',merged);rm.f.to_splat(OUT/'raw-core-fused.splat',merged)
    write_json('crop-report.json',dict(sources=reports,raw_core_fused_points=len(merged),notes=['Each source file retains its own territory plus a narrow neighboring seam band','The raw-core fusion includes primary territory points only, avoiding double-layer overlap','Original Gaussian colors/opacity are preserved; only pose and seam footprint bounds change','Common-surface texture export is a separate comparison, not claimed to be raw PLY concatenation']))

if __name__=='__main__':run()
