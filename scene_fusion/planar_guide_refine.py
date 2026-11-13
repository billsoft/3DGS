"""Locally supported planar priors on oriented guide samples, then Poisson.

Planes must come from automatic multi-station depth verification. This is a
bounded geometric prior, not ground truth or a semantic floor/ceiling repair.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def regularize(points,normals,planes,spacing):
    p=np.asarray(points,dtype=float).copy();n=np.asarray(normals,dtype=float).copy()
    if spacing<=0 or p.ndim!=2 or p.shape[1]!=3 or n.shape!=p.shape or len(p)<3 or not np.isfinite(p).all() or not np.isfinite(n).all():raise ValueError('Invalid guide samples')
    length=np.linalg.norm(n,axis=1)
    if np.any(length<1e-9):raise ValueError('Zero guide normal')
    n/=length[:,None]
    tree=cKDTree(p);smooth=np.zeros(len(p),bool)
    for lo in range(0,len(p),16000):
        hi=min(lo+16000,len(p));dist,ids=tree.query(p[lo:hi],k=min(24,len(p)),workers=4)
        patch=p[ids];center=patch.mean(1);delta=patch-center[:,None]
        cov=np.einsum('nki,nkj->nij',delta,delta)/patch.shape[1]
        val=np.linalg.eigvalsh(cov)
        smooth[lo:hi]=(val[:,0]/np.maximum(val.sum(1),1e-12)<.025)&(dist[:,-1]<spacing*5)
    score=np.full(len(p),np.inf);owner=np.full(len(p),-1,int);offsets=np.zeros(len(p));targets=np.zeros_like(p)
    for i,plane in enumerate(planes):
        if len(set(plane.get('supporting_stations',[])))<3:continue
        normal=np.array(plane['normal'],dtype=float);norm=np.linalg.norm(normal)
        if not np.isfinite(norm) or norm<1e-9:raise ValueError('Invalid plane normal')
        normal/=norm
        distance=p@normal+plane['offset']/norm;cos=n@normal
        cost=abs(distance)+spacing*(1-abs(cos))
        keep=smooth&(abs(distance)<spacing*2)&(abs(cos)>.9)&(cost<score)
        score[keep]=cost[keep];owner[keep]=i;offsets[keep]=distance[keep]
        targets[keep]=normal
    valid=owner>=0
    # Taper the outer band, leaving curved edges and distinct surfaces untouched.
    weight=np.clip((spacing*2-abs(offsets))/(spacing),0,1)*.9;weight[~valid]=0
    shift=-weight[:,None]*offsets[:,None]*targets
    p+=shift
    target=targets*np.sign(np.sum(targets*n,axis=1))[:,None]
    n=n*(1-weight[:,None])+target*weight[:,None];n/=np.maximum(np.linalg.norm(n,axis=1)[:,None],1e-12)
    return p,n,dict(selected_points=int(valid.sum()),locally_planar_points=int(smooth.sum()),total_points=len(p),
        max_displacement=float(np.linalg.norm(shift,axis=1).max()),distance_gate=2*spacing,
        plane_counts={str(i):int((owner==i).sum()) for i in np.unique(owner[valid])},
        inferred_geometry=True,semantic_rules=False)

def run(a):
    import open3d as o3d
    from mesh_surface_repair import visibility
    cloud=o3d.io.read_point_cloud(str(a.guide));meta=json.loads(a.planes.read_text('utf8'));spacing=meta['spacing']
    p,n,report=regularize(np.asarray(cloud.points),np.asarray(cloud.normals),meta['planes'],spacing)
    a.out.mkdir(parents=True,exist_ok=True);cloud.points=o3d.utility.Vector3dVector(p);cloud.normals=o3d.utility.Vector3dVector(n)
    o3d.io.write_point_cloud(str(a.out/'geometry-guide.ply'),cloud)
    report.update(guide_sha256=hashlib.sha256(a.guide.read_bytes()).hexdigest(),planes_sha256=hashlib.sha256(a.planes.read_bytes()).hexdigest())
    (a.out/'planar-guide.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(report,flush=True)
    surface,density=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud,depth=a.depth,scale=1.05,linear_fit=True,n_threads=8)
    surface.remove_degenerate_triangles();surface.remove_duplicated_triangles();surface.remove_unreferenced_vertices()
    raw=len(surface.triangles)
    if raw>a.max_faces:surface=surface.simplify_quadric_decimation(a.max_faces)
    comp,counts,areas=surface.cluster_connected_triangles();remove=(np.asarray(counts)<=24)&(np.asarray(areas)<6*spacing**2)
    surface.remove_triangles_by_mask(remove[np.asarray(comp)]);surface.remove_unreferenced_vertices()
    q=np.asarray(surface.vertices).astype('f');tri=np.asarray(surface.triangles).astype('i');path=a.out/'surface.npz'
    np.savez_compressed(path,p=q,tri=tri,geometry_required=np.ones(len(tri),bool))
    poisson=dict(method='Poisson from bounded locally planar guide',depth=a.depth,samples=len(p),raw_faces=raw,faces=len(tri),planar_guide=report,inferred_geometry=True)
    (a.out/'poisson.json').write_text(json.dumps(poisson,indent=2),encoding='utf8');print(poisson,flush=True)
    C=np.load(a.cameras/'aligned-cameras.npz')['C'];vis,coverage=visibility(q,tri,C,spacing)
    np.savez_compressed(a.out/'visibility.npz',visible=vis,coverage=coverage,geometry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),camera_sha256=hashlib.sha256((a.cameras/'aligned-cameras.npz').read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--guide',type=Path,required=True);p.add_argument('--planes',type=Path,required=True);p.add_argument('--cameras',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--depth',type=int,default=10);p.add_argument('--max-faces',type=int,default=2000000);run(p.parse_args())
