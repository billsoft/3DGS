"""Topology-local bilateral face-normal filtering and bounded vertex fitting.

Independent implementation of the two-stage normal-filter/vertex-update idea in
Zheng et al., Bilateral Normal Filtering for Mesh Denoising (TVCG 2011).
Not a port of GuidedDenoising. No semantic masks, camera counts or object fits.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix


def denoise(p,tri,iterations=5,fit_iterations=12,cap_ratio=.8,normal_bandwidth=.09,min_cosine=.5):
    p=np.asarray(p,dtype=np.float64);tri=np.asarray(tri,dtype=np.int32)
    if iterations<0 or fit_iterations<0 or cap_ratio<0 or not np.isfinite(p).all():raise ValueError('Invalid denoise input')
    if not np.isfinite(normal_bandwidth) or normal_bandwidth<=0 or not -1<=min_cosine<=1:
        raise ValueError('Invalid normal filtering bandwidth or cutoff')
    original=p.copy();xyz=p[tri];center=xyz.mean(1)
    cross=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);area=np.linalg.norm(cross,axis=1)
    if np.any(area<1e-12):raise ValueError('Degenerate input triangles')
    normals=cross/area[:,None];initial=normals.copy()
    edge,counts=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0,return_counts=True)
    spacing=float(np.median(np.linalg.norm(p[edge[:,0]]-p[edge[:,1]],axis=1)))
    fixed=np.zeros(len(p),bool);fixed[np.unique(edge[counts!=2])]=True
    # Vertex-incident neighbors follow topology, never Euclidean proximity across sheets.
    incidence=coo_matrix((np.ones(tri.size),(tri.ravel(),np.repeat(np.arange(len(tri)),3))),shape=(len(p),len(tri))).tocsr()
    graph=(incidence.T@incidence).tocoo();a,b=graph.row,graph.col
    spatial=np.exp(-np.sum((center[a]-center[b])**2,axis=1)/(2*(2*spacing)**2))*area[b]
    for _ in range(iterations):
        cosine=np.sum(normals[a]*normals[b],axis=1)
        weight=spatial*np.exp(-(1-np.clip(cosine,-1,1))/normal_bandwidth)
        weight[cosine<min_cosine]=0
        W=coo_matrix((weight,(a,b)),shape=(len(tri),len(tri))).tocsr()
        out=W@normals;normals=out/np.maximum(np.linalg.norm(out,axis=1)[:,None],1e-12)
    # Fit to filtered tangent planes without tangential Laplacian shrinkage.
    vertex_weight=np.asarray(incidence@area).ravel()
    for _ in range(fit_iterations):
        centers=p[tri].mean(1);delta=np.zeros_like(p)
        for k in range(3):
            step=np.sum((centers-p[tri[:,k]])*normals,axis=1)[:,None]*normals*area[:,None]
            np.add.at(delta,tri[:,k],step)
        proposal=p+.6*delta/np.maximum(vertex_weight[:,None],1e-12)
        shift=proposal-original;length=np.linalg.norm(shift,axis=1)
        proposal=original+shift*np.minimum(1,cap_ratio*spacing/np.maximum(length,1e-12))[:,None]
        proposal[fixed]=original[fixed];p=proposal
    # Local rollback propagates until no triangle flips or nearly collapses.
    rolled=np.zeros(len(p),bool)
    for _ in range(64):
        new=np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]])
        bad=(new*initial).sum(1)<area*.15
        if not bad.any():break
        ids=np.unique(tri[bad]);rolled[ids]=True;p[ids]=original[ids]
    if bad.any():p=original.copy();rolled[:]=True
    return p.astype('f'),dict(method='topology-local bilateral face normals and bounded tangent-plane fitting',
        iterations=iterations,fit_iterations=fit_iterations,spacing=spacing,displacement_cap=cap_ratio*spacing,
        normal_bandwidth=normal_bandwidth,min_cosine=min_cosine,
        max_displacement=float(np.linalg.norm(p-original,axis=1).max()),locked_vertices=int(fixed.sum()),
        rollback_vertices=int(rolled.sum()),topology_unchanged=True)


def run(a):
    z=np.load(a.mesh);p,report=denoise(z['p'],z['tri'],a.iterations,a.fit_iterations,a.cap_ratio,a.normal_bandwidth,a.min_cosine)
    a.out.mkdir(parents=True,exist_ok=True);surface=a.out/'surface.npz'
    np.savez_compressed(surface,p=p,tri=z['tri'],geometry_required=np.ones(len(z['tri']),bool))
    report['input_sha256']=hashlib.sha256(a.mesh.read_bytes()).hexdigest()
    (a.out/'mesh-denoise.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(report,flush=True)
    import open3d as o3d
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(p),o3d.utility.Vector3iVector(z['tri']));mesh.compute_vertex_normals()
    cloud=o3d.geometry.PointCloud(mesh.vertices);cloud.normals=mesh.vertex_normals;o3d.io.write_point_cloud(str(a.out/'geometry-guide.ply'),cloud)
    from mesh_surface_repair import visibility
    centers=np.load(a.cameras/'aligned-cameras.npz')['C'];vis,coverage=visibility(p,z['tri'],centers,report['spacing'])
    np.savez_compressed(a.out/'visibility.npz',visible=vis,coverage=coverage,
        geometry_sha256=hashlib.sha256(surface.read_bytes()).hexdigest(),camera_sha256=hashlib.sha256((a.cameras/'aligned-cameras.npz').read_bytes()).hexdigest())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mesh',type=Path,required=True);ap.add_argument('--cameras',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--iterations',type=int,default=5);ap.add_argument('--fit-iterations',type=int,default=12);ap.add_argument('--cap-ratio',type=float,default=.8)
    ap.add_argument('--normal-bandwidth',type=float,default=.09,help='Larger values sacrifice sharp features; experimental, defaults unchanged')
    ap.add_argument('--min-cosine',type=float,default=.5,help='Exclude opposing face normals from the topology-local filter')
    run(ap.parse_args())
