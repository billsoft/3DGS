"""Screened Poisson surface reconstruction from an oriented, cleaned mesh.

This changes topology and infers missing surfaces. Keep the candidate separate:
closure is not evidence that a hidden object's shape was actually observed.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import open3d as o3d
from mesh_surface_repair import visibility


def run(a):
    o3d.utility.random.seed(721);z=np.load(a.mesh);p=z['p'];tri=z['tri']
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(p),o3d.utility.Vector3iVector(tri));mesh.compute_vertex_normals()
    cloud=mesh.sample_points_uniformly(a.samples,use_triangle_normal=False)
    normals=np.asarray(cloud.normals);valid=np.isfinite(normals).all(1)&(np.linalg.norm(normals,axis=1)>.5)
    cloud=cloud.select_by_index(np.flatnonzero(valid));cloud.normalize_normals()
    print('Poisson oriented samples',len(cloud.points),'depth',a.depth,flush=True)
    surface,density=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud,depth=a.depth,scale=1.05,linear_fit=True,n_threads=8)
    surface.remove_degenerate_triangles();surface.remove_duplicated_triangles();surface.remove_unreferenced_vertices()
    # Do not percentile-trim low-density vertices: it reopens the very small gaps
    # that the surface solver closed. Record their density instead for inspection.
    raw=len(surface.triangles)
    if raw>a.max_faces:surface=surface.simplify_quadric_decimation(a.max_faces)
    comp,counts,areas=surface.cluster_connected_triangles();comp=np.asarray(comp);counts=np.asarray(counts);areas=np.asarray(areas)
    spacing=float(np.median(np.linalg.norm(p[tri[:,1]]-p[tri[:,0]],axis=1)))
    remove=(counts<=24)&(areas<6*spacing**2);surface.remove_triangles_by_mask(remove[comp]);surface.remove_unreferenced_vertices()
    q=np.asarray(surface.vertices).astype('f');faces=np.asarray(surface.triangles).astype('i')
    if not len(faces) or not np.isfinite(q).all():raise ValueError('Invalid Poisson surface')
    a.out.mkdir(parents=True,exist_ok=True);path=a.out/'surface.npz'
    np.savez_compressed(path,p=q,tri=faces,geometry_required=np.ones(len(faces),bool))
    report=dict(method='Open3D Screened Poisson from oriented cleaned TSDF surface',depth=a.depth,samples=len(cloud.points),raw_faces=raw,faces=len(faces),
        input_sha256=hashlib.sha256(a.mesh.read_bytes()).hexdigest(),inferred_geometry=True,density_quantiles=np.quantile(np.asarray(density),[0,.01,.5,.99,1]).tolist(),removed_components=int(remove.sum()),spacing=spacing)
    (a.out/'poisson.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(report,flush=True)
    C=np.load(a.cameras/'aligned-cameras.npz')['C'];visible,coverage=visibility(q,faces,C,spacing)
    np.savez_compressed(a.out/'visibility.npz',visible=visible,coverage=coverage,geometry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),camera_sha256=hashlib.sha256((a.cameras/'aligned-cameras.npz').read_bytes()).hexdigest())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mesh',type=Path,required=True);p.add_argument('--cameras',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=int,default=700000);p.add_argument('--depth',type=int,default=10);p.add_argument('--max-faces',type=int,default=1600000);run(p.parse_args())
