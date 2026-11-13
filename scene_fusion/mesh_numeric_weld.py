"""Weld numerical-scale duplicate vertices before rejecting degenerate faces.

Poisson/export float32 duplicates can leave zero-area connectors. Removing the
connectors alone opens topological seams. This does not repair large real holes.
"""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np

def weld(p,tri,tolerance):
    import open3d as o3d
    if tolerance<=0:raise ValueError('Positive weld tolerance required')
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(p,dtype='f')),o3d.utility.Vector3iVector(tri))
    mesh.merge_close_vertices(tolerance);mesh.remove_degenerate_triangles();mesh.remove_duplicated_triangles();mesh.remove_unreferenced_vertices()
    q=np.asarray(mesh.vertices).astype('f');faces=np.asarray(mesh.triangles).astype('i')
    area=np.linalg.norm(np.cross(q[faces[:,1]]-q[faces[:,0]],q[faces[:,2]]-q[faces[:,0]]),axis=1)
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1);_,count=np.unique(edges,axis=0,return_counts=True)
    return q,faces,dict(tolerance=tolerance,input_faces=len(tri),faces=len(faces),input_vertices=len(p),vertices=len(q),
        tiny_faces=int((area<=1e-10).sum()),boundary_edges=int((count==1).sum()),nonmanifold_edges=int((count>2).sum()))

def run(a):
    z=np.load(a.mesh);p,tri=z['p'],z['tri'];spacing=float(np.median(np.linalg.norm(p[tri[:,1]]-p[tri[:,0]],axis=1)))
    q,faces,report=weld(p,tri,spacing*.001)
    if report['tiny_faces']:raise ValueError('Numerical weld left tiny triangles: '+str(report))
    a.out.mkdir(parents=True,exist_ok=True);path=a.out/'surface.npz'
    np.savez_compressed(path,p=q,tri=faces,geometry_required=np.ones(len(faces),bool))
    report['input_sha256']=hashlib.sha256(a.mesh.read_bytes()).hexdigest()
    (a.out/'numeric-weld.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    for name in ['poisson.json','geometry-guide.ply']:
        if (a.mesh.parent/name).exists():shutil.copyfile(a.mesh.parent/name,a.out/name)
    from mesh_surface_repair import visibility
    C=np.load(a.cameras/'aligned-cameras.npz')['C'];vis,cov=visibility(q,faces,C,spacing)
    np.savez_compressed(a.out/'visibility.npz',visible=vis,coverage=cov,geometry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),camera_sha256=hashlib.sha256((a.cameras/'aligned-cameras.npz').read_bytes()).hexdigest())
    print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mesh',type=Path,required=True);p.add_argument('--cameras',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
