"""Deliberately low-detail guide resampling, coarse Poisson and Taubin fairing.

Implements the user's low-pass/resample analogy, not a reversible coordinate
scale. Thin objects may disappear; this is an independent optional branch.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def adaptive_cloud(original,coarse,source,reference_mesh,voxel):
    import open3d as o3d
    from scipy.ndimage import gaussian_filter,sobel,maximum_filter,map_coordinates,zoom
    from scipy.spatial import cKDTree
    meta=json.loads((source/'scene.json').read_text('utf8'));cam=np.load(source/'aligned-cameras.npz');points=np.asarray(original.points)
    z=np.load(reference_mesh);rayscene=o3d.t.geometry.RaycastingScene()
    rayscene.add_triangles(o3d.core.Tensor(z['p'].astype('f')),o3d.core.Tensor(z['tri'].astype('uint32')))
    support=np.zeros(len(points),int)
    for i,station in enumerate(meta['stations']):
        rgb=np.asarray(o3d.io.read_image(station['photo']));gray=rgb[:,:,:3]@np.array([.299,.587,.114])/255
        gray=zoom(gray,(1024/gray.shape[0],2048/gray.shape[1]),order=1)
        gray=gaussian_filter(gray,.8);edge=np.hypot(sobel(gray,axis=0),sobel(gray,axis=1))/8
        edge=maximum_filter(edge,size=5)
        delta=points-cam['C'][i];dist=np.linalg.norm(delta,axis=1);local=delta@cam['Q'][i]
        u=(np.arctan2(-local[:,1],local[:,0])/(2*np.pi)+.5)*2048-.5
        v=np.arctan2(np.linalg.norm(local[:,:2],axis=1),local[:,2])/np.pi*1024-.5
        response=map_coordinates(edge,np.array([v,u]),order=1,mode='wrap')
        ids=np.flatnonzero(response>.045)
        rays=np.c_[np.broadcast_to(cam['C'][i],(len(ids),3)),delta[ids]/dist[ids,None]].astype('f')
        hit=rayscene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        support[ids]+=(np.isfinite(hit)&(abs(hit-dist[ids])<voxel))
    # High-resolution samples are retained only near edges supported by two photos.
    selected=support>=2
    if selected.sum():
        fine=original.select_by_index(np.flatnonzero(selected));cp=np.asarray(coarse.points)
        distance=cKDTree(points[selected]).query(cp,workers=4)[0]
        coarse=coarse.select_by_index(np.flatnonzero(distance>voxel*.8))
        coarse+=fine
    return coarse,dict(edge_supported_points=int(selected.sum()),minimum_supporting_photos=2,
        purpose='preserve sampling near image edges, not force every image edge into geometry')

def run(a):
    import open3d as o3d
    from mesh_surface_repair import visibility
    cloud=o3d.io.read_point_cloud(str(a.guide));original=len(cloud.points)
    if original<3 or not cloud.has_normals():raise ValueError('An oriented surface guide with at least three points is required')
    extent=float(np.ptp(np.asarray(cloud.points),axis=0).max());voxel=extent*a.relative_voxel
    cloud=cloud.voxel_down_sample(voxel);n=np.asarray(cloud.normals);good=np.linalg.norm(n,axis=1)>.5
    cloud=cloud.select_by_index(np.flatnonzero(good));cloud.normalize_normals()
    adaptive=None
    if a.edge_source:
        original_cloud=o3d.io.read_point_cloud(str(a.guide))
        cloud,adaptive=adaptive_cloud(original_cloud,cloud,a.edge_source,a.reference_mesh,voxel)
    a.out.mkdir(parents=True,exist_ok=True);o3d.io.write_point_cloud(str(a.out/'geometry-guide.ply'),cloud)
    mesh,density=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud,depth=a.depth,scale=1.05,linear_fit=True,n_threads=8)
    mesh.remove_degenerate_triangles();mesh.remove_duplicated_triangles();mesh.remove_unreferenced_vertices()
    before=np.asarray(mesh.vertices).copy()
    mesh=mesh.filter_smooth_taubin(number_of_iterations=a.iterations)
    shift=np.linalg.norm(np.asarray(mesh.vertices)-before,axis=1)
    comp,counts,areas=mesh.cluster_connected_triangles();remove=(np.asarray(counts)<=24)&(np.asarray(areas)<6*voxel**2)
    mesh.remove_triangles_by_mask(remove[np.asarray(comp)]);mesh.remove_unreferenced_vertices()
    p=np.asarray(mesh.vertices).astype('f');tri=np.asarray(mesh.triangles).astype('i');path=a.out/'surface.npz'
    np.savez_compressed(path,p=p,tri=tri,geometry_required=np.ones(len(tri),bool))
    report=dict(method='Voxel guide resampling, coarse Screened Poisson, Taubin surface fairing',
        original_samples=original,samples=len(cloud.points),relative_voxel=a.relative_voxel,voxel=voxel,depth=a.depth,
        taubin_iterations=a.iterations,max_fairing_displacement=float(shift.max()),faces=len(tri),
        input_sha256=hashlib.sha256(a.guide.read_bytes()).hexdigest(),inferred_geometry=True,
        intentional_detail_loss=True,warning='Not straight-line or primitive fitting; thin structure can shrink or disappear.')
    report['adaptive_edges']=adaptive
    (a.out/'poisson.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(report,flush=True)
    C=np.load(a.cameras/'aligned-cameras.npz')['C'];vis,cov=visibility(p,tri,C,voxel*.4)
    np.savez_compressed(a.out/'visibility.npz',visible=vis,coverage=cov,geometry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),camera_sha256=hashlib.sha256((a.cameras/'aligned-cameras.npz').read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--guide',type=Path,required=True);p.add_argument('--cameras',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--relative-voxel',type=float,default=.004);p.add_argument('--depth',type=int,default=8);p.add_argument('--iterations',type=int,default=12)
    p.add_argument('--edge-source',type=Path);p.add_argument('--reference-mesh',type=Path)
    a=p.parse_args()
    if not 0<a.relative_voxel<.1 or not 5<=a.depth<=12 or not 0<=a.iterations<=100:raise ValueError('Invalid low-pass settings')
    if bool(a.edge_source)!=bool(a.reference_mesh):raise ValueError('Adaptive edges require both source cameras and reference mesh')
    run(a)
