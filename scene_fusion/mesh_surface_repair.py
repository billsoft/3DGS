"""Scale-relative component cleanup, bounded Open3D hole fill, exact visibility.

No semantic objects or fixed camera count. Hole caps are inferred geometry;
small isolated genuine details may be lost by the explicit cleanup threshold.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import open3d as o3d

def repair_arrays(p,tri,spacing,fragment_faces=24,hole_radius=3.):
    if spacing<=0 or fragment_faces<0 or hole_radius<=0:raise ValueError('Invalid scale or repair thresholds')
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(p),o3d.utility.Vector3iVector(tri))
    labels,counts,areas=mesh.cluster_connected_triangles()
    print('Component analysis finished',len(counts),flush=True)
    labels=np.asarray(labels);counts=np.asarray(counts);areas=np.asarray(areas)
    lo=np.full((len(counts),3),np.inf);hi=-lo.copy()
    for k in range(3):
        np.minimum.at(lo,labels,p[tri[:,k]]);np.maximum.at(hi,labels,p[tri[:,k]])
    diameter=np.linalg.norm(hi-lo,axis=1)
    remove=(counts<=fragment_faces)&(areas<=6*spacing**2)&(diameter<=5*spacing)
    keep=~remove[labels];kept=tri[keep]
    print('Removing fragments',int((~keep).sum()),'faces; filling bounded holes',flush=True)
    cleaned=o3d.t.geometry.TriangleMesh(o3d.core.Tensor(p.astype('f')),o3d.core.Tensor(kept.astype('int64')))
    filled=cleaned.fill_holes(hole_size=hole_radius*spacing)
    q=filled.vertex.positions.numpy().copy();faces=filled.triangle.indices.numpy().astype('i')
    # VTK preserves original points/faces and appends caps. Verify, never infer.
    if not np.array_equal(q,p.astype('f')) or not np.array_equal(faces[:len(kept)],kept):
        raise ValueError('Hole filling changed existing geometry or reordered faces')
    required=np.arange(len(faces))>=len(kept)
    # Suppress accidental duplicate caps on exterior boundaries of tiny sheets.
    keys=np.sort(faces,axis=1);_,first=np.unique(keys,axis=0,return_index=True)
    selected=np.sort(first);faces=faces[selected];required=required[selected]
    # Larger holes may be closed only where their whole boundary is planar.
    # Small curved caps remain bounded by the original three-spacing radius.
    cap_ids=np.flatnonzero(required);rejected_curved=0
    if len(cap_ids):
        caps=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(q),o3d.utility.Vector3iVector(faces[cap_ids]))
        cap_labels,cap_counts,_=caps.cluster_connected_triangles();cap_labels=np.asarray(cap_labels)
        accept=np.ones(len(faces),bool)
        for c in range(len(cap_counts)):
            ids=cap_ids[cap_labels==c];points=q[np.unique(faces[ids])]
            if np.linalg.norm(np.ptp(points,axis=0))<=6*spacing:continue
            centered=points-points.mean(0);_,_,axes=np.linalg.svd(centered,full_matrices=False)
            if np.max(abs(centered@axes[-1]))>.35*spacing:
                accept[ids]=False;rejected_curved+=len(ids)
        faces=faces[accept];required=required[accept]
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    _,inverse,edge_counts=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
    overfull=(edge_counts[inverse].reshape(3,-1).T>2).any(1)
    rejected=required&overfull;rejected_count=int(rejected.sum())
    faces=faces[~rejected];required=required[~rejected]
    return q,faces,required,dict(components=len(counts),removed_components=int(remove.sum()),removed_faces=int((~keep).sum()),
        added_faces=int(required.sum()),rejected_nonmanifold_caps=rejected_count,rejected_large_curved_caps=rejected_curved,spacing=spacing,max_fragment_diameter=5*spacing,max_hole_radius=hole_radius*spacing,
        original_vertices_unchanged=True,method='Open3D connected components and bounded VTK boundary-loop triangulation')

def visibility(p,tri,centers,spacing):
    scene=o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(p.astype('f')),o3d.core.Tensor(tri.astype('uint32')))
    center=p[tri].mean(1);vis=np.zeros((len(tri),len(centers)),bool);coverage=np.zeros(vis.shape,'f')
    for i,c in enumerate(centers):
        for start in range(0,len(tri),50000):
            end=min(start+50000,len(tri));t=tri[start:end];m=center[start:end]
            probes=np.concatenate([m[:,None],p[t]*.8+m[:,None]*.2],axis=1)
            delta=probes-c;distance=np.linalg.norm(delta,axis=2)
            directions=delta/np.maximum(distance[:,:,None],1e-9)
            rays=np.concatenate([np.broadcast_to(c,delta.shape),directions],axis=2).astype('f')
            result=scene.cast_rays(o3d.core.Tensor(rays));hit=result['t_hit'].numpy()
            seen=np.isfinite(hit)&(abs(hit-distance)<=max(spacing*.25,1e-5))
            vis[start:end,i]=seen.any(1);coverage[start:end,i]=seen.mean(1)
        print('Ray visibility',i+1,int(vis[:,i].sum()),flush=True)
    return vis,coverage

def run(a):
    z=np.load(a.mesh);p=z['p'].astype('f');tri=z['tri'].astype('i')
    spacing=float(np.median(np.linalg.norm(p[tri[:,1]]-p[tri[:,0]],axis=1)))
    p,tri,required,report=repair_arrays(p,tri,spacing,a.fragment_faces,a.hole_radius)
    a.out.mkdir(parents=True,exist_ok=True);surface=a.out/'surface.npz'
    np.savez_compressed(surface,p=p,tri=tri,geometry_required=required)
    report.update(input=str(a.mesh),input_sha256=hashlib.sha256(a.mesh.read_bytes()).hexdigest(),faces=len(tri))
    (a.out/'repair.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(report,flush=True)
    centers=np.load(a.cameras/'aligned-cameras.npz')['C'];visible,coverage=visibility(p,tri,centers,spacing)
    np.savez_compressed(a.out/'visibility.npz',visible=visible,coverage=coverage,
        geometry_sha256=hashlib.sha256(surface.read_bytes()).hexdigest(),camera_sha256=hashlib.sha256((a.cameras/'aligned-cameras.npz').read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mesh',type=Path,required=True);p.add_argument('--cameras',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fragment-faces',type=int,default=24);p.add_argument('--hole-radius',type=float,default=3.)
    run(p.parse_args())
