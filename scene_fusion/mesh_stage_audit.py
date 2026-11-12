"""Raycast mesh stages with identical calibrated rays to locate geometry loss."""
import argparse,json
from pathlib import Path
import numpy as np
import open3d as o3d

def run(a):
    meta=json.loads((a.cameras/'data-512/dataset.json').read_text('utf8'))
    views=[v for v in meta['views'] if v['source'] is not None and v['name'].endswith(('-3','-4','-5'))]
    a.out.mkdir(parents=True,exist_ok=True);results=[];depths=[]
    for path in a.mesh:
        z=np.load(path);p=z['p'];tri=z['tri']
        legacy=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(p),o3d.utility.Vector3iVector(tri))
        comp,counts,areas=legacy.cluster_connected_triangles();counts=np.asarray(counts);areas=np.asarray(areas)
        edges=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1)
        _,ec=np.unique(edges,axis=0,return_counts=True)
        scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
        rendered=[];records=[];size=a.size;f=size/2
        K=np.array([[f,0,f],[0,f,f],[0,0,1.]])
        for v in views:
            rays=scene.create_rays_pinhole(K,np.array(v['viewmat']),size,size)
            hit=scene.cast_rays(rays);d=hit['t_hit'].numpy().copy();rendered.append(d)
            normal=abs(hit['primitive_normals'].numpy());normal[~np.isfinite(d)]=0
            o3d.io.write_image(str(a.out/(path.parent.name+'-'+v['name']+'.png')),o3d.geometry.Image(np.uint8(np.clip(normal,0,1)*255)))
            records.append(dict(view=v['name'],misses=int((~np.isfinite(d)).sum())))
        depths.append(rendered)
        results.append(dict(mesh=str(path),faces=len(tri),components=len(counts),tiny_components=int((counts<32).sum()),tiny_faces=int(counts[counts<32].sum()),boundary_edges=int((ec==1).sum()),nonmanifold_edges=int((ec>2).sum()),views=records))
        print(results[-1],flush=True)
    for stage in range(1,len(depths)):
        for j,v in enumerate(views):
            before,after=depths[0][j],depths[stage][j];valid=np.isfinite(before)
            results[stage]['views'][j]['lost_hits']=int((valid&~np.isfinite(after)).sum())
            results[stage]['views'][j]['surface_retreat_gt_0_1']=int((valid&(after>before+.1)).sum())
    (a.out/'audit.json').write_text(json.dumps(results,indent=2),encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cameras',type=Path,required=True);p.add_argument('--mesh',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--size',type=int,default=512)
    run(p.parse_args())
