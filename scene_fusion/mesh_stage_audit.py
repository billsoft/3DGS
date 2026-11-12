"""Raycast mesh stages with identical calibrated rays to locate geometry loss."""
import argparse,json
from pathlib import Path
import numpy as np
import open3d as o3d

def run(a):
    meta=json.loads((a.cameras/'data-512/dataset.json').read_text('utf8'))
    views=[v for v in meta['views'] if v['source'] is not None and v['name'].endswith(('-3','-4','-5'))]
    if a.views:views+=json.loads(a.views.read_text('utf8'))['views']
    capture=json.loads((a.depth_capture/'capture.json').read_text('utf8')) if a.depth_capture else None
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
            record=dict(view=v['name'],misses=int((~np.isfinite(d)).sum()))
            if capture:
                frame=next((f for f in capture['frames'] if np.allclose(f['viewmat'],v['viewmat'],atol=1e-6)),None)
                if frame:
                    depth_data=np.load(a.depth_capture/frame['depth']);source=depth_data['depth'];source_alpha=depth_data['alpha']
                    ix=np.minimum(((np.arange(size)+.5)*source.shape[0]/size).astype(int),source.shape[0]-1)
                    dd=source[ix[:,None],ix];aa=source_alpha[ix[:,None],ix]
                    supported=np.isfinite(dd)&(dd>.1)&(dd<30)&(aa>=.95)
                    record['misses_despite_valid_input_depth']=int((~np.isfinite(d)&supported).sum())
            if 'owner' in z:
                primitive=hit['primitive_ids'].numpy();valid=np.isfinite(d);unknown=np.zeros_like(valid)
                unknown[valid]=z['owner'][primitive[valid]]<0
                record['untextured_hits']=int(unknown.sum())
            records.append(record)
        depths.append(rendered)
        results.append(dict(mesh=str(path),faces=len(tri),components=len(counts),tiny_components=int((counts<32).sum()),tiny_faces=int(counts[counts<32].sum()),boundary_edges=int((ec==1).sum()),nonmanifold_edges=int((ec>2).sum()),views=records))
        print(path.parent.name,'faces',len(tri),'views',len(records),'missed rays',sum(v['misses'] for v in records),flush=True)
    for stage in range(1,len(depths)):
        for j,v in enumerate(views):
            before,after=depths[0][j],depths[stage][j];valid=np.isfinite(before)
            results[stage]['views'][j]['lost_hits']=int((valid&~np.isfinite(after)).sum())
            results[stage]['views'][j]['surface_retreat_gt_0_1']=int((valid&(after>before+.1)).sum())
            common=valid&np.isfinite(after);delta=after[common]-before[common]
            results[stage]['views'][j]['surface_advance_gt_0_1']=int((delta<-.1).sum())
            results[stage]['views'][j]['absolute_depth_change_median']=float(np.median(abs(delta))) if len(delta) else None
            results[stage]['views'][j]['absolute_depth_change_p95']=float(np.quantile(abs(delta),.95)) if len(delta) else None
    (a.out/'audit.json').write_text(json.dumps(results,indent=2),encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cameras',type=Path,required=True);p.add_argument('--mesh',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--size',type=int,default=512)
    p.add_argument('--views',type=Path,help='Additional calibrated inspection views, including translated grazing angles')
    p.add_argument('--depth-capture',type=Path,help='Trace original-camera misses to the pre-integration synthetic depth/alpha')
    run(p.parse_args())
