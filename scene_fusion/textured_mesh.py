"""Bake calibrated panoramas onto one triangle surface, with visibility ownership.

This does not treat Gaussian centers as a measured surface. The input is an
explicit reconstructed mesh; its errors remain a separate geometric limitation.
No object names, hand-painted masks, or fixed camera counts are used.
"""
import argparse, json, zipfile, hashlib
from pathlib import Path
import numpy as np
import cv2 as cv
from numba import njit
from scipy.sparse import coo_matrix
from core import load, prep, write_json, scene_id
from surface_export import adjacency, sample_photo
from surface_volume import fields
from mesh_cleanup import remove_unsupported_islands


@njit(cache=True)
def raster(xy, z, triangles, height, width, perspective=True):
    depth = np.full((height, width), np.inf, np.float32)
    face = np.full((height, width), -1, np.int32)
    bary = np.zeros((height, width, 2), np.float32)
    for fi in range(len(triangles)):
        ia, ib, ic = triangles[fi]
        za, zb, zc = z[ia], z[ib], z[ic]
        if min(za, zb, zc) <= .01:
            continue
        ax, ay = xy[ia]; bx, by = xy[ib]; cx, cy = xy[ic]
        det = (by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(det) < 1e-9:
            continue
        x0 = max(0, int(np.floor(min(ax, bx, cx))))
        x1 = min(width-1, int(np.ceil(max(ax, bx, cx))))
        y0 = max(0, int(np.floor(min(ay, by, cy))))
        y1 = min(height-1, int(np.ceil(max(ay, by, cy))))
        for y in range(y0, y1+1):
            for x in range(x0, x1+1):
                a = ((by-cy)*(x-cx)+(cx-bx)*(y-cy))/det
                b = ((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/det
                c = 1-a-b
                if min(a,b,c) < -1e-6:
                    continue
                d = 1/(a/za+b/zb+c/zc) if perspective else a*za+b*zb+c*zc
                if d < depth[y,x]:
                    depth[y,x] = d; face[y,x] = fi
                    if perspective:
                        bary[y,x,0] = a/za*d; bary[y,x,1] = b/zb*d
                    else:
                        bary[y,x,0] = a; bary[y,x,1] = b
    return depth, face, bary


def erp_uv(p, C, Q):
    q = (p-C)@Q
    return np.c_[np.arctan2(-q[:,1],q[:,0])/(2*np.pi)+.5,
                 np.arctan2(np.linalg.norm(q[:,:2],axis=1),q[:,2])/np.pi]


def unwrap_triangle_uv(uv):
    result = uv.copy()
    wrapped = np.ptp(result[:,:,0],axis=1) > .5
    result[:,:,0] += wrapped[:,None] & (result[:,:,0]<.5)
    return result


def smooth_mesh(p, tri, limit, iterations=5, protected=None):
    original = p.copy()
    original_normal = np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]])
    edge, count = np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0,return_counts=True)
    a,b = edge.T
    fixed = np.zeros(len(p),bool); fixed[np.unique(edge[count!=2])]=True
    if protected is not None: fixed|=protected
    normal = np.zeros_like(p)
    for k in range(3): np.add.at(normal,tri[:,k],original_normal)
    normal /= np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-10)
    dot = (normal[a]*normal[b]).sum(1)
    fixed[np.unique(edge[dot<.75])]=True
    w = np.clip((dot-.75)/.25,0,1)**3
    W = coo_matrix((np.r_[w,w],(np.r_[a,b],np.r_[b,a])),shape=(len(p),len(p))).tocsr()
    degree = np.maximum(np.asarray(W.sum(1)).ravel(),1e-10)
    for _ in range(iterations):
        average = W@p/degree[:,None]
        step = np.sum((average-p)*normal,axis=1)*.35; step[fixed]=0
        shift = np.clip(np.sum((p-original)*normal,axis=1)+step,-limit,limit)
        p = original+shift[:,None]*normal
    for _ in range(64):
        new_normal=np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]])
        bad=(new_normal*original_normal).sum(1)<=0
        if not bad.any(): break
        p[np.unique(tri[bad])]=original[np.unique(tri[bad])]
    new_normal=np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]])
    bad=(new_normal*original_normal).sum(1)<=0
    if bad.any():
        p=original.copy();bad[:]=False
    return p, dict(max_displacement=float(np.linalg.norm(p-original,axis=1).max()),
                   locked_vertices=int(fixed.sum()),invalid_faces=int(bad.sum()))


def face_probes(p,tri):
    center=p[tri].mean(1)
    return np.concatenate([center[:,None],p[tri]*.8+center[:,None]*.2],axis=1)


def mesh_visibility(p,tri,C,Q,size,tolerance,footprint=False):
    centers=face_probes(p,tri).reshape(-1,3) if footprint else p[tri].mean(1); q=(centers-C)@Q
    faceid=np.argmax(np.stack([q@b[:,2] for b in prep.FACES]),axis=0)
    visible=np.zeros(len(centers),bool)
    for k,b in enumerate(prep.FACES):
        v=(p-C)@Q@b; xy=(v[:,:2]/np.maximum(v[:,2,None],1e-7)+1)*size/2-.5
        depth,_,_=raster(xy.astype('f'),v[:,2].astype('f'),tri,size,size)
        ids=np.flatnonzero(faceid==k); v=(centers[ids]-C)@Q@b
        uv=np.rint((v[:,:2]/np.maximum(v[:,2,None],1e-7)+1)*size/2-.5).astype(int).clip(0,size-1)
        observed=depth[uv[:,1],uv[:,0]]
        visible[ids]=np.isfinite(observed)&(v[:,2]>.01)&(v[:,2]<=observed+tolerance)
    if footprint:
        samples=visible.reshape(-1,4)
        return samples.any(1),samples.mean(1)
    return visible


def choose_owners(p,tri,C,Q,views,images,spacing,hints=None,footprint=False,cached_visibility=None,photometric_report=None):
    center=p[tri].mean(1); normal=np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]])
    area=np.linalg.norm(normal,axis=1); normal/=np.maximum(area[:,None],1e-10)
    scores=np.zeros((len(tri),len(C)),'f'); colors=np.zeros((len(tri),len(C),3),'f')
    eligible=np.zeros(len(tri),bool)
    coverages=np.ones((len(tri),len(C)),'f')
    for i,view in enumerate(views):
        if cached_visibility is not None:
            coverage=cached_visibility[1][:,i]
            # Geometry survives if any footprint sample is seen; an occluded
            # centroid must not delete a triangle with visible corners.
            visible=cached_visibility[0][:,i]|(coverage>0)
        elif footprint:visible,coverage=mesh_visibility(p,tri,C[i],Q[i],768,spacing*.7,True)
        else:visible=mesh_visibility(p,tri,C[i],Q[i],768,spacing*.7);coverage=np.ones(len(tri))
        delta,confidence,dist=view.sample(center)
        angle=np.abs(np.sum((C[i]-center)*normal,axis=1))/np.maximum(dist,1e-5)
        depth_support=np.exp(-np.abs(delta)/(.18+.04*dist))
        scores[:,i]=visible*(.1+.9*angle)**1.5/np.maximum(dist,.4)**1.6*(.2+.8*confidence)*depth_support
        eligible|=scores[:,i]>1e-5
        if footprint:
            delta_probe,confidence_probe,dist_probe=view.sample(face_probes(p,tri).reshape(-1,3))
            support=((abs(delta_probe)<.10+.03*dist_probe)&(confidence_probe>.15)).reshape(-1,4).mean(1)
            scores[:,i]*=np.maximum(.03,coverage**4)*(.1+.9*support**2)
        colors[:,i]=sample_photo(images[i],center,C[i],Q[i])
        coverages[:,i]=coverage
        print('Visibility/texture evidence',i+1,int(visible.sum()),flush=True)
    # Footprint penalties select a safer photo; they must not delete geometry
    # merely because the numerical scoring scale changed.
    if photometric_report is not None:
        from texture_patch_consensus import reject_photo_outliers
        scores,report=reject_photo_outliers(scores,colors,coverages)
        photometric_report.update(report);print('Photo consensus',report,flush=True)
    good=eligible&(area>1e-10)&(scores.max(1)>0)
    owner=scores.argmax(1); neighbors=adjacency(tri)
    unary=-np.log(np.maximum(scores,1e-20)/np.maximum(scores.max(1)[:,None],1e-20))
    # Checkerboard updates reduce oscillation; pair costs penalize visible seams.
    ids=np.arange(len(tri))
    for _ in range(8):
        for parity in [0,1]:
            costs=unary.copy()
            for i in range(len(C)):
                for k in range(3):
                    other=owner[neighbors[:,k]]
                    difference=np.linalg.norm(colors[:,i]-colors[ids,other],axis=1)
                    costs[:,i]+=(other!=i)*(.45+np.minimum(difference,.5)*3)
            take=(ids%2==parity)&good; owner[take]=costs[take].argmin(1)
    if hints is not None:
        # A verified planar patch uses one appearance source, avoiding mixtures
        # of reflections. The patch still needs visibility from that source.
        use=(hints>=0)&(hints<len(C)); selected=np.flatnonzero(use)
        selected=selected[scores[selected,hints[selected]]>1e-6]
        owner[selected]=hints[selected]
    return owner,good,scores,colors


def boundary_weights(tri,owner,count,vertices):
    weights=np.zeros((vertices,count),'f')
    for k in range(3): np.add.at(weights,(tri[:,k],owner),1)
    weights/=np.maximum(weights.sum(1)[:,None],1)
    return weights


def blend_linear(base,other,weight):
    def linear(x): return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
    color=linear(base)*(1-weight[:,None])+linear(other)*weight[:,None]
    return np.where(color<=.0031308,color*12.92,1.055*np.maximum(color,0)**(1/2.4)-.055)


def bake(source,out,geometry,texture_width=4096,smoothing=.6,warped=False,footprint=False,clean_fragments=False,visibility_cache=None,coherent=False,graphcut=False,photo_consensus=False,conflict_regions=False):
    if not 512<=texture_width<=8192 or texture_width%2 or not 0<=smoothing<=1:
        raise ValueError('Texture width must be even, 512–8192; smoothing must be 0–1')
    out.mkdir(parents=True,exist_ok=True); meta=load(source)
    for item in meta['stations']:
        if hashlib.sha256(Path(item['photo']).read_bytes()).hexdigest()!=item['photo_sha256']:
            raise ValueError('Panorama changed after camera calibration: '+item['label'])
    cameras=np.load(source/'aligned-cameras.npz'); C,Q=cameras['C'],cameras['Q']
    data=np.load(geometry); p=data['p'].astype('f'); tri=data['tri'].astype('i')
    cached=None
    if visibility_cache:
        if smoothing!=0:raise ValueError('Cached ray visibility requires unchanged geometry (smoothing=0)')
        cache=np.load(visibility_cache)
        if str(cache['geometry_sha256'])!=hashlib.sha256(geometry.read_bytes()).hexdigest() or str(cache['camera_sha256'])!=hashlib.sha256((source/'aligned-cameras.npz').read_bytes()).hexdigest():
            raise ValueError('Ray visibility cache belongs to different geometry or cameras')
        cached=(cache['visible'],cache['coverage'])
        if any(v.shape!=(len(tri),len(C)) for v in cached):raise ValueError('Invalid visibility cache dimensions')
    area=np.linalg.norm(np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]]),axis=1)
    nondegenerate=area>1e-10; degenerate_removed=int((~nondegenerate).sum())
    if cached is not None:cached=tuple(v[nondegenerate] for v in cached)
    hints=data['owner_hint'][nondegenerate] if 'owner_hint' in data else None; tri=tri[nondegenerate]
    required=data['geometry_required'][nondegenerate] if 'geometry_required' in data else np.zeros(len(tri),bool)
    protected=np.zeros(len(p),bool);protected[np.unique(tri[required])]=True
    edges=np.linalg.norm(p[tri[:,1]]-p[tri[:,0]],axis=1)
    spacing=float(np.median(edges)); p,smooth_report=smooth_mesh(p,tri,spacing*smoothing,protected=protected)
    print('Mesh smoothing',smooth_report,flush=True)
    views=fields(source,warped)
    images=[cv.resize(cv.imdecode(np.fromfile(s['photo'],np.uint8),1),(2048,1024)) for s in meta['stations']]
    color_report={} if photo_consensus else None
    owner,good,scores,colors=choose_owners(p,tri,C,Q,views,images,spacing,hints,footprint,cached,color_report)
    patch_report=None
    if graphcut:
        from texture_graphcut import assign
        owner,patch_report=assign(p,tri,scores,good,owner,spacing)
        write_json(out/'texture-graphcut.json',patch_report);print(patch_report,flush=True)
    if coherent:
        from texture_patch_consensus import coherent_patches
        patch_area=np.linalg.norm(np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]]),axis=1)*.5
        owner,patch_report=coherent_patches(tri,owner,scores,colors,good,adjacency(tri),patch_area,spacing)
        write_json(out/'texture-patches.json',patch_report);print(patch_report,flush=True)
    conflict_report=None
    if conflict_regions:
        from texture_patch_consensus import coherent_conflict_regions
        face_area=np.linalg.norm(np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]]),axis=1)*.5
        owner,conflict_report=coherent_conflict_regions(owner,scores,colors,good,adjacency(tri),face_area)
        write_json(out/'texture-conflicts.json',conflict_report)
        print('Conflict regions',len(conflict_report['regions']),conflict_report['reassigned_faces'],flush=True)
    cleanup_report=None
    if clean_fragments:
        selected=np.flatnonzero(good|required)
        keep,cleanup_report=remove_unsupported_islands(p,tri[selected],views,spacing,required[selected])
        good[selected[~keep]]=False
        write_json(out/'fragment-cleanup.json',cleanup_report)
        print('Fragment cleanup',cleanup_report['removed_faces'],'faces',flush=True)
    unknown_tri=tri[required&~good];required_count=int(required.sum());known_required=required[good]
    rejected=int((~good&~required).sum()); tri=tri[good]; owner=owner[good]; scores=scores[good]
    np.savez_compressed(out/'surface-owned.npz',p=p,tri=np.vstack([tri,unknown_tri]),owner=np.r_[owner,np.full(len(unknown_tri),-1)],scores=np.vstack([scores,np.zeros((len(unknown_tri),len(C)))]),geometry_required=np.r_[known_required,np.ones(len(unknown_tri),bool)])
    weights=boundary_weights(tri,owner,len(C),len(p))
    rows=[]; groups=[]; offset=0; blended=[]
    H,W=texture_width//2,texture_width
    for i,item in enumerate(meta['stations']):
        ids=np.flatnonzero(owner==i)
        if not len(ids): continue
        uv=unwrap_triangle_uv(erp_uv(p[tri[ids]].reshape(-1,3),C[i],Q[i]).reshape(-1,3,2))
        xyz=p[tri[ids]]; index=np.arange(len(ids)*3,dtype='i').reshape(-1,3)
        image=cv.resize(cv.imdecode(np.fromfile(item['photo'],np.uint8),1),(W,H),interpolation=cv.INTER_AREA)
        original=image.copy()
        # Rasterize only boundary faces to bake a narrow one-ring feather band.
        seam=np.any(weights[tri[ids],i]<.999,axis=1)
        seam_ids=np.flatnonzero(seam)
        blend_count=0
        if len(seam_ids):
            vertices=xyz[seam_ids].reshape(-1,3)
            tx=uv[seam_ids].reshape(-1,2)*[W,H]-.5
            # ERP wrap duplicates are clipped into the same texture, not stretched.
            ix=np.arange(len(vertices),dtype='i').reshape(-1,3)
            z=np.linalg.norm(vertices-C[i],axis=1).astype('f')
            for shift in [0,-W]:
                _,face,bary=raster((tx+[shift,0]).astype('f'),z,ix,H,W,False)
                yy,xx=np.nonzero(face>=0)
                for start in range(0,len(yy),24000):
                    y,x=yy[start:start+24000],xx[start:start+24000]
                    f=seam_ids[face[y,x]]; b=np.c_[bary[y,x],1-bary[y,x].sum(1)]
                    pos=np.einsum('nk,nkj->nj',b,xyz[f])
                    w=np.einsum('nk,nkj->nj',b,weights[tri[ids[f]]]); w[:,i]=0
                    secondary=w.argmax(1); amount=np.minimum(.5,w.max(1))
                    base=original[y,x,::-1].astype('f')/255; result=base.copy()
                    for j in np.unique(secondary):
                        chosen=np.flatnonzero((secondary==j)&(amount>1e-5))
                        if not len(chosen): continue
                        rgb=sample_photo(images[j],pos[chosen],C[j],Q[j])
                        # Never smear an occlusion edge or a differently reflected object.
                        difference=np.linalg.norm(rgb-base[chosen],axis=1)
                        gate=np.clip((.14-difference)/.08,0,1)
                        delta,conf,dist=views[j].sample(pos[chosen])
                        gate*=((abs(delta)<.10+.03*dist)&(conf>.15)&(scores[ids[f[chosen]],j]>1e-6))
                        a=amount[chosen]*gate
                        result[chosen]=blend_linear(base[chosen],rgb,a)
                        blend_count+=int((a>.01).sum())
                    image[y,x]=np.uint8(np.clip(result[:,::-1]*255,0,255))
        name=f'texture-{i+1}.jpg'; cv.imwrite(str(out/name),image,[cv.IMWRITE_JPEG_QUALITY,96])
        rows.append(np.c_[xyz.reshape(-1,3),uv.reshape(-1,2)].astype('<f4'))
        groups.append(dict(source=i,label=item['label'],start=offset,count=len(ids)*3,texture=name)); offset+=len(ids)*3
        blended.append(blend_count); print('Baked source',i+1,'triangles',len(ids),'feather pixels',blend_count,flush=True)
    if len(unknown_tri):
        # Missing photo support must not cut a validated closed surface open.
        # Unobserved texture is explicitly neutral, never sampled through an occluder.
        name='unobserved.jpg';cv.imwrite(str(out/name),np.full((8,8,3),127,'u1'))
        unknown=p[unknown_tri].reshape(-1,3);rows.append(np.c_[unknown,np.full((len(unknown),2),.5)].astype('<f4'))
        groups.append(dict(source=-1,label='未观测表面',start=offset,count=len(unknown),texture=name))
    rows=np.concatenate(rows); rows.tofile(out/'mesh.bin')
    # View poses come from this scene, never from a different reconstruction.
    dataset=json.loads((source/'data-512/dataset.json').read_text(encoding='utf8'))
    up=Q[:,:,2].mean(0); up/=np.linalg.norm(up)
    result=dict(triangles=len(tri)+len(unknown_tri),vertices=len(p),groups=groups,centers=C.tolist(),up=up.tolist(),
                stations=[s['label'] for s in meta['stations']],views=dataset['views'],midpoint_edges=dataset['midpoint_edges'],
                method='One reconstructed triangle surface; visible coherent photo ownership; gated one-ring linear-light seam feather',
                input_surface=str(geometry),surface_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest(),
                camera_sha256=hashlib.sha256((source/'aligned-cameras.npz').read_bytes()).hexdigest(),
                warped_depth=warped,smoothing=smooth_report,degenerate_faces_removed=degenerate_removed,unobserved_faces_removed=rejected,
                protected_geometry_faces=required_count,protected_untextured_faces=len(unknown_tri),
                footprint_visibility=footprint,visibility_method='raycast' if cached is not None else 'raster',
                fragment_cleanup=None if cleanup_report is None else {k:v for k,v in cleanup_report.items() if k!='candidates'},
                texture_size=[W,H],feather_pixels=blended,experimental=True)
    result['texture_patch_consensus']=patch_report
    result['texture_conflict_regions']=conflict_report
    result['photo_outlier_rejection']=color_report
    capture_path=source/'capture.json'
    if capture_path.exists():
        capture=json.loads(capture_path.read_text('utf8'))
        result.update(source_depth='single-GS-rendered',source_model_sha256=capture['model_sha256'],
            reference_scene=scene_id(Path(capture['model']).parent),
            preview_title='单一 PLY 重建 · 高清原片贴图',
            preview_note='从已融合单一高斯的渲染深度提取网格，再按可见性投影原始全景。细结构可能变形或缺失，不能视作无损转换。')
    repair_report=geometry.parent/'repair.json'
    if repair_report.exists():
        result['surface_repair']=json.loads(repair_report.read_text('utf8'))
        result['preview_title']='单一 PLY 网格 · 碎片清理与局部补洞'
        result['preview_note']='使用连通片清理、有限尺寸补洞和射线可见性贴图。小碎片和孔洞减少；椅子等复杂遮挡处仍可能缺损，小孤立真实细节也可能被清理。'
    if capture_path.exists() and capture.get('representation')=='2D surface Gaussians':
        result.update(source_depth='surface-Gaussian-median',reference_scene=None,preview_title='表面约束重建 · 原片高清贴图',
            preview_note='从原照片优化表面高斯的法线与深度一致性，使用中值深度重建网格并贴原片。仍是实验模型；斜视、细结构和未观测区域可能有缺损。')
    poisson_path=geometry.parent/'poisson.json'
    if poisson_path.exists():
        from oblique_views import make_views
        result.update(poisson_reconstruction=json.loads(poisson_path.read_text('utf8')),preview_title='连续表面重建 · 原片高清贴图',
            inspection_views=make_views(C,Q,dataset['midpoint_edges']),
            preview_note='使用带法线的表面样本进行泊松重建，再按射线可见性贴原片。补洞包含推断；未观测表面显示灰色，细杆、椅背和遮挡边缘仍可能变形。')
        if result['poisson_reconstruction'].get('point_smoothing'):
            result.update(preview_title='三维几何磨皮 · 高清原片贴图',geometry_guide_scene=geometry.parent.name,
                preview_note='先平滑重建引导点的位置与法线，再进行连续曲面重建和原片贴图。保留原始高斯；补面包含推断，细结构和部分边缘仍可能变形。')
        if result['poisson_reconstruction'].get('intentional_detail_loss'):
            result.update(preview_title='低频曲面重建 · 光滑优先',geometry_guide_scene=geometry.parent.name,
                preview_note='主动减少几何采样并平滑曲面，再贴高清原片。轮廓更圆滑，但细杆、缝隙及薄物体可能变钝或消失；不是精确结构拟合。')
            if result['poisson_reconstruction'].get('adaptive_edges'):
                result.update(preview_title='自适应几何平滑 · 高清原片贴图',
                    preview_note='低频曲面重建，并在多张照片支持的边缘保留较密采样。大面更平顺；电视、灯等边缘仍可能弯曲或串色，原基线保留。')
        if result['poisson_reconstruction'].get('planar_guide'):
            result.update(preview_title='局部平面约束 · 高清原片贴图',geometry_guide_scene=geometry.parent.name,
                preview_note='仅在多机位支持的局部平坦区域整理几何引导点，再重建并贴原片。仍包含形状推断，边缘与薄物体可能有误差。')
    repair_path=geometry.parent/'sphere-repair.json'
    weld_path=geometry.parent/'numeric-weld.json'
    if weld_path.exists():result['numeric_weld']=json.loads(weld_path.read_text('utf8'))
    denoise_path=geometry.parent/'mesh-denoise.json'
    if denoise_path.exists():
        from oblique_views import make_views
        result.update(mesh_denoise=json.loads(denoise_path.read_text('utf8')),geometry_guide_scene=geometry.parent.name,
            inspection_views=make_views(C,Q,dataset['midpoint_edges']),preview_title='曲面保边降噪 · 连贯原片贴图',
            preview_note='对面法线滤波并限制顶点移动，再按遮挡证据整理小块照片分区。保持网格连接关系；细结构与反射接缝仍可能有误差。')
    if repair_path.exists():
        repair=json.loads(repair_path.read_text(encoding='utf8'))
        result['repaired_surfaces']=[dict(center=s['center'],radius=s['radius'],type='sphere',supporting_cameras=s['validation']['supporting_cameras']) for s in repair['candidates'] if s['validation']['accepted']]
    if result.get('geometry_guide_scene'):result['geometry_guide_scene']=scene_id(geometry.parent)
    if capture_path.exists():
        model_meta=Path(capture['model']).parent/'data/dataset.json'
        if model_meta.exists():result['native_scene']=json.loads(model_meta.read_text('utf8')).get('native_scene')
    write_json(out/'mesh.json',result)
    with (out/'room.obj').open('w',encoding='utf8') as f:
        f.write('mtllib room.mtl\n'); np.savetxt(f,rows[:,:3],fmt='v %.6f %.6f %.6f')
        np.savetxt(f,np.c_[rows[:,3],1-rows[:,4]],fmt='vt %.7f %.7f')
        for i,g in enumerate(groups):
            f.write(f'usemtl photo{i}\n'); idx=np.arange(g['start']+1,g['start']+g['count']+1).reshape(-1,3)
            np.savetxt(f,np.repeat(idx,2,axis=1),fmt='f %d/%d %d/%d %d/%d')
    (out/'room.mtl').write_text(''.join(f'newmtl photo{i}\nKa 1 1 1\nKd 1 1 1\nillum 1\nmap_Kd {g["texture"]}\n\n' for i,g in enumerate(groups)),encoding='utf8')
    with zipfile.ZipFile(out/'textured-room.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in ['room.obj','room.mtl','mesh.json']+[g['texture'] for g in groups]: archive.write(out/name,name)
    print('Completed',len(tri)+len(unknown_tri),'triangles',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--geometry',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--texture-width',type=int,default=4096); parser.add_argument('--smoothing',type=float,default=.6)
    parser.add_argument('--warped',action='store_true')
    parser.add_argument('--footprint',action='store_true')
    parser.add_argument('--clean-fragments',action='store_true')
    parser.add_argument('--visibility-cache',type=Path)
    parser.add_argument('--coherent-patches',action='store_true')
    parser.add_argument('--graphcut',action='store_true',help='Optional PyMaxflow mesh photo assignment')
    parser.add_argument('--photo-consensus',action='store_true')
    parser.add_argument('--conflict-regions',action='store_true',help='Experimental bounded single-photo conflict patches; defaults unchanged')
    a=parser.parse_args(); bake(a.source,a.out,a.geometry,a.texture_width,a.smoothing,a.warped,a.footprint,a.clean_fragments,a.visibility_cache,a.coherent_patches,a.graphcut,a.photo_consensus,a.conflict_regions)
