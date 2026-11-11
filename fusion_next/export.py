"""Single-surface export, geometry-aware narrow seam feathering, provenance colors."""
from common import *
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from edge_detail import gradient_map,choose_density,texture_jacobian,edge_covariance,sample_map
def run():
    detail=os.environ.get('GS_DETAIL','0')=='1';density_stats=[];contracted=0
    geo=np.load(OUT/'geometry.npz');labels=np.load(OUT/'ownership.npz');p,tri=geo['p'],geo['tri'];owner=labels['owner'];partner=labels['partner'];distance=labels['distance'];scores=labels['scores']
    images=photos();corr=np.load(OUT/'correspondences.npz');reliable=(corr['quality']>.4)&~corr['held'];tree=cKDTree(corr['target'][reliable])
    low=[cv.GaussianBlur(cv.resize(im,(1920,960)),(0,0),1.25) for im in images] if detail else []
    seam_offset=np.load(OUT/'seam-offset.npy') if (OUT/'seam-offset.npy').exists() else np.zeros((len(tri),3))
    prior=np.load(OUT/'prior.npz');prior_meta=json.loads((BASE/'mesh.json').read_text());prior_images=[cv.imread(str(BASE/g['texture'])) for g in prior_meta['groups']]
    prior_valid=np.array([float(im.std())>1 for im in prior_images])[prior['group']]
    fallback=(owner==4)&prior_valid
    # A new source must not replace a known texture on weak depth evidence
    # when its color contradicts that texture. No object names enter this gate.
    for k,im in enumerate(prior_images):
        ids=np.flatnonzero((prior['group']==k)&(owner<4)&prior_valid)
        if not len(ids):continue
        old=sample(im,prior['uv'][ids].mean(1))[:,::-1]/255
        if k<4:old=correct_rgb(old,k)
        new=labels['colors'][ids,owner[ids]]
        evidence_distance,_=tree.query(p[tri[ids]].mean(1))
        weak=((scores[ids]>scores[ids].max(1)[:,None]*.3).sum(1)<2)|(evidence_distance>.08)
        fallback[ids]=(np.linalg.norm(old-new,axis=1)>.25)&weak
    groups=[];meshrows=[];arrays=[];ids_all=[];blended=0;max_weight=0.;eligible_count=0
    if os.environ.get('GS_TERRITORY')=='1':
        np.savez_compressed(OUT/'effective-ownership.npz',owner=np.where(fallback,4,owner),fallback=fallback,prior_group=prior['group'])
    jobs=[(i,None) for i in range(4)]+[(4,k) for k in range(len(prior_images))]
    for source,prior_index in jobs:
        ids=np.flatnonzero((owner==source)&~fallback) if source<4 else np.flatnonzero(fallback&(prior['group']==prior_index))
        if not len(ids):continue
        xyz=p[tri[ids]]
        texture=uv(project(xyz.reshape(-1,3),source)).reshape(-1,3,2) if source<4 else prior['uv'][ids].copy()
        wrap=np.ptp(texture[:,:,0],axis=1)>.5;texture[:,:,0]+=wrap[:,None]&(texture[:,:,0]<.5)
        name=f'tex-{source}.jpg' if source<4 else f'prior-{prior_index}.jpg'
        meshrows.append(np.concatenate([xyz,texture],2).astype('<f4'));groups.append(dict(texture=name,start=sum(g['count'] for g in groups),count=len(ids)*3))
        import shutil
        image=images[source] if source<4 else prior_images[prior_index]
        color_source=source if source<4 else prior_index
        if color_source<4:
            adjusted=np.uint8(np.rint(correct_rgb(image[:,:,::-1]/255,color_source)*255))[:,:,::-1];cv.imwrite(str(OUT/name),adjusted,[cv.IMWRITE_JPEG_QUALITY,98])
        else:shutil.copyfile(BASE/prior_meta['groups'][prior_index]['texture'],OUT/name)
        edges=np.stack([xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,1],xyz[:,0]-xyz[:,2]],1);length=np.linalg.norm(edges,axis=2).max(1);area=np.linalg.norm(np.cross(edges[:,0],-edges[:,2]),axis=1)
        proximity=np.min(np.linalg.norm(xyz.mean(1)[:,None,:]-C[None],axis=2),axis=1);spacing=np.clip(.016+.004*proximity,.020,.040)
        subdivision=np.maximum(1,np.ceil(length/spacing)).astype(int)
        if detail:
            grad=gradient_map(image);original_n=subdivision.copy()
            subdivision,strength=choose_density(texture,subdivision,grad,image.shape[:2])
            jacobian=texture_jacobian(xyz,texture,image.shape[:2])
            density_stats.append(dict(source=int(source),prior_index=prior_index,faces=int(len(ids)),refined_faces=int((subdivision>original_n).sum()),original_samples=int(np.sum(original_n.astype('d')**2)),samples=int(np.sum(subdivision.astype('d')**2))))
        for n in np.unique(subdivision):
          selected=np.flatnonzero((subdivision==n)&(area>1e-10));u,v=np.indices((n,n));ok=u+v<n;uu=u[ok];vv=v[ok]
          bary=[np.c_[1-(uu+vv+2/3)/n,(uu+1/3)/n,(vv+1/3)/n]];ok=u+v<n-1;uu=u[ok];vv=v[ok]
          if len(uu):bary.append(np.c_[1-(uu+vv+4/3)/n,(uu+2/3)/n,(vv+2/3)/n])
          bary=np.vstack(bary);batch=max(1,20000//len(bary))
          for start in range(0,len(selected),batch):
            sel=selected[start:start+batch];faces=ids[sel];q=xyz[sel]-xyz[sel].mean(1)[:,None];cov=np.einsum('nki,nkj->nij',q,q)*(.4/n**2)
            eig,axes=np.linalg.eigh(cov);eig[:,0]=.0015**2;eig[:,1:]=np.maximum(eig[:,1:],.0015**2);axes[:,:,0]*=np.linalg.det(axes)[:,None]
            pos=np.einsum('bk,nkj->nbj',bary,xyz[sel]).reshape(-1,3);faceid=np.repeat(faces,len(bary))
            sample_uv=uv(project(pos,source)) if source<4 else np.einsum('bk,nkj->nbj',bary,texture[sel]).reshape(-1,2)
            rgb=sample(image,sample_uv)[:,::-1]/255
            if color_source<4:rgb=correct_rgb(rgb,color_source)
            if detail:
                local_cov=np.repeat(cov,len(bary),axis=0)
                local_cov,edge_strength=edge_covariance(local_cov,np.repeat(jacobian[sel],len(bary),axis=0),sample_map(grad,sample_uv))
                local_eig,local_axes=np.linalg.eigh(local_cov)
                local_eig[:,0]=.0015**2;local_eig[:,1:]=np.maximum(local_eig[:,1:],.0008**2)
                local_axes[:,:,0]*=np.linalg.det(local_axes)[:,None]
                contracted+=int((edge_strength>.1).sum())
            other=partner[faceid];band=(distance[faceid]<.055)&(other>=0)&(source<4);candidate=np.flatnonzero(band);eligible_count+=len(candidate)
            if len(candidate):
              close,_=tree.query(pos[candidate]);candidate=candidate[close<.10]
              for target in range(4):
                jj=candidate[other[candidate]==target]
                if not len(jj):continue
                color=correct_rgb(sample(images[target],uv(project(pos[jj],target)))[:,::-1]/255,target)
                err,conf,dep=cams[target].sample(cams[target].project(pos[jj]));difference=np.linalg.norm(rgb[jj]-color,axis=1)
                safe=(difference<.10)&(conf>.25)&(abs(err)<.06+.015*dep)&(scores[faceid[jj],target]>.4*scores[faceid[jj],source])
                weight=.45*np.clip(1-distance[faceid[jj]]/.055,0,1)*np.clip(1-difference/.10,0,1)*safe
                if detail:
                    primary_low=correct_rgb(sample(low[source],uv(project(pos[jj],source)))[:,::-1]/255,source)
                    secondary_low=correct_rgb(sample(low[target],uv(project(pos[jj],target)))[:,::-1]/255,target)
                    rgb[jj]=np.clip(rgb[jj]+weight[:,None]*np.clip(secondary_low-primary_low,-.05,.05),0,1)
                else:rgb[jj]=rgb[jj]*(1-weight[:,None])+color*weight[:,None]
                blended+=int((weight>.01).sum());max_weight=max(max_weight,float(weight.max()))
            if source<4:rgb=np.clip(rgb+seam_offset[faceid],0,1)
            a=np.empty((len(pos),14),'f');a[:,:3]=pos;a[:,3:6]=(rgb-.5)/.2820947918;a[:,6]=4.6;a[:,7:10]=np.repeat(.5*np.log(eig),len(bary),axis=0);a[:,10:14]=np.repeat(Rotation.from_matrix(axes).as_quat()[:,[3,0,1,2]],len(bary),axis=0)
            if detail:
                a[:,7:10]=.5*np.log(local_eig);a[:,10:14]=Rotation.from_matrix(local_axes).as_quat()[:,[3,0,1,2]]
            arrays.append(a);ids_all.append(np.full(len(a),source,'u1'))
        print('export source',stations[source] if source<4 else f'prior-{prior_index}',len(ids),'faces',flush=True)
    a=np.concatenate(arrays);source_ids=np.concatenate(ids_all)
    if not np.isfinite(a).all():raise ValueError('Non-finite export')
    rm.f.write(OUT/'fused.ply',a);rm.f.to_splat(OUT/'fused.splat',a)
    palette=np.array([[.95,.27,.23],[.2,.8,.35],[.25,.5,.98],[.98,.7,.15],[.8,.8,.8]])
    diagnostic=a.copy();diagnostic[:,3:6]=(palette[source_ids]-.5)/.2820947918;rm.f.to_splat(OUT/'owners.splat',diagnostic)
    rows=np.concatenate(meshrows);rows.tofile(OUT/'mesh.bin');meta=dict(groups=groups,triangles=len(rows),splats=len(a),method='validated dense-flow geometry + color-seam graph ownership + gated narrow surface feathering',centers=C.tolist(),up=rm.meta['up'],stations=stations)
    # No stale object-specific refinements are presented as part of this algorithm.
    for key in ['appearance_refinement','smoothing','reused_patches','table_duplicate_triangles_removed','globe_removed_triangles']:meta.pop(key,None)
    meta['initialization_note']='v3 geometry prior contains earlier TV/globe/table fits; no lamp template or semantic branch is used in this pipeline.'
    if detail:meta['method']='image-gradient adaptive surface Gaussians / edge-normal contraction / low-frequency seam blending'
    write_json('mesh.json',meta);write_json('fusion.json',meta)
    if detail:write_json('detail-export-report.json',dict(method='image-gradient adaptive density / edge-normal covariance contraction / low-frequency seam blending',groups=density_stats,contracted_splats=contracted,splats=len(a),maximum_density_budget_factor=1.8,minimum_edge_axis_scale=.7,semantic_rules=False,synthetic_texture=False))
    write_json('export-report.json',dict(splats=len(a),faces=len(rows),unsupported_faces_removed=int(((owner==4)&~fallback).sum()),prior_texture_retained_faces=int(fallback.sum()),eligible_seam_splats=eligible_count,blended_splats=blended,blend_fraction=blended/len(a),maximum_secondary_weight=max_weight,band_width_model_units=.055,notes=['Geometry ownership is fixed in world space','Gray diagnostic regions retain the valid initialization texture when new evidence is weak','Mesh preview uses hard texture seams; PLY includes narrow-band color feathering','Surface Gaussians are regenerated, not jointly trained original Gaussians']))
if __name__=='__main__':run()
