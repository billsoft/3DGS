"""Discover and validate closed spherical surface hypotheses, without scene ROIs.

Geometry holdout, angular coverage, image contour proximity and robust depth
votes are independent gates. Photo edges alone are not silhouette certification.
"""
import argparse,json
from pathlib import Path
import numpy as np
import cv2 as cv
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from core import load,write_json
from surface_volume import fields
from textured_mesh import erp_uv

def propose_spheres(p,spacing,seed_count=7000):
    tree=cKDTree(p);rng=np.random.default_rng(20261112)
    seeds=rng.choice(len(p),min(seed_count,len(p)),False)
    _,neighbors=tree.query(p[seeds],k=min(100,len(p)),workers=4)
    candidates=[]
    for indices in neighbors:
        q=p[indices].astype(float);origin=q.mean(0);q-=origin
        coef=np.linalg.lstsq(np.c_[2*q,np.ones(len(q))],(q*q).sum(1),rcond=None)[0]
        radius=np.sqrt(max(0,coef[3]+coef[:3]@coef[:3]));center=coef[:3]+origin
        if not 4*spacing<radius<28*spacing:continue
        error=np.abs(np.linalg.norm(q-coef[:3],axis=1)-radius)
        if np.median(error)>max(.025*radius,.15*spacing):continue
        if np.linalg.eigvalsh(np.cov(q.T))[0]<.005*spacing**2:continue
        if any(np.linalg.norm(center-c)<spacing*1.5 and abs(radius-r)<spacing for c,r in candidates):continue
        candidates.append((center,radius))
    results=[]
    for center,radius in candidates:
        q=p[tree.query_ball_point(center,radius+spacing*2)]
        q=q[abs(np.linalg.norm(q-center,axis=1)-radius)<max(.055*radius,.65*spacing)]
        if len(q)<160:continue
        direction=(q-center)/np.linalg.norm(q-center,axis=1)[:,None]
        eigen=np.linalg.eigvalsh(direction.T@direction/len(q))
        bins=np.c_[np.floor((np.arctan2(direction[:,1],direction[:,0])+np.pi)/(2*np.pi)*16).clip(0,15),np.floor((direction[:,2]+1)*4).clip(0,7)].astype(int)
        coverage=len(np.unique(bins,axis=0))/128
        if coverage<.55 or eigen[0]<.20:continue
        fit=least_squares(lambda x:np.linalg.norm(q[::2]-x[:3],axis=1)-x[3],np.r_[center,radius],loss='soft_l1',f_scale=spacing*.5,max_nfev=40).x
        error=abs(np.linalg.norm(q[1::2]-fit[:3],axis=1)-fit[3])
        if fit[3]<=0 or np.median(error)/fit[3]>.03 or np.quantile(error,.9)/fit[3]>.08:continue
        results.append(dict(center=fit[:3].tolist(),radius=float(fit[3]),support_points=len(q),angular_coverage=coverage,eigenvalues=eigen.tolist(),held_median=float(np.median(error)),held_p90=float(np.quantile(error,.9)),score=float(coverage*np.sqrt(len(q))*eigen[0]/max(np.median(error)/radius,.01))))
    results.sort(key=lambda x:-x['score']);unique=[]
    for candidate in results:
        if any(np.linalg.norm(np.array(candidate['center'])-s['center'])<max(candidate['radius'],s['radius']) for s in unique):continue
        unique.append(candidate)
    return unique

def sample_sphere(count=2048):
    k=np.arange(count);z=1-2*(k+.5)/count;angle=k*np.pi*(3-np.sqrt(5))
    return np.c_[np.sqrt(1-z*z)*np.cos(angle),np.sqrt(1-z*z)*np.sin(angle),z]

def validate(candidate,views,images,C,Q,spacing):
    center=np.array(candidate['center']);radius=candidate['radius'];normal=sample_sphere();points=center+radius*normal
    support=np.zeros(len(points),int);free=np.zeros(len(points),int);records=[]
    for i,view in enumerate(views):
        delta,confidence,_=view.sample(points);front=((C[i]-points)*normal).sum(1)>0
        eligible=front&(confidence>.25);near=eligible&(abs(delta)<max(1.5*spacing,.15*radius));contradiction=eligible&(delta>max(2*spacing,.25*radius))
        support+=near;free+=contradiction
        v=center-C[i];distance=np.linalg.norm(v)
        if distance<=radius:continue
        axis=v/distance;reference=np.eye(3)[np.argmin(abs(axis))];u=np.cross(axis,reference);u/=np.linalg.norm(u);w=np.cross(axis,u)
        a=np.linspace(0,2*np.pi,180,endpoint=False)
        contour=center-axis*radius*radius/distance+radius*np.sqrt(1-radius**2/distance**2)*(np.cos(a)[:,None]*u+np.sin(a)[:,None]*w)
        edge=cv.Canny(cv.cvtColor(images[i],cv.COLOR_BGR2GRAY),40,100)
        h,width=edge.shape
        distance_map=cv.distanceTransform(255-edge,cv.DIST_L2,3) if edge.any() else np.full(edge.shape,max(h,width),dtype='f')
        uv=erp_uv(contour,C[i],Q[i])*[width,h]-.5
        error=cv.remap(distance_map,uv[:,0,None].astype('f'),uv[:,1,None].astype('f'),cv.INTER_LINEAR,borderMode=cv.BORDER_WRAP)[:,0]
        records.append(dict(source=i,depth_samples=int(eligible.sum()),near_fraction=float(near.sum()/max(1,eligible.sum())),free_fraction=float(contradiction.sum()/max(1,eligible.sum())),contour_near_edge_fraction=float((error<3).mean()),contour_median_px=float(np.median(error))))
    good_views=sum(r['near_fraction']>.60 and r['contour_near_edge_fraction']>.75 for r in records)
    multi_free=float(((free>=2)&(free>support)).mean())
    covered=float((support>=1).mean())
    accepted=good_views>=3 and multi_free<.10 and covered>.70
    return dict(accepted=bool(accepted),supporting_cameras=good_views,surface_depth_coverage=covered,multiple_camera_free_space_fraction=multi_free,per_camera=records,note='Depth votes are predicted evidence; contour-near-edge test is not an object segmentation oracle.')

def closed_sphere(center,radius,latitudes=32,longitudes=64):
    center=np.asarray(center);p=[[0,0,1]]
    for i in range(1,latitudes):
        theta=np.pi*i/latitudes
        for j in range(longitudes):
            phi=2*np.pi*j/longitudes;p.append([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    p.append([0,0,-1]);tri=[];bottom=len(p)-1
    for j in range(longitudes):tri.append([0,1+j,1+(j+1)%longitudes])
    for i in range(latitudes-2):
        for j in range(longitudes):
            a=1+i*longitudes+j;b=1+i*longitudes+(j+1)%longitudes;c=b+longitudes;d=a+longitudes
            tri.extend([[a,b,c],[a,c,d]])
    start=1+(latitudes-2)*longitudes
    for j in range(longitudes):tri.append([start+j,bottom,start+(j+1)%longitudes])
    p=np.asarray(p)*radius+center;tri=np.asarray(tri,dtype='i')
    normals=np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]])
    reverse=(normals*(p[tri].mean(1)-center)).sum(1)<0;tri[reverse]=tri[reverse,::-1]
    return p.astype('f'),tri

def run(source,mesh,out,warped=False):
    meta=load(source);cameras=np.load(source/'aligned-cameras.npz');C,Q=cameras['C'],cameras['Q']
    data=np.load(mesh);p=data['p'];tri=data['tri'];hints=data['owner_hint'] if 'owner_hint' in data else np.full(len(tri),-1,'i')
    spacing=float(np.median(np.linalg.norm(p[tri[:,1]]-p[tri[:,0]],axis=1)))
    candidates=propose_spheres(p,spacing);print('Geometry candidates',len(candidates),flush=True)
    images=[cv.resize(cv.imdecode(np.fromfile(s['photo'],np.uint8),1),(2048,1024)) for s in meta['stations']]
    views=fields(source,warped);required=np.zeros(len(tri),bool);report=[]
    for candidate in candidates:
        evidence=validate(candidate,views,images,C,Q,spacing);record=dict(**candidate,validation=evidence);report.append(record)
        print('Candidate',record,flush=True)
        if not evidence['accepted']:continue
        center=np.array(candidate['center']);radius=candidate['radius']
        remove=np.linalg.norm(p[tri].mean(1)-center,axis=1)<radius*1.15
        tri=tri[~remove];hints=hints[~remove];required=required[~remove]
        newp,newt=closed_sphere(center,radius);newt+=len(p);p=np.vstack([p,newp]);tri=np.vstack([tri,newt]);hints=np.r_[hints,np.full(len(newt),-1,'i')];required=np.r_[required,np.ones(len(newt),bool)]
        edge,counts=np.unique(np.sort(np.concatenate([newt[:,[0,1]],newt[:,[1,2]],newt[:,[2,0]]]),axis=1),axis=0,return_counts=True)
        record.update(removed_faces=int(remove.sum()),inserted_faces=len(newt),boundary_edges=int((counts==1).sum()),nonmanifold_edges=int((counts!=2).sum()))
    out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'surface.npz',p=p,tri=tri,owner_hint=hints,geometry_required=required)
    write_json(out/'sphere-repair.json',dict(candidates=report,accepted=sum(s['validation']['accepted'] for s in report),semantic_labels=False,manual_coordinates=False,spacing=spacing,input_mesh=str(mesh)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--mesh',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--warped',action='store_true');a=p.parse_args();run(a.source,a.mesh,a.out,a.warped)
