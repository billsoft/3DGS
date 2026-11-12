"""Conservative small-hole closure and optional image-verified planar patches.

Bounded planes are explicit measured constraints, not semantic lamp/TV rules.
Provisional constraints remain marked experimental and require an explicit flag.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
import cv2 as cv
from core import write_json

def inside_quad(xy,polygon,margin=0):
    edge=np.roll(polygon,-1,axis=0)-polygon
    delta=xy[:,None]-polygon[None]
    signed=(edge[None,:,0]*delta[:,:,1]-edge[None,:,1]*delta[:,:,0])/np.linalg.norm(edge,axis=1)[None]
    return (signed>=-margin).all(1)|(signed<=margin).all(1)

def close_small_holes(p,tri,spacing):
    edge,count=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0,return_counts=True)
    border=edge[count==1]; graph={}
    for a,b in border:graph.setdefault(a,[]).append(b);graph.setdefault(b,[]).append(a)
    order=np.argsort(tri.ravel());offset=np.r_[0,np.cumsum(np.bincount(tri.ravel(),minlength=len(p)))];incident=order//3
    visited=set(); added=[]; closed=0
    for start in graph:
        if start in visited:continue
        stack=[start];component=[]
        while stack:
            v=stack.pop()
            if v in visited:continue
            visited.add(v);component.append(v);stack.extend(n for n in graph[v] if n not in visited)
        if not 3<=len(component)<=24 or any(len(graph[v])!=2 for v in component):continue
        loop=[start];prev=None;current=start
        while True:
            nxt=next(v for v in graph[current] if v!=prev)
            if nxt==start:break
            loop.append(nxt);prev,current=current,nxt
        points=p[loop];center=points.mean(0)
        if np.linalg.norm(np.ptp(points,axis=0))>spacing*4:continue
        _,_,axes=np.linalg.svd(points-center)
        if np.max(np.abs((points-center)@axes[2]))>spacing*.12:continue
        xy=(points-center)@axes[:2].T
        if len(loop)==3:faces=np.array([[0,1,2]])
        else:
            try:faces=Delaunay(xy).simplices
            except Exception:continue
        contour=xy.astype('f')
        neighbors=np.unique(np.concatenate([incident[offset[v]:offset[v+1]] for v in loop]))
        adjacent=(p[tri[neighbors]]-center)@axes[:2].T
        reference=np.cross(p[tri[neighbors,1]]-p[tri[neighbors,0]],p[tri[neighbors,2]]-p[tri[neighbors,0]]).sum(0)
        before=len(added)
        for face in faces:
            test=[xy[face].mean(0)]+[(xy[face[k]]+xy[face[(k+1)%3]])/2 for k in range(3)]
            if not all(cv.pointPolygonTest(contour,tuple(map(float,q)),False)>=0 for q in test):continue
            # An exterior boundary of a tiny component is not a hole. Avoid
            # duplicate caps over triangles already occupying the polygon.
            point=tuple(map(float,test[0]))
            if any(abs(cv.contourArea(t.astype('f')))>1e-12 and cv.pointPolygonTest(t.astype('f'),point,False)>=0 for t in adjacent):continue
            indices=np.array(loop)[face]
            n=np.cross(p[indices[1]]-p[indices[0]],p[indices[2]]-p[indices[0]])
            if n@reference<0:indices=indices[::-1]
            added.append(indices)
        closed+=len(added)>before
    return np.asarray(added,dtype='i').reshape(-1,3),closed

def run(mesh,source,out,planes=None,allow_provisional=False):
    data=np.load(mesh);p=data['p'].copy();tri=data['tri'].copy()
    spacing=float(np.median(np.linalg.norm(p[tri[:,1]]-p[tri[:,0]],axis=1)))
    area=np.linalg.norm(np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]]),axis=1)
    tri=tri[area>1e-10];hints=np.full(len(tri),-1,'i');report=[];provisional=False
    if planes:
        constraints=json.loads(planes.read_text(encoding='utf8'));provisional=constraints['held_mean_limit']>3.5 or constraints['held_max_limit']>6
        if provisional and not allow_provisional:raise ValueError('Explicit --allow-provisional required for experimental plane constraints')
        C=np.load(source/'aligned-cameras.npz')['C']
        for plane in constraints['planes']:
            if len(set(plane['cameras']))<3 or plane['held_mean_px']>constraints['held_mean_limit'] or plane['held_max_px']>constraints['held_max_limit']:raise ValueError('Plane validation failed')
            corners=np.asarray(plane['corners']);normal=np.asarray(plane['normal']);normal/=np.linalg.norm(normal)
            u=corners[1]-corners[0];u/=np.linalg.norm(u);v=np.cross(normal,u);basis=np.stack([u,v],axis=1)
            polygon=(corners-corners[0])@basis;center=p[tri].mean(1)
            remove=inside_quad((center-corners[0])@basis,polygon,spacing*.8)&(abs((center-corners[0])@normal)<spacing*5)
            tri=tri[~remove];hints=hints[~remove]
            # Subdivide the bounded quad so equirectangular UV interpolation remains local.
            nu=max(2,int(np.ceil(np.linalg.norm(corners[1]-corners[0])/spacing)));nv=max(2,int(np.ceil(np.linalg.norm(corners[3]-corners[0])/spacing)))
            x,y=np.meshgrid(np.linspace(0,1,nu+1),np.linspace(0,1,nv+1));x=x.ravel();y=y.ravel()
            patch=(1-x[:,None])*(1-y[:,None])*corners[0]+x[:,None]*(1-y[:,None])*corners[1]+x[:,None]*y[:,None]*corners[2]+(1-x[:,None])*y[:,None]*corners[3]
            idx=np.arange((nu+1)*(nv+1)).reshape(nv+1,nu+1);a=idx[:-1,:-1].ravel();b=idx[:-1,1:].ravel();c=idx[1:,1:].ravel();d=idx[1:,:-1].ravel();faces=np.vstack([np.c_[a,b,c],np.c_[a,c,d]])+len(p)
            delta=C[plane['cameras']]-corners.mean(0);dist=np.linalg.norm(delta,axis=1);scores=abs(delta@normal)/np.maximum(dist,.1)**3;owner=plane['cameras'][int(scores.argmax())]
            p=np.vstack([p,patch]).astype('f');tri=np.vstack([tri,faces]).astype('i');hints=np.r_[hints,np.full(len(faces),owner,'i')]
            report.append(dict(removed_faces=int(remove.sum()),inserted_faces=len(faces),owner=owner,held_mean_px=plane['held_mean_px'],held_max_px=plane['held_max_px']))
    added,closed=close_small_holes(p,tri,spacing);tri=np.vstack([tri,added]);hints=np.r_[hints,np.full(len(added),-1,'i')]
    out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'surface.npz',p=p,tri=tri,owner_hint=hints)
    result=dict(input_mesh=str(mesh),vertices=len(p),triangles=len(tri),spacing=spacing,small_loops_closed=closed,hole_faces_added=len(added),plane_patches=report,provisional_constraints=provisional,constraints=str(planes) if planes else None)
    write_json(out/'regularize.json',result);print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mesh',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--planes',type=Path);p.add_argument('--allow-provisional',action='store_true');a=p.parse_args();run(a.mesh,a.source,a.out,a.planes,a.allow_provisional)
