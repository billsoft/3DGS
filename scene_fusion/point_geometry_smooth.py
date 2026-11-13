"""Bounded bilateral normal/point smoothing for reconstruction guides.

Does not touch source Gaussian colors or treat Gaussian centers as surface samples.
Input points must already be estimated surface samples with oriented normals.
"""
import numpy as np
from scipy.spatial import cKDTree

def smooth_oriented_points(points,normals,spacing,iterations=4,neighbors=24):
    if spacing<=0 or iterations<0 or neighbors<3:raise ValueError('Invalid point smoothing settings')
    p=np.asarray(points,dtype=float).copy();n=np.asarray(normals,dtype=float).copy();start=p.copy()
    if p.ndim!=2 or p.shape[1]!=3 or n.shape!=p.shape or len(p)<3:raise ValueError('At least three points and matching normals required')
    if not np.isfinite(p).all() or not np.isfinite(n).all():raise ValueError('Nonfinite geometry guide')
    if (np.linalg.norm(n,axis=1)<1e-6).any():raise ValueError('Normals must be nonzero')
    n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12);radius=2*spacing
    for _ in range(iterations):
        tree=cKDTree(p);q=p.copy();nn=n.copy()
        for lo in range(0,len(p),12000):
            hi=min(lo+12000,len(p));dist,ids=tree.query(p[lo:hi],k=min(neighbors,len(p)),workers=4)
            delta=p[ids]-p[lo:hi,None];cos=np.sum(n[ids]*n[lo:hi,None],axis=2)
            offset=np.sum(delta*n[lo:hi,None],axis=2)
            weight=np.exp(-.5*(dist/radius)**2-.5*(offset/spacing)**2)*np.maximum(cos,0)**4
            weight[(dist>3*radius)|(cos<.8)]=0;total=weight.sum(1)
            normal=np.sum(weight[:,:,None]*n[ids],axis=1);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
            move=np.sum(weight*np.sum(delta*normal[:,None],axis=2),axis=1)/np.maximum(total,1e-12)
            proposal=p[lo:hi]+.7*np.clip(move,-spacing,spacing)[:,None]*normal
            shift=proposal-start[lo:hi];length=np.linalg.norm(shift,axis=1)
            q[lo:hi]=start[lo:hi]+shift*np.minimum(1,spacing/np.maximum(length,1e-12))[:,None]
            nn[lo:hi]=normal
        p,n=q,nn
    return p,n,dict(iterations=iterations,neighbors=neighbors,spatial_sigma=radius,max_displacement=float(np.max(np.linalg.norm(p-start,axis=1))),displacement_cap=spacing)
