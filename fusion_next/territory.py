"""Camera-position domains on visible surface graphs; no object-specific rules."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra,connected_components

def ray_plane_jacobian(ray,normal,ratio):
    denominator=np.sum(ray*normal,axis=1)
    if np.any(abs(denominator)<1e-6):raise ValueError('Grazing ray cannot be projected safely')
    return ratio[:,None,None]*(np.eye(3)[None]-ray[:,:,None]*normal[:,None,:]/denominator[:,None,None])

def domains(center,cameras,scores,a,b,band=.22,quality_weight=.65):
    delta=center[:,None,:]-cameras[None,:,:]
    squared=np.sum(delta*delta,axis=2)
    eligible=scores>np.maximum(scores.max(1)[:,None]*.08,1e-7)
    spacing=np.linalg.norm(cameras[:,None]-cameras[None,:],axis=2)
    scale=max(float(np.median(spacing[spacing>0])**2),1e-6)
    quality=-np.log(np.maximum(scores,1e-9)/np.maximum(scores.max(1)[:,None],1e-9))
    original_quality=np.minimum(quality,5)
    length=np.linalg.norm(center[a]-center[b],axis=1)
    weight=np.exp(-(length/.12)**2)
    W=coo_matrix((np.r_[weight,weight],(np.r_[a,b],np.r_[b,a])),shape=(len(center),len(center))).tocsr()
    degree=W@eligible.astype(float)
    quality=original_quality.copy()
    for _ in range(3):
        local=(W@(quality*eligible))/np.maximum(degree,1e-9)
        local=np.where(degree>1e-9,local,quality)
        quality=(2*original_quality+local)/3
    cost=np.where(eligible,squared/scale+quality_weight*quality,np.inf)
    macro=cost.argmin(1);macro[~eligible.any(1)]=4
    graph=coo_matrix((np.r_[length,length],(np.r_[a,b],np.r_[b,a])),shape=(len(center),len(center))).tocsr()
    boundary=np.unique(np.r_[a[macro[a]!=macro[b]],b[macro[a]!=macro[b]]])
    distance=dijkstra(graph,directed=False,indices=boundary,min_only=True,limit=band*1.5) if len(boundary) else np.full(len(center),np.inf)
    # Neighbor labels may compete only in the surface-geodesic boundary band.
    allowed=np.zeros_like(scores,dtype=bool)
    good=macro<4;allowed[np.flatnonzero(good),macro[good]]=True
    for _ in range(8):
        previous=allowed.copy()
        for left,right in [(a,b),(b,a)]:
            near=distance[left]<band
            np.logical_or.at(allowed,left[near],previous[right[near]])
        allowed&=eligible
    allowed[distance>=band]=False;allowed[np.flatnonzero(good),macro[good]]=True
    nearest=np.min(np.where(eligible,squared,np.inf),axis=1)
    prior=np.maximum(0,squared-nearest[:,None]);prior[~np.isfinite(prior)]=0
    return macro,distance,allowed,prior/max(scale,1e-6)

def fragmentation(owner,a,b):
    same=owner[a]==owner[b];a,b=a[same],b[same]
    graph=coo_matrix((np.ones(len(a)*2),(np.r_[a,b],np.r_[b,a])),shape=(len(owner),len(owner))).tocsr()
    _,label=connected_components(graph,directed=False)
    result=[]
    for source in range(4):
        _,counts=np.unique(label[owner==source],return_counts=True)
        result.append(dict(source=source,components_ge20=int((counts>=20).sum()),faces=int(counts.sum()),largest=int(counts.max()) if len(counts) else 0))
    return result
