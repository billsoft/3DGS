"""Remove small fragmented photo-label islands without overriding visibility.

Operates on mesh connectivity, preserves coordinates and samples original photos.
No color inpainting and no propagation through a known occluder.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def reject_photo_outliers(scores,colors,coverage):
    """Reject gross projection colors only with a clear multi-photo consensus.

    Ambiguous two-view evidence or competing equal clusters stays unchanged.
    Visibility is never expanded; this only removes inconsistent candidates.
    """
    result=scores.copy();removed=0;changed=0
    for lo in range(0,len(scores),24000):
        hi=min(lo+24000,len(scores));s=scores[lo:hi];c=colors[lo:hi]
        valid=(s>np.maximum(s.max(1)[:,None]*.02,1e-10))&(coverage[lo:hi]>=.75)
        distance=np.linalg.norm(c[:,:,None]-c[:,None,:],axis=3)
        support=((distance<.10)&valid[:,None,:]).sum(2);support[~valid]=0
        best=support.argmax(1);count=support[np.arange(len(s)),best]
        confident=(count>=3)&(count>=.6*valid.sum(1))
        reject=confident[:,None]&(distance[np.arange(len(s)),best]>.18)&(s>0)
        result[lo:hi][reject]=0;removed+=int(reject.sum());changed+=int(reject.any(1).sum())
    return result,dict(rejected_candidates=removed,affected_faces=changed,min_consensus_views=3,
        cluster_color_radius=.10,outlier_color_radius=.18,visibility_expanded=False)


def coherent_patches(tri,owner,scores,colors,good,neighbors,area,spacing,max_faces=96):
    owner=owner.copy();original=owner.copy();changed=0;patches=0
    face=np.arange(len(tri));a=np.repeat(face,3);b=neighbors.ravel()
    for _ in range(3):
        same=(owner[a]==owner[b])&good[a]&good[b]
        graph=coo_matrix((np.ones(same.sum(),dtype=np.int8),(a[same],b[same])),shape=(len(tri),len(tri))).tocsr()
        count,labels=connected_components(graph,directed=False)
        sizes=np.bincount(labels);areas=np.bincount(labels,weights=area)
        order=np.argsort(labels);offset=np.r_[0,np.cumsum(sizes)]
        for label in np.flatnonzero((sizes<=max_faces)&(areas<48*spacing**2)):
            ids=order[offset[label]:offset[label+1]]
            if not good[ids].all():continue
            ring=np.unique(neighbors[ids]);ring=ring[(labels[ring]!=label)&good[ring]]
            if len(ring)<3:continue
            options,votes=np.unique(owner[ring],return_counts=True);target=options[votes.argmax()]
            if votes.max()<.8*len(ring) or target==owner[ids[0]]:continue
            # Every face must already have direct visibility support from this photo.
            best=scores[ids].max(1);candidate=scores[ids,target]
            if np.any(candidate<np.maximum(best*.015,1e-10)):continue
            # Avoid transferring a patch to a photograph of a different object.
            difference=np.linalg.norm(colors[ids,target]-colors[ids,owner[ids]],axis=1)
            if np.quantile(difference,.9)>.18:continue
            owner[ids]=target;changed+=len(ids);patches+=1
    return owner,dict(method='bounded mesh-connected photo-label consensus with visibility and color gates',
        reassigned_faces=int((owner!=original).sum()),patch_updates=patches,max_patch_faces=max_faces)
