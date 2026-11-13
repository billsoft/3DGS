"""Visibility-constrained Potts photo assignment on the mesh dual graph.

Uses optional PyMaxflow; independently implements alpha expansion, not mapMAP.
Inspired by MVS-Texturing's mesh-face data costs and coherent photo patches.
"""
import numpy as np

def energy(labels,unary,a,b,w):
    return float(unary[np.arange(len(labels)),labels].sum()+np.sum(w*(labels[a]!=labels[b])))

def expand(labels,unary,a,b,w,alpha):
    import maxflow
    n=len(labels);ids=np.arange(n)
    d0=unary[ids,labels].copy();d1=unary[:,alpha].copy()
    e00=w*(labels[a]!=labels[b]);e01=w*(labels[a]!=alpha);e10=w*(labels[b]!=alpha)
    pair=np.maximum(0,(e01+e10-e00)*.5)
    np.add.at(d1,a,e10-e00-pair);np.add.at(d1,b,e01-e00-pair)
    shift=np.minimum(d0,d1);d0-=shift;d1-=shift
    graph=maxflow.Graph[float](n,len(a));nodes=graph.add_nodes(n)
    graph.add_edges(a.astype(np.int64),b.astype(np.int64),pair,pair)
    graph.add_grid_tedges(nodes,d1,d0);graph.maxflow()
    result=labels.copy();result[graph.get_grid_segments(nodes)]=alpha
    return result

def optimize(unary,a,b,w,initial=None,cycles=3):
    labels=np.argmin(unary,axis=1) if initial is None else initial.copy()
    history=[energy(labels,unary,a,b,w)]
    for _ in range(cycles):
        old=history[-1]
        for alpha in range(unary.shape[1]):
            trial=expand(labels,unary,a,b,w,alpha);cost=energy(trial,unary,a,b,w)
            if cost<history[-1]-1e-7:labels=trial;history.append(cost)
        if old-history[-1]<1e-6:break
    return labels,history

def assign(p,tri,scores,good,initial,spacing,strength=2.):
    edge=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1)
    face=np.tile(np.arange(len(tri)),3);order=np.lexsort((edge[:,1],edge[:,0]))
    same=(edge[order[:-1]]==edge[order[1:]]).all(1)
    left,right=order[:-1][same],order[1:][same];a,b=face[left],face[right]
    valid=good[a]&good[b];a,b,left=a[valid],b[valid],left[valid]
    length=np.linalg.norm(p[edge[left,0]]-p[edge[left,1]],axis=1)
    w=strength*np.clip(length/spacing,.05,4.)
    area=np.linalg.norm(np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]]),axis=1)*.5
    unary=-np.log(np.maximum(scores,1e-30)/np.maximum(scores.max(1)[:,None],1e-30))
    unary*=np.clip(area/spacing**2,.1,5)[:,None]
    allowed=scores>1e-10;unary[~allowed]=1e8
    unary[~good]=1e8;unary[np.flatnonzero(~good),initial[~good]]=0
    owner,history=optimize(unary,a,b,w,initial)
    if np.any(good&~allowed[np.arange(len(tri)),owner]):raise ValueError('Graph cut violated visibility')
    report=dict(method='visibility-constrained alpha-expansion Potts assignment',strength=strength,
        initial_energy=history[0],final_energy=history[-1],accepted_moves=len(history)-1,
        reassigned_faces=int((owner!=initial).sum()),boundary_edges_before=int((initial[a]!=initial[b]).sum()),
        boundary_edges_after=int((owner[a]!=owner[b]).sum()),visibility_overrides=0)
    return owner,report
