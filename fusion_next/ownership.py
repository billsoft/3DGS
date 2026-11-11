"""Five-label surface graph: four cameras + unsupported, optimized by alpha expansion.
Pair costs are a metric on sampled edge colors, so binary moves are submodular.
"""
from common import *
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow,breadth_first_order,dijkstra
from collider_audit import scene_for
def run():
    territorial=os.environ.get('GS_TERRITORY')=='1'
    geo=np.load(OUT/'geometry.npz');p,tri=geo['p'],geo['tri'];xyz=p[tri];center=xyz.mean(1);N=len(tri)
    normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]);normal/=np.maximum(1e-10,np.linalg.norm(normal,axis=1))[:,None]
    ims=photos((3840,1920));scores=[];colors=[]
    prior=np.load(OUT/'prior.npz');base_meta=json.loads((BASE/'mesh.json').read_text());supported_groups=np.array([cv.imread(str(BASE/g['texture'])).std()>1 for g in base_meta['groups']]);occluder_tri=tri[supported_groups[prior['group']]]
    scene,scene_mesh=scene_for(p,occluder_tri)
    for i,cam in enumerate(cams):
        err,conf,dep=cam.sample(cam.project(center));direction=C[i]-center;cosine=np.abs((direction*normal).sum(1))/np.maximum(.01,np.linalg.norm(direction,axis=1))
        tol=.075+.020*dep;ok=(conf>.13)&(err>-tol)&(err<tol*1.8)
        ray=center-C[i];length=np.linalg.norm(ray,axis=1);ray/=np.maximum(length[:,None],1e-8)
        hit=scene.run(np.ascontiguousarray(np.tile(C[i],(N,1)),'f'),np.ascontiguousarray(ray,'f'),query='DISTANCE')
        ok&=hit>length-.035
        score=conf*(.1+.9*cosine**1.5)*np.exp(-abs(err)/np.maximum(.06,.018*dep))/(.4+dep)**1.6
        scores.append(np.where(ok,score,0));colors.append(correct_rgb(sample(ims[i],uv(project(center,i)))[:,::-1]/255,i))
    scores=np.stack(scores,1);colors=np.stack(colors,1);best=scores.max(1);valid=best>1e-6
    unary=np.full((N,5),12.,dtype='f');unary[:,:4]=-np.log(np.maximum(scores,1e-9)/np.maximum(best[:,None],1e-9))
    unary[:,:4][scores<np.maximum(best[:,None]*.15,1e-7)]=30
    unary[:,4]=np.where(valid,4.5,0);owner=unary.argmin(1)
    a,b=adjacency(tri);dot=(normal[a]*normal[b]).sum(1);edge_weight=np.clip((dot-.65)/.35,0,1)*.8
    keep=edge_weight>.02;a=a[keep];b=b[keep];edge_weight=edge_weight[keep]
    if territorial:
        from territory import domains,fragmentation
        macro,macro_distance,allowed,position_prior=domains(center,C,scores,a,b)
        unary[:,:4]=np.minimum(unary[:,:4],3)+position_prior*5
        unary[:,:4][~allowed]=100
        unary[:,4]=np.where(valid,100,0)
        owner=unary.argmin(1);edge_weight*=4
    features=(colors[a]+colors[b])*.5
    if territorial:
        common_vertex=(tri[a,:,None]==tri[b,None,:]).any(2)
        order=np.argsort(~common_vertex,axis=1)[:,:2]
        endpoints=p[np.take_along_axis(tri[a],order,axis=1)]
        t=np.linspace(.05,.95,5)
        probes=endpoints[:,:1]*(1-t[None,:,None])+endpoints[:,1:]*t[None,:,None]
        features=np.stack([correct_rgb(sample(ims[i],uv(project(probes.reshape(-1,3),i)))[:,::-1]/255,i).reshape(len(a),5,3) for i in range(4)],axis=1)
    unknown=features.mean(1);features=np.concatenate([features,unknown[:,None]],1).astype('f');E=len(a);ei=np.arange(E)
    def pair(la,lb):
        delta=np.abs(features[ei,la]-features[ei,lb]).sum(-1)
        if territorial:delta=delta.max(1)
        return edge_weight*((la!=lb)*.25+delta*2.0)
    def energy(label):return float(unary[np.arange(N),label].sum()+pair(label[a],label[b]).sum())
    history=[energy(owner)];initial=owner.copy()
    for iteration in range(3):
      changed=0
      for alpha in range(5):
        old=owner;u0=unary[np.arange(N),old].astype('d');u1=unary[:,alpha].astype('d').copy();la=old[a];lb=old[b];aa=np.full(E,alpha)
        e00=pair(la,lb);e01=pair(la,aa);e10=pair(aa,lb);edge=np.maximum(0,(e01+e10-e00)*.5)
        np.add.at(u1,a,e10-e00-edge);np.add.at(u1,b,e01-e00-edge)
        minimum=np.minimum(u0,u1);u0-=minimum;u1-=minimum
        source=N;sink=N+1;ids=np.arange(N);row=np.r_[np.full(N,source),ids,a,b];col=np.r_[ids,np.full(N,sink),b,a]
        values=np.r_[u1,u0,edge,edge];capacity=coo_matrix((np.rint(values*1000).astype(np.int64),(row,col)),shape=(N+2,N+2)).tocsr()
        flow=maximum_flow(capacity,source,sink);residual=capacity-flow.flow;residual.data=(residual.data>0).astype('i');residual.eliminate_zeros()
        reachable=breadth_first_order(residual,source,directed=True,return_predecessors=False);stay=np.zeros(N+2,bool);stay[reachable]=True;candidate=old.copy();candidate[~stay[:N]]=alpha
        cost=energy(candidate)
        if cost<history[-1]-1e-4:
            changed+=int((candidate!=owner).sum());owner=candidate;history.append(cost)
        print('ownership',iteration,alpha,'energy',history[-1],flush=True)
      if not changed:break
    # Surface-geodesic distance to ownership seams; no proximity links through air.
    seam=owner[a]!=owner[b];length=np.linalg.norm(center[a]-center[b],axis=1)
    graph=coo_matrix((np.r_[length,length],(np.r_[a,b],np.r_[b,a])),shape=(N,N)).tocsr()
    boundary=np.unique(np.r_[a[seam],b[seam]])
    distance=dijkstra(graph,directed=False,indices=boundary,min_only=True,limit=.07) if len(boundary) else np.full(N,np.inf)
    partner=np.full(N,-1,'i')
    for left,right in zip(a[seam],b[seam]):
        if owner[left]<4 and owner[right]<4:partner[left]=owner[right];partner[right]=owner[left]
    # Propagate only within the same owned surface, for a bounded four-ring band.
    for _ in range(4):
        for left,right in [(a,b),(b,a)]:
            good=(partner[left]<0)&(partner[right]>=0)&(owner[left]==owner[right])&(distance[left]<.055)
            partner[left[good]]=partner[right[good]]
    np.savez_compressed(OUT/'ownership.npz',owner=owner,partner=partner,distance=distance,scores=scores,colors=colors,center=center)
    if territorial:
        np.savez_compressed(OUT/'territory.npz',macro=macro,macro_distance=macro_distance,allowed=allowed,owner=owner)
        baseline=np.load(ROOT/'four-room/baselines/2026-11-11-fusion-next/ownership.npz')['owner']
        write_json('territory-report.json',dict(method='quality-weighted camera Voronoi core + surface quality smoothing + multi-sample seam graph optimization',quality_prior_weight=.65,seam_edge_samples=5,seam_search_band_model_units=.22,overlap_band_model_units=.055,core_changes=int(((macro_distance>=.22)&(macro!=owner)).sum()),changed_from_baseline=int((baseline!=owner).sum()),baseline_components=fragmentation(baseline,a,b),new_components=fragmentation(owner,a,b),note='Surface connectivity depends on input geometry. Camera cores use calibrated optical centers, not Gaussian cloud centroids.'))
    write_json('ownership-report.json',dict(method='metric color-seam alpha expansion / integer max-flow',initial_energy=history[0],final_energy=history[-1],accepted_moves=len(history)-1,changed_faces=int((owner!=initial).sum()),unsupported_faces=int((owner==4).sum()),seam_edges=int(seam.sum()),boundary_band_faces=int(((distance<.055)&(partner>=0)).sum()),history=history))
if __name__=='__main__':run()
