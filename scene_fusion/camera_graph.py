"""Build a connected route graph only from validated registration evidence.

This is an observation graph, not proof that the straight route is collision-free.
No capture order, ring closure or fixed number of stations is assumed.
"""
import argparse,json
from pathlib import Path
import numpy as np

def build_graph(centers,ids,registration,neighbors=2):
    C=np.asarray(centers,float);n=len(ids)
    if C.shape!=(n,3) or n<2 or not np.isfinite(C).all():raise ValueError('Invalid camera centers')
    if registration['stations']!=list(ids):raise ValueError('Registration station order differs')
    candidates=[];rejected=[]
    weights=registration.get('graph',{}).get('edge_weights',{})
    for i in range(n):
        for j in range(i+1,n):
            e=registration['edges'].get(f'{ids[i]}-{ids[j]}',{})
            w=float(weights.get(f'{i}-{j}',1))
            d=float(np.linalg.norm(C[i]-C[j]))
            if e.get('accepted') and e.get('global_held_inliers',0)>=4 and w>=.1 and np.isfinite(w) and d>1e-8:
                candidates.append((d,i,j))
            elif e.get('accepted'):rejected.append([i,j])
    parent=list(range(n))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    chosen=set()
    for d,i,j in sorted(candidates):
        if find(i)!=find(j):parent[find(i)]=find(j);chosen.add((i,j))
    if len({find(i) for i in range(n)})!=1:
        groups={}
        for i in range(n):groups.setdefault(find(i),[]).append(ids[i])
        raise ValueError('Disconnected validated camera graph; add overlapping captures: '+str(list(groups.values())))
    for i in range(n):
        nearby=sorted((d,a,b) for d,a,b in candidates if i in (a,b))
        chosen.update((a,b) for _,a,b in nearby[:neighbors])
    return dict(stations=list(ids),edges=[list(e) for e in sorted(chosen)],validated_edges=len(candidates),rejected_edges=rejected,
                method='validated match edges, spatial minimum spanning tree plus nearest supported neighbors',
                assumes_ring=False,collision_free_routes=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);a=p.parse_args()
    meta=json.loads((a.scene/'scene.json').read_text('utf8'));reg=json.loads((a.scene/'registration.json').read_text('utf8'))
    report=build_graph(np.load(a.scene/'aligned-cameras.npz')['C'],[s['id'] for s in meta['stations']],reg)
    (a.scene/'camera-graph.json').write_text(json.dumps(report,indent=2),encoding='utf8')
