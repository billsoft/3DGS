"""Generate photo constraints and camera-graph midpoint guards for any scene count."""
import argparse,json
from pathlib import Path
import numpy as np
import cv2 as cv
from core import load,prep,write_json,viewmat,look

def run(out,size,metadata_only=False,evidence_graph=False):
    scene=load(out);c=np.load(out/'aligned-cameras.npz');C,Q=c['C'],c['Q'];N=len(C)
    folder=out/f'data-{size}';folder.mkdir(exist_ok=True);views=[];targets=[]
    yy,xx=np.indices((size,size));rays=np.stack([(xx+.5-size/2)/(size/2),(yy+.5-size/2)/(size/2),np.ones_like(xx)],-1)
    def add(name,center,basis,source,photo=None,role='train'):
        views.append(dict(name=name,center=center.tolist(),basis=basis.tolist(),viewmat=viewmat(center,basis).tolist(),source=source,role=role))
        if metadata_only:return
        if source is None:targets.append(np.zeros((size,size,3),'u1'));return
        local=rays@basis.T@Q[source];lon=np.arctan2(-local[:,:,1],local[:,:,0]);lat=np.arctan2(np.linalg.norm(local[:,:,:2],axis=2),local[:,:,2])
        tx=((lon/(2*np.pi)+.5)*photo.shape[1]-.5).astype('f');ty=(lat/np.pi*photo.shape[0]-.5).astype('f')
        targets.append(cv.remap(photo,tx,ty,cv.INTER_LINEAR,borderMode=cv.BORDER_WRAP)[:,:,::-1])
    for i,item in enumerate(scene['stations']):
        photo=None if metadata_only else cv.imdecode(np.fromfile(item['photo'],np.uint8),1)
        for k,b in enumerate(prep.FACES):add(f"source-{item['id']}-{k}",C[i],Q[i]@b,i,photo)
    # Use nearest spatial neighbors, independent of filenames or presumed ring order.
    edges=set()
    for i in range(N):
        for j in np.argsort(np.linalg.norm(C-C[i],axis=1))[1:min(3,N)]:edges.add(tuple(sorted((i,int(j)))))
    if evidence_graph:
        from camera_graph import build_graph
        graph=build_graph(C,[s['id'] for s in scene['stations']],json.loads((out/'registration.json').read_text('utf8')))
        write_json(out/'camera-graph.json',graph);edges={tuple(e) for e in graph['edges']}
    up=Q[:,:,2].mean(0);up/=np.linalg.norm(up);target=C.mean(0)-up*.6
    for i,j in sorted(edges):
        center=(C[i]+C[j])/2
        for label,basis in [('room',look(center,target,up)),('up',Q[i]@prep.FACES[4]),('down',Q[i]@prep.FACES[5])]:
            add(f'mid-{i}-{j}-{label}',center,basis,None,role='novel')
    if not metadata_only:
        np.save(folder/'targets.npy',np.stack(targets));np.save(folder/'initial.npy',np.load(out/'initial.npy'))
    write_json(folder/'dataset.json',dict(size=size,views=views,centers=C.tolist(),up=up.tolist(),stations=[s['id'] for s in scene['stations']],midpoint_edges=sorted(edges)))
    print('Dataset',N,'cameras',len(views),'views',size,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--size',type=int,default=512)
    p.add_argument('--metadata-only',action='store_true');p.add_argument('--evidence-graph',action='store_true')
    a=p.parse_args();run(a.out,a.size,a.metadata_only,a.evidence_graph)
