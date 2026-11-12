"""Refine detected quad edges against photo gradients before triangulation."""
import argparse,json
from pathlib import Path
import numpy as np,cv2 as cv
from core import load,prep,write_json
from planar_constraints import triangulate

def refine_quad(image,quad,radius=7):
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY).astype('f')/255
    gx=cv.Sobel(gray,cv.CV_32F,1,0,ksize=3);gy=cv.Sobel(gray,cv.CV_32F,0,1,ksize=3)
    lines=[];scores=[]
    for i in range(4):
        a,b=quad[i],quad[(i+1)%4];d=b-a;length=np.linalg.norm(d);d/=length;n=np.array([-d[1],d[0]])
        t=np.linspace(.08,.92,int(np.clip(length/2,32,240)));off=np.linspace(-radius,radius,4*radius+1)
        p=a+t[:,None,None]*(b-a)+off[None,:,None]*n
        x,y=p[:,:,0].astype('f'),p[:,:,1].astype('f')
        strength=abs(cv.remap(gx,x,y,cv.INTER_LINEAR)*n[0]+cv.remap(gy,x,y,cv.INTER_LINEAR)*n[1])
        # Prefer the expected edge when two nearby parallel contrasts compete.
        strength*=np.exp(-.5*(off/(radius*.8))**2)[None]
        best=strength.argmax(1);v=strength[np.arange(len(t)),best];selected=p[np.arange(len(t)),best]
        good=v>=max(.035,np.quantile(v,.35));scores.append(int(good.sum()))
        if good.sum()<20:return quad,False
        fit=cv.fitLine(selected[good].astype('f'),cv.DIST_HUBER,0,.01,.01).ravel();direction=fit[:2];normal=np.array([-direction[1],direction[0]])
        if abs(direction@d)<.995:return quad,False
        lines.append(np.r_[normal,-normal@fit[2:]])
    points=[]
    for i in range(4):
        cross=np.cross(lines[i-1],lines[i])
        if abs(cross[2])<1e-6:return quad,False
        points.append(cross[:2]/cross[2])
    points=np.array(points)
    if np.linalg.norm(points-quad,axis=1).max()>radius*1.6:return quad,False
    return points,True

def run(scene,observations,out):
    meta=load(scene);data=json.loads(observations.read_text(encoding='utf8'));c=np.load(scene/'aligned-cameras.npz');C,Q=c['C'],c['Q'];S=data['size'];updated=0;obs=data['observations']
    for i,item in enumerate(meta['stations']):
        image=cv.imdecode(np.fromfile(item['photo'],np.uint8),1)
        for k,b in enumerate(prep.FACES):
            candidates=[o for o in obs if o['camera']==i and o['face']==k]
            if not candidates:continue
            im=prep.render.photo_view(image,b,S)
            for o in candidates:
                q,ok=refine_quad(im,np.array(o['uv'],dtype=float))
                if not ok:continue
                rays=np.c_[(q+.5-S/2)/(S/2),np.ones(4)]@b.T@Q[i].T;rays/=np.linalg.norm(rays,axis=1)[:,None]
                o['uv']=q.tolist();o['rays']=rays.tolist();updated+=1
    for pair in data['pairs']:
        a,b=obs[pair['a']],obs[pair['b']]
        p,g,t,s,den=triangulate(C[a['camera']],np.array(a['rays']),C[b['camera']],np.array(b['rays'])[pair['order']])
        normal=np.linalg.svd(p-p.mean(0))[2][-1];pair['points']=p.tolist();pair['normal']=normal.tolist()
    data['boundary_refinement']=dict(updated=updated,total=len(obs),radius_pixels=7)
    write_json(out,data);print('Refined',updated,'of',len(obs),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.scene,a.observations,a.out)
