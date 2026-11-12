"""Multi-view bounded-plane evidence from image quadrilaterals, without labels."""
import argparse,json
from pathlib import Path
import numpy as np,cv2 as cv
from core import load,prep,write_json

def line_quads(gray):
    detected=cv.createLineSegmentDetector().detect(gray)[0]
    if detected is None:return []
    segments=[]
    for line in detected[:,0]:
        p,q=line[:2],line[2:];length=np.linalg.norm(q-p)
        if length<35:continue
        direction=(q-p)/length
        if direction[0]<0:direction=-direction
        merged=False
        for k,(a,b) in enumerate(segments):
            v=(b-a)/np.linalg.norm(b-a)
            if abs(v@direction)<.995 or max(abs(np.cross(v,p-a)),abs(np.cross(v,q-a)))>4:continue
            coords=np.array([0,np.linalg.norm(b-a),(p-a)@v,(q-a)@v])
            if min(coords[2:])>coords[1]+45 or max(coords[2:])<-45:continue
            segments[k]=(a+v*coords.min(),a+v*coords.max());merged=True;break
        if not merged:segments.append((p,q))
    segments=sorted(segments,key=lambda x:np.linalg.norm(x[1]-x[0]),reverse=True)
    horizontal=[s for s in segments if abs((s[1]-s[0])[0])>abs((s[1]-s[0])[1])][:22]
    vertical=[s for s in segments if abs((s[1]-s[0])[0])<=abs((s[1]-s[0])[1])][:22]
    if len(horizontal)<2 or len(vertical)<2:return []
    def lines(items):
        a=np.array([x[0] for x in items]);b=np.array([x[1] for x in items]);v=b-a;n=np.c_[-v[:,1],v[:,0]];n/=np.linalg.norm(n,axis=1)[:,None];return np.c_[n,-(n*a).sum(1)]
    H,V=lines(horizontal),lines(vertical);cross=np.cross(H[:,None],V[None]);inter=cross[:,:,:2]/np.where(abs(cross[:,:,2:])<1e-6,np.nan,cross[:,:,2:])
    from itertools import combinations
    hp=np.array(list(combinations(range(len(H)),2)));vp=np.array(list(combinations(range(len(V)),2)))
    hp=hp[abs((H[hp[:,0],:2]*H[hp[:,1],:2]).sum(1))>.85];vp=vp[abs((V[vp[:,0],:2]*V[vp[:,1],:2]).sum(1))>.85]
    results=[];S=gray.shape[0]
    for h1,h2 in hp:
        q=np.stack([inter[h1,vp[:,0]],inter[h1,vp[:,1]],inter[h2,vp[:,1]],inter[h2,vp[:,0]]],axis=1)
        valid=np.isfinite(q).all(axis=(1,2))&(q>6).all(axis=(1,2))&(q<S-7).all(axis=(1,2))
        edges=np.roll(q,-1,axis=1)-q;length=np.linalg.norm(edges,axis=2);valid&=(length.min(1)>30)&(abs(np.cross(q[:,1]-q[:,0],q[:,3]-q[:,0]))>3000)
        for index in np.flatnonzero(valid):
            quad=q[index].astype('f')
            if not cv.isContourConvex(quad):continue
            v1,v2=vp[index];supports=[horizontal[h1],vertical[v2],horizontal[h2],vertical[v1]];ok=True
            for k,(a,b) in enumerate(supports):
                start,end=quad[k],quad[(k+1)%4];direction=(end-start)/np.linalg.norm(end-start);length=np.linalg.norm(end-start);t=sorted([(a-start)@direction,(b-start)@direction]);overlap=max(0,min(length,t[1])-max(0,t[0]))
                if overlap<length*.4:ok=False;break
            if ok:results.append(quad)
    return results

def detect(im):
    g=cv.cvtColor(im,cv.COLOR_BGR2GRAY);g=cv.GaussianBlur(g,(3,3),.6);masks=[cv.Canny(g,40,100)]
    masks.extend(cv.threshold(g,t,255,cv.THRESH_BINARY)[1] for t in [32,64,96,128,160,192,224])
    candidates=[]
    for mask in masks:
        contours,_=cv.findContours(mask,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            perimeter=cv.arcLength(contour,True)
            if perimeter<160:continue
            for eps in [.01,.02,.035]:
                q=cv.approxPolyDP(contour,eps*perimeter,True).reshape(-1,2)
                if len(q)!=4 or not cv.isContourConvex(q) or cv.contourArea(q)<3000:continue
                if np.any(q<6) or np.any(q>im.shape[0]-7):continue
                q=q.astype('f');edges=np.roll(q,-1,axis=0)-q;l=np.linalg.norm(edges,axis=1)
                if l.min()<25 or l.max()/l.min()>15:continue
                if np.max(abs(np.sum(edges*np.roll(edges,1,axis=0),axis=1)/(l*np.roll(l,1))))>.95:continue
                center=q.mean(0)
                if any(np.linalg.norm(center-v.mean(0))<8 and abs(cv.contourArea(q)/cv.contourArea(v)-1)<.08 for v in candidates):continue
                candidates.append(q)
    for q in line_quads(g):
        center=q.mean(0)
        if not any(np.linalg.norm(center-v.mean(0))<8 and abs(cv.contourArea(q)/cv.contourArea(v)-1)<.08 for v in candidates):candidates.append(q)
    return sorted(candidates,key=lambda q:cv.contourArea(q),reverse=True)[:70]

def triangulate(C1,r1,C2,r2):
    dot=(r1*r2).sum(1);baseline=C2-C1;den=1-dot**2
    t=(r1@baseline-dot*(r2@baseline))/np.maximum(den,1e-8);s=(dot*(r1@baseline)-r2@baseline)/np.maximum(den,1e-8)
    a=C1+r1*t[:,None];b=C2+r2*s[:,None]
    return (a+b)/2,np.linalg.norm(a-b,axis=1),t,s,den

def run(source,out):
    out.mkdir(parents=True,exist_ok=True);meta=load(source);cam=np.load(source/'aligned-cameras.npz');C,Q=cam['C'],cam['Q'];records=[];S=1024
    for i,item in enumerate(meta['stations']):
        image=cv.imdecode(np.fromfile(item['photo'],np.uint8),1)
        for k,b in enumerate(prep.FACES):
            im=prep.render.photo_view(image,b,S);quads=detect(im);canvas=im.copy()
            for q in quads:
                rays=np.c_[(q+.5-S/2)/(S/2),np.ones(4)]@b.T@Q[i].T;rays/=np.linalg.norm(rays,axis=1)[:,None]
                records.append(dict(camera=i,face=k,uv=q.tolist(),rays=rays.tolist()))
                cv.polylines(canvas,[q.astype('i')],True,(0,0,255),2);cv.putText(canvas,str(len(records)-1),tuple(q[0].astype(int)),cv.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)
            cv.imwrite(str(out/f'detect-{i}-{k}.jpg'),canvas);print('Quadrilaterals',i,k,len(quads),flush=True)
    pairs=[]
    for ai,a in enumerate(records):
        i=a['camera'];r1=np.array(a['rays'])
        for bi in range(ai+1,len(records)):
            b=records[bi];j=b['camera']
            if i==j:continue
            r=np.array(b['rays'])
            if np.dot(r1.mean(0),r.mean(0))<.25:continue
            best=None
            for reverse in [False,True]:
                for shift in range(4):
                    order=np.roll(np.arange(4)[::-1] if reverse else np.arange(4),shift);r2=r[order]
                    epi=np.cross(r1,C[j]-C[i]);epi/=np.maximum(np.linalg.norm(epi,axis=1)[:,None],1e-8)
                    if np.max(abs((epi*r2).sum(1)))*S/2>5:continue
                    points,gap,t,s,den=triangulate(C[i],r1,C[j],r2)
                    if np.min(t)<.3 or np.min(s)<.3 or np.max(t)>25 or np.max(s)>25 or np.min(den)<.005:continue
                    error=gap/np.minimum(t,s)*S/2
                    if np.max(error)>5 or np.median(error)>2:continue
                    center=points.mean(0);_,sv,V=np.linalg.svd(points-center);normal=V[-1];planarity=sv[-1]/max(sv[1],1e-8)
                    e=np.roll(points,-1,axis=0)-points;length=np.linalg.norm(e,axis=1);angle=abs((e*np.roll(e,1,axis=0)).sum(1)/(length*np.roll(length,1)))
                    if planarity>.025 or max(angle)>.18 or length.min()<.15:continue
                    opposite=max(abs(length[0]/length[2]-1),abs(length[1]/length[3]-1))
                    if opposite>.15:continue
                    score=float(np.mean(error)+planarity*30+np.mean(angle)*3)
                    if best is None or score<best['score']:best=dict(a=ai,b=bi,order=order.tolist(),points=points.tolist(),normal=normal.tolist(),score=score,reprojection_px=error.tolist(),planarity=float(planarity))
            if best:pairs.append(best)
    write_json(out/'quads.json',dict(observations=records,pairs=pairs,size=S));print('Total quads',len(records),'verified pairs',len(pairs),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.source,a.out)
