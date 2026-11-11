"""Depth-warped bidirectional DIS flow converted to multi-camera 3D constraints."""
from common import *
def run():
    ims=photos((2048,1024));S=384;y,x=np.indices((S,S),dtype='f');grid=np.stack([x,y],-1)
    local=np.stack([(x+.5-S/2)/(S/2),(y+.5-S/2)/(S/2),np.ones_like(x)],-1)
    engine=cv.DISOpticalFlow_create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    chunks=[];reports=[]
    for i,n in enumerate(stations):
      for face,b in enumerate(rm.f.prep.FACES):
        z=np.load(rm.room/n/f'depth-{face}.npz');depth=cv.resize(z['depth'],(S,S));alpha=cv.resize(z['alpha'],(S,S))
        r=local@b.T;rw=r@Q[i].T;unit=rw/np.linalg.norm(rw,axis=2)[:,:,None]
        initial=C[i]+rw*depth[:,:,None]*scales[i]
        reference=sample(ims[i],uv(r.reshape(-1,3))).reshape(S,S,3);gray=cv.cvtColor(reference,cv.COLOR_BGR2GRAY)
        grad=cv.magnitude(cv.Sobel(gray,cv.CV_32F,1,0),cv.Sobel(gray,cv.CV_32F,0,1))
        span=cv.dilate(depth,np.ones((3,3),'u1'))-cv.erode(depth,np.ones((3,3),'u1'))
        source_valid=(alpha>.97)&(span<.055*np.maximum(depth,.1))&(depth>.2)&(gray<248)&(grad>18)
        for j in range(i+1,4):
          q=project(initial.reshape(-1,3),j);target_ray=q/np.maximum(1e-8,np.linalg.norm(q,axis=1))[:,None]
          warped=sample(ims[j],uv(q)).reshape(S,S,3);g=cv.cvtColor(warped,cv.COLOR_BGR2GRAY)
          err,conf,dep=cams[j].sample(cams[j].project(initial.reshape(-1,3)));visible=((conf>.2)&(abs(err)<.10+.04*dep)).reshape(S,S)
          forward=engine.calc(gray,g,None);backward=engine.calc(g,gray,None);dst=grid+forward
          def remap(a):return cv.remap(a,dst[:,:,0],dst[:,:,1],cv.INTER_LINEAR,borderMode=cv.BORDER_CONSTANT)
          fb=np.linalg.norm(forward+remap(backward),axis=2);photometric=abs(gray.astype('f')-remap(g).astype('f'))
          v=remap(target_ray.reshape(S,S,3));v=v@Q[j].T;v/=np.maximum(1e-8,np.linalg.norm(v,axis=2))[:,:,None]
          dot=(unit*v).sum(2);w=C[j]-C[i];b1=unit@w;b2=v@w;den=1-dot*dot
          t=(b1-dot*b2)/np.maximum(den,1e-6);s=(dot*b1-b2)/np.maximum(den,1e-6)
          p1=C[i]+unit*t[:,:,None];p2=C[j]+v*s[:,:,None];target=(p1+p2)/2
          gap=np.linalg.norm(p1-p2,axis=2);shift=np.linalg.norm(target-initial,axis=2)
          ok=source_valid&visible&(remap(visible.astype('f'))>.95)&(fb<.7)&(photometric<18)&(np.linalg.norm(forward,axis=2)<10)&(den>.0027)&(t>.2)&(s>.2)&(gap<.025)&(shift<.20)
          ok&=(x>12)&(x<S-13)&(y>12)&(y<S-13)&((x.astype(int)%3)==0)&((y.astype(int)%3)==0)
          yy,xx=np.where(ok);held=((xx//48+3*(yy//48)+face)%5)==0
          quality=np.exp(-fb[ok]**2/.3)*np.minimum(1,grad[ok]/80)
          chunks.append(dict(initial=initial[ok],target=target[ok],quality=quality,held=held,pair=np.tile([i,j],(ok.sum(),1))))
          reports.append(dict(source=n,target=stations[j],face=face,accepted=int(ok.sum()),median_fb=float(np.median(fb[ok])) if ok.any() else None))
          print('flow',n,stations[j],face,int(ok.sum()),flush=True)
    merged={key:np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
    np.savez_compressed(OUT/'correspondences.npz',**merged);write_json('flow-report.json',dict(method='depth-warped DIS forward/backward + visibility + triangulation',pairs=reports,total=len(merged['target']),held=int(merged['held'].sum())))
if __name__=='__main__':run()
