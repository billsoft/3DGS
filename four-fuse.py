"""Fuse calibrated Gaussian scenes using source visibility and depth agreement."""
import argparse,json
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
import cv2 as cv
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
spec=spec_from_file_location('prep',Path(__file__).with_name('four-prepare.py'))
prep=module_from_spec(spec);spec.loader.exec_module(prep)

def transform(a,m):
    a=a.copy();a[:,:3]=a[:,:3]@m[:3,:3].T+m[:3,3]
    scale=np.cbrt(np.linalg.det(m[:3,:3]));a[:,7:10]+=np.log(scale)
    q=Rotation.from_matrix(m[:3,:3]/scale)*Rotation.from_quat(a[:,[11,12,13,10]])
    a[:,10:14]=q.as_quat()[:,[3,0,1,2]]
    return a

def write(path,a):
    names='x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3'.split()
    header='ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(a))+'\n'
    header+=''.join('property float '+name+'\n' for name in names)+'end_header\n'
    with path.open('wb') as f:f.write(header.encode());a.astype('<f4',copy=False).tofile(f)

def table_plane(clouds,centers,up):
    center=centers.mean(0);height=np.median(centers@up)
    x=np.cross([0,1,0],up);x/=np.linalg.norm(x);y=np.cross(up,x)
    axes=np.stack([x,y,up],1)
    planes=[];members=[]
    for a in clouds:
        p=(a[:,:3]-center)@axes;rgb=np.clip(.5+.2820947918*a[:,3:6],0,1)
        bright=(rgb[:,0]>.57)&(rgb[:,0]-rgb[:,2]>.03)&(rgb[:,0]-rgb[:,2]<.27)
        eligible=bright&(p[:,2]>-2.3)&(p[:,2]<-.40)
        bins=np.arange(-2.30,-.39,.05)
        counts,_=np.histogram(p[eligible,2],bins);peak=np.argmax(counts)
        z0=(bins[peak]+bins[peak+1])/2
        ids=np.flatnonzero(eligible&(abs(p[:,2]-z0)<.11))
        if len(ids)<2500:return None
        sample=p[ids[::max(1,len(ids)//6000)]]
        A=np.c_[sample[:,:2],np.ones(len(sample))]
        v=least_squares(lambda v:A@v-sample[:,2],[0,0,z0],loss='soft_l1',f_scale=.025,
                        bounds=([-.18,-.18,z0-.10],[.18,.18,z0+.10])).x
        inside=ids[abs(np.c_[p[ids,:2],np.ones(len(ids))]@v-p[ids,2])<.055]
        if len(inside)<2500:return None
        planes.append(v);members.append(p[inside])
    planes=np.array(planes)
    if np.ptp(planes[:,2])>.25 or np.max(np.ptp(planes[:,:2],axis=0))>.15:return None
    target=np.median(planes,axis=0)
    xy=np.vstack(members)[:,:2];lo,hi=np.quantile(xy,[.01,.99],axis=0)
    moved=[]
    for a,cam,v in zip(clouds,centers,planes):
        p=(a[:,:3]-center)@axes;o=(cam-center)@axes
        delta=p-o;residual=p[:,2]-np.c_[p[:,:2],np.ones(len(p))]@v
        dist=np.abs(residual)
        within=((p[:,:2]>lo-.15)&(p[:,:2]<hi+.15)).all(1)
        # Limit the edit to nearly horizontal splats on the fitted sheet.
        rot=Rotation.from_quat(a[:,[11,12,13,10]]).as_matrix()
        n=rot[np.arange(len(a)),:,np.argmin(a[:,7:10],axis=1)]
        flat=(np.abs(n@up)>.78)
        weight=np.clip((.092-dist)/.045,0,1)*within*flat
        normal=np.r_[-target[:2],1]
        den=delta@normal
        t=np.divide(target[2]-o@normal,den,out=np.ones(len(a)),where=abs(den)>.10)
        move=np.clip(t-1,-.16,.16)*weight
        a[:,:3]+=(delta*move[:,None])@axes.T
        moved.append(int((abs(move)>.001).sum()))
    return dict(source_planes=planes.tolist(),common_plane=target.tolist(),corrected=moved)

class View:
    def __init__(self,d,m):
        cam=np.load(d/'cache.npz');self.R=cam['R'];self.C=cam['C'];self.inv=np.linalg.inv(m)
        self.world_center=m[:3,:3]@self.C+m[:3,3];self.maps=[]
        for k in range(6):
            z=np.load(d/f'depth-{k}.npz');dep=z['depth']
            kern=np.ones((3,3),np.uint8)
            spread=cv.dilate(dep,kern)-cv.erode(dep,kern)
            reliable=(z['alpha']>.96)&(spread<.10*np.maximum(.25,dep))
            self.maps.append((dep,reliable.astype('f')))
    def observe(self,p):
        q=((p@self.inv[:3,:3].T+self.inv[:3,3])-self.C)@self.R.T
        face=np.argmax(np.stack([q@b[:,2] for b in prep.FACES]),axis=0)
        good=np.zeros(len(p),bool);diff=np.zeros(len(p));depth=np.zeros(len(p))
        for k,b in enumerate(prep.FACES):
            ids=np.flatnonzero(face==k)
            for k0 in range(0,len(ids),25000):
                j=ids[k0:k0+25000];v=q[j]@b
                z=v[:,2];uv=(v[:,:2]/np.maximum(.001,z[:,None])*320+320).astype('f')
                def sample(a):return cv.remap(a,uv[:,0,None],uv[:,1,None],cv.INTER_NEAREST,borderMode=cv.BORDER_REPLICATE)[:,0]
                d,mask=self.maps[k];obs=sample(d)
                good[j]=(sample(mask)>.5)&(z>.15)&(obs>.15)
                diff[j]=z-obs;depth[j]=obs
        return good,diff,depth

def to_splat(path,a):
    b=np.empty(len(a),dtype=[('p','<f4',3),('s','<f4',3),('c','u1',4),('q','u1',4)])
    b['p']=a[:,:3];b['s']=np.exp(a[:,7:10]);b['c'][:,:3]=np.clip((.5+.2820947918*a[:,3:6])*255,0,255)
    b['c'][:,3]=255/(1+np.exp(-np.clip(a[:,6],-30,30)))
    q=a[:,10:14];q=q/np.linalg.norm(q,axis=1)[:,None];b['q']=np.clip(q*128+128,0,255)
    b.tofile(path)

def run(out):
    reg=json.loads((out/'registration.json').read_text());stations=reg['stations'];ms=np.array(reg['matrices'])
    views=[View(out/n,m) for n,m in zip(stations,ms)];centers=np.array([v.world_center for v in views])
    clouds=[transform(np.load(out/n/'visible.npy'),m) for n,m in zip(stations,ms)]
    up=np.array([m[:3,:3]@v.R[2] for m,v in zip(ms,views)])
    up=np.mean(up/np.linalg.norm(up,axis=1)[:,None],0);up/=np.linalg.norm(up)
    table=table_plane(clouds,centers,up);print('Table',table,flush=True)
    fused=[];report={}
    for i,a in enumerate(clouds):
        p=a[:,:3];prior=(.30+np.linalg.norm(p-centers[i],axis=1))**-2.3
        total=prior.copy();support=np.zeros(len(a),'i');conflict=np.zeros(len(a),'i')
        for j,v in enumerate(views):
            if i==j:continue
            ok,delta,depth=v.observe(p)
            tolerance=.07+.04*depth
            compatible=ok&(abs(delta)<tolerance)
            support+=compatible
            conflict+=ok&(delta<-2*tolerance)
            confidence=np.exp(-.5*(delta/np.maximum(tolerance,.001))**2)
            total+=np.where(compatible,confidence,0)*( .30+np.linalg.norm(p-centers[j],axis=1))**-2.3
        # Never penalize a surface solely because another camera has an occluder.
        density=prior/total
        attenuate=(conflict>=2)&(support==0)
        density[attenuate]*=.28
        alpha=1/(1+np.exp(-np.clip(a[:,6],-30,30)))
        adjusted=np.clip(-np.expm1(np.log1p(-np.minimum(alpha,.99999))*density),1e-7,1-1e-7)
        a[:,6]=np.log(adjusted/(1-adjusted))
        keep=adjusted>.008
        fused.append(a[keep]);report[stations[i]]=dict(cross_supported=int((support>0).sum()),
                       uncertain_attenuated=int(attenuate.sum()),retained=int(keep.sum()))
        print(stations[i],report[stations[i]],flush=True)
    from four_screen import replace_screen
    images=[cv.imread(str(out/n/'source-3.jpg')) for n in stations]
    bases=[m[:3,:3]@v.R.T@prep.FACES[3] for m,v in zip(ms,views)]
    fused,screen=replace_screen(fused,images,bases,centers)
    print('Screen',screen,flush=True)
    a=np.vstack(fused)
    write(out/'fused.ply',a);to_splat(out/'fused.splat',a)
    payload=dict(stations=stations,centers=centers.tolist(),up=up.tolist(),table=table,screen=screen,splats=len(a),fusion=report)
    (out/'fusion.json').write_text(json.dumps(payload,indent=2))
    print('Fused',len(a),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path)
    run(p.parse_args().out)
