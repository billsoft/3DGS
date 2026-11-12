"""Recover a common coordinate frame from panorama feature correspondences."""
import argparse,json
from itertools import combinations
from pathlib import Path
import cv2 as cv
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('prepare',Path(__file__).resolve().parents[1]/'four-prepare.py')
prep=module_from_spec(spec);spec.loader.exec_module(prep)
def features(photo,station):
    raw=cv.imdecode(np.fromfile(photo,np.uint8),cv.IMREAD_COLOR)
    W,H=3840,1920
    im=cv.resize(raw,(W,H),interpolation=cv.INTER_AREA)
    uv,des=prep.features(im,cv.SIFT_create(nfeatures=28000,contrastThreshold=.012))
    if des is None or len(uv)<12:raise ValueError('Insufficient panorama features: '+str(photo))
    lon=(uv[:,0]/W-.5)*2*np.pi;lat=uv[:,1]/H*np.pi
    rays=np.stack([np.cos(lon)*np.sin(lat),-np.sin(lon)*np.sin(lat),np.cos(lat)],1)
    points=np.zeros_like(rays);valid=np.zeros(len(rays),bool)
    face=np.argmax(np.array([rays@b[:,2] for b in prep.FACES]),axis=0)
    C=np.load(station/'cache.npz')['C']
    for k,b in enumerate(prep.FACES):
        ids=np.flatnonzero(face==k)
        if not len(ids):continue
        q=rays[ids]@b
        xy=q[:,:2]/q[:,2,None]*320+320
        xx,yy=xy.T.astype('f')
        z=np.load(station/f'depth-{k}.npz')
        def sample(im):return cv.remap(im,xx[:,None],yy[:,None],cv.INTER_LINEAR,borderMode=cv.BORDER_REPLICATE)[:,0]
        dep=sample(z['depth']);opacity=sample(z['alpha'])
        kernel=np.ones((3,3),np.uint8)
        span=sample(cv.dilate(z['depth'],kernel)-cv.erode(z['depth'],kernel))
        points[ids]=(q/q[:,2,None]*dep[:,None])@z['basis'].T+C
        valid[ids]=(opacity>.95)&(span<.12*dep)&(dep>.1)
    cells=(uv.astype(int)//144)
    held=(cells[:,0]+3*cells[:,1])%5==0
    return des[valid],points[valid],held[valid],cells[valid]

def similarity(x,y):
    a=x-x.mean(0);b=y-y.mean(0)
    u,s,v=np.linalg.svd(a.T@b)
    flip=np.diag([1,1,np.linalg.det(v.T@u.T)])
    r=v.T@flip@u.T;scale=(s@flip.diagonal())/max((a*a).sum(),1e-10)
    m=np.eye(4);m[:3,:3]=scale*r;m[:3,3]=y.mean(0)-m[:3,:3]@x.mean(0)
    return m
def apply(m,x):return x@m[:3,:3].T+m[:3,3]
def pack(m):
    s=np.cbrt(np.linalg.det(m[:3,:3]))
    return np.r_[Rotation.from_matrix(m[:3,:3]/s).as_rotvec(),m[:3,3],np.log(s)]
def unpack(v):
    m=np.eye(4);m[:3,:3]=Rotation.from_rotvec(v[:3]).as_matrix()*np.exp(v[6]);m[:3,3]=v[3:6]
    return m

def pair(a,b,rng):
    ab=prep.matches(a[0],b[0],.79,True);ba=dict(prep.matches(b[0],a[0],.79,True))
    ids=np.array([(i,j) for i,j in ab if ba.get(j)==i],dtype=int).reshape(-1,2)
    if len(ids)<18:return None
    i,j=ids.T;x=b[1][j];y=a[1][i];hold=a[2][i]|b[2][j]
    train=np.flatnonzero(~hold)
    if len(train)<12:return None
    # Relative to the scene's robust extent, not a presumed metric room size.
    extent=np.median(np.linalg.norm(y-np.median(y,axis=0),axis=1))
    tol=max(.02,extent*.065);best=np.zeros(len(x),bool)
    expected=np.median(np.linalg.norm(y-np.median(y,axis=0),axis=1))/max(np.median(np.linalg.norm(x-np.median(x,axis=0),axis=1)),1e-9)
    budget=10000
    for iteration in range(10000):
        if iteration>=max(512,budget):break
        k=rng.choice(train,3,False);m=similarity(x[k],y[k])
        scale=np.cbrt(np.linalg.det(m[:3,:3]))
        if not expected/4<scale<expected*4:continue
        good=(np.linalg.norm(apply(m,x)-y,axis=1)<tol)&~hold
        if good.sum()>best.sum():
            best=good;fraction=float(best.sum())/len(train)
            if fraction>.05:budget=min(budget,int(np.ceil(np.log(.001)/np.log(max(1e-12,1-fraction**3)))))
    if best.sum()<6:return None
    for _ in range(4):
        m=similarity(x[best],y[best]);err=np.linalg.norm(apply(m,x)-y,axis=1)
        best=(err<tol)&~hold
    cells=len(np.unique(a[3][i[best]],axis=0))
    return dict(m=m,x=x,y=y,hold=hold,train=best,cells=cells,tol=tol,err=err)

def solve(inputs,out,numbers):
    fs=[]
    for item in inputs:
        path=out/item['id']/'registration-features.npz'
        # The runner owns checkpoints; a rerun must not reuse stale calibration features.
        row=features(Path(item['photo']),out/item['id']);fs.append(row)
        np.savez_compressed(path,desc=row[0],points=row[1],held=row[2],cells=row[3])
        print('Features',item['id'],len(fs[-1][0]),flush=True)
    edges={};results={};rng=np.random.default_rng(44)
    for i,j in combinations(range(len(numbers)),2):
        e=pair(fs[i],fs[j],rng)
        if e is None:continue
        h=e['hold'];passed=int((h&(e['err']<e['tol'])).sum())
        stats=dict(matches=len(h),train=int(e['train'].sum()),holdout=int(h.sum()),held_inliers=passed,cells=e['cells'])
        stats['accepted']=stats['train']>=12 and passed>=4 and e['cells']>=4
        results[f'{numbers[i]}-{numbers[j]}']=stats
        if stats['accepted']:edges[i,j]=e
        print(numbers[i],numbers[j],stats,flush=True)
    poses={0:np.eye(4)}
    for _ in numbers:
        for (i,j),e in sorted(edges.items(),key=lambda p:-p[1]['train'].sum()):
            if i in poses and j not in poses:poses[j]=poses[i]@e['m']
            if j in poses and i not in poses:poses[i]=poses[j]@np.linalg.inv(e['m'])
    if len(poses)!=len(numbers):
        (out/'registration-failed.json').write_text(json.dumps(results,indent=2))
        raise ValueError(f'Validation disconnected stations {set(range(len(numbers)))-set(poses)}')
    init=np.concatenate([pack(poses[i]) for i in range(1,len(numbers))])
    def matrices(v):return [np.eye(4)]+[unpack(p) for p in v.reshape(-1,7)]
    def residual(v):
        ms=matrices(v);errors=[]
        for (i,j),e in edges.items():
            k=e['train']
            errors.append((apply(ms[j],e['x'][k])-apply(ms[i],e['y'][k])).ravel())
        return np.r_[np.concatenate(errors),.08*(v-init)]
    sol=least_squares(residual,init,loss='soft_l1',f_scale=.065,max_nfev=120)
    ms=matrices(sol.x)
    for (i,j),e in edges.items():
        residuals=np.linalg.norm(apply(ms[j],e['x'])-apply(ms[i],e['y']),axis=1)
        valid=e['hold']&(residuals<e['tol'])
        results[f'{numbers[i]}-{numbers[j]}'].update(global_held_inliers=int(valid.sum()),
            global_inlier_median=float(np.median(residuals[valid])) if valid.any() else None)
    payload=dict(stations=numbers,matrices=[m.tolist() for m in ms],edges=results)
    (out/'registration.json').write_text(json.dumps(payload,indent=2))
    print('Registered',numbers,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();scene=json.loads((args.out/'scene.json').read_text(encoding='utf8'));items=scene['stations'];numbers=[item['id'] for item in items]
    solve(items,args.out,numbers)
