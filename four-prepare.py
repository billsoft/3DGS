"""Calibrate 4 matching panorama/3DGS pairs without scene landmarks."""
import argparse,importlib.util,json
from pathlib import Path
import cv2 as cv
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
def import_script(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
render=import_script('render',HERE/'calibration-render.py')
visibility=import_script('visibility',HERE/'visibility-test.py')
SIZE=640
FACES=render.bases()
K=np.array([[320.,0,320],[0,320.,320],[0,0,1.]])
SIFT=cv.SIFT_create(nfeatures=7000,contrastThreshold=.015)

def read_gs(path):
    with path.open('rb') as f:
        lines=[]
        while (line:=f.readline()).strip()!=b'end_header':
            if not line: raise ValueError(f'Incomplete PLY: {path}')
            lines.append(line.decode().strip())
        names=[x.split()[-1] for x in lines if x.startswith('property float ')]
        if names!=['x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']:
            raise ValueError(f'Unexpected Gaussian PLY fields: {path}')
        a=np.fromfile(f,'<f4').reshape(-1,14)
        if not np.isfinite(a).all(): raise ValueError(f'Non-finite PLY values: {path}')
        return a

def features(im,detector=SIFT):
    keys,desc=detector.detectAndCompute(cv.cvtColor(im,cv.COLOR_BGR2GRAY),None)
    return np.array([k.pt for k in keys]),desc

def matches(a,b,ratio=.76,approx=False):
    matcher=cv.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=96)) if approx else cv.BFMatcher()
    return np.array([(m.queryIdx,m.trainIdx) for pair in matcher.knnMatch(a,b,k=2)
                     if len(pair)==2 for m,n in [pair] if m.distance<ratio*n.distance],int).reshape(-1,2)

def gaussian_parameters(a):
    rot=Rotation.from_quat(a[:,[11,12,13,10]]).as_matrix()
    cov=(rot*np.exp(2*a[:,None,7:10]))@rot.transpose(0,2,1)
    rgb=np.clip(.5+.2820947918*a[:,3:6],0,1)
    opacity=1/(1+np.exp(-np.clip(a[:,6],-30,30)))
    return cov,rgb,opacity

def draw_views(a,params,rotation,center):
    cov,rgb,alpha=params
    points=(a[:,:3]-center).astype('d')
    results=[]
    for face in FACES:
        basis=rotation.T@face
        image,depth,acc=render.raster(points,cov,rgb,alpha,basis,SIZE)
        results.append((np.uint8(np.clip(image[:,:,::-1]*255,0,255)),depth,acc,basis,center))
    return results

def calibration(photo_faces,rendered):
    points=[];desc=[]
    for im,depth,alpha,basis,center in rendered:
        uv,ds=features(im);x,y=uv.round().astype(int).clip(0,SIZE-1).T
        good=alpha[y,x]>.94
        p=np.c_[(uv-320)/320,np.ones(len(uv))]*depth[y,x,None]
        points.append((p@basis.T+center)[good]);desc.append(ds[good])
    points=np.vstack(points);desc=np.vstack(desc)
    candidates=[];observations=[]
    for k,im in enumerate(photo_faces):
        uv,ds=features(im);pairs=matches(ds,desc)
        if len(pairs)<10:continue
        p=points[pairs[:,1]];xy=uv[pairs[:,0]];b=FACES[k]
        rays=np.c_[(xy-320)/320,np.ones(len(xy))]@b.T
        rays/=np.linalg.norm(rays,axis=1)[:,None]
        cells=xy.astype(int)//96
        held=(cells[:,0]+3*cells[:,1]+k)%5==0
        observations.append((p,rays,held,np.full(len(p),k)))
        if (~held).sum()<10:continue
        ok,rv,t,_=cv.solvePnPRansac(p[~held],xy[~held],K,None,iterationsCount=3000,reprojectionError=4,
                                  confidence=.999,flags=cv.SOLVEPNP_EPNP)
        if ok:
            rot=cv.Rodrigues(rv)[0]
            candidates.append((b@rot,-rot.T@t[:,0]))
    if not candidates:raise ValueError('No valid panorama-to-PLY camera pose')
    p,rays,held,face=[np.concatenate(c) for c in zip(*observations)]
    def error(R,C):
        q=(p-C)@R.T
        q/=np.linalg.norm(q,axis=1)[:,None]
        return q-rays
    R,C=max(candidates,key=lambda rc:np.sum((np.linalg.norm(error(*rc),axis=1)<.025)&~held))
    train=(np.linalg.norm(error(R,C),axis=1)<.035)&~held
    v=least_squares(lambda v:error(Rotation.from_rotvec(v[:3]).as_matrix(),v[3:])[train].ravel(),
                    np.r_[Rotation.from_matrix(R).as_rotvec(),C],loss='soft_l1',f_scale=.008).x
    R=Rotation.from_rotvec(v[:3]).as_matrix();C=v[3:]
    e=np.linalg.norm(error(R,C),axis=1);good=e<.02
    report=dict(held_inliers=int((held&good).sum()),held_total=int(held.sum()),
                train_inliers=int((~held&good).sum()),supported_faces=int(len(np.unique(face[good]))),
                held_inlier_median_deg=float(np.rad2deg(np.median(e[held&good]))) if (held&good).any() else None)
    return R,C,report

def prepare(photo,path,dest):
    if (dest/'cache.npz').exists():return
    dest.mkdir(parents=True,exist_ok=True)
    photo=cv.imdecode(np.fromfile(photo,np.uint8),cv.IMREAD_COLOR)
    if photo is None or abs(photo.shape[1]/photo.shape[0]-2)>.02:raise ValueError('Expected 2:1 equirectangular image')
    a=read_gs(path);params=gaussian_parameters(a)
    faces=[render.photo_view(photo,b,SIZE) for b in FACES]
    views=draw_views(a,params,np.eye(3),np.zeros(3))
    R,C,coarse=calibration(faces,views)
    R,C,report=calibration(faces,draw_views(a,params,R,C))
    print(path.stem,'camera',report,flush=True)
    if report['held_inliers']<12 or report['supported_faces']<2:
        raise ValueError(f'Camera evidence insufficient: {path.stem}: {report}')
    views=draw_views(a,params,R,C)
    max_contribution=np.zeros(len(a)); projected=np.zeros(len(a));contributed=np.zeros(len(a))
    cov,_,alpha=params
    for k,b in enumerate(FACES):
        peak,total,visible=visibility.contribution((a[:,:3]-C).astype('d'),cov,alpha,R.T@b,512)
        max_contribution=np.maximum(max_contribution,peak);projected+=total;contributed+=visible
        rgb,depth,opacity,basis,center=views[k]
        cv.imwrite(str(dest/f'source-{k}.jpg'),faces[k])
        cv.imwrite(str(dest/f'render-{k}.jpg'),rgb)
        np.savez_compressed(dest/f'depth-{k}.npz',depth=depth,alpha=opacity,basis=basis)
    keep=(max_contribution>.003)|(contributed/np.maximum(projected,1e-9)>.02)
    report.update(input=len(a),kept=int(keep.sum()),removed=int((~keep).sum()))
    (dest/'camera.json').write_text(json.dumps(report,indent=2))
    np.save(dest/'visible.npy',a[keep])
    np.savez(dest/'cache.npz',R=R,C=C)
    print(path.stem,'visibility',report,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('inputs',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    for path in sorted(args.inputs.glob('*.ply')):
        prepare(args.inputs/f'{path.stem}_7680.jpg',path,args.output/path.stem)
