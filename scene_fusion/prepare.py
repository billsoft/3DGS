"""Calibrate each panorama against its own PLY, then audit source visibility."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
import cv2 as cv
import torch
from core import prep,initialize,write_json,viewmat
from refit_render import load_scene,render
from input_io import read_gaussians,normalize
from run_state import digest

def verified_cache(folder,fingerprint):
    marker=folder/'complete.json'
    if not marker.exists():return False
    try:
        info=json.loads(marker.read_text(encoding='utf8'));artifacts=info.get('artifacts',{})
        return info.get('fingerprint')==fingerprint and bool(artifacts) and all((folder/name).is_file() and digest(folder/name)==sha for name,sha in artifacts.items())
    except (OSError,ValueError):return False

def prepare(item,out,cache_root=None):
    folder=out/item['id'];folder.mkdir(exist_ok=True)
    code_files=[Path(__file__),Path(__file__).with_name('input_io.py'),Path(prep.__file__),Path(prep.render.__file__),Path(prep.visibility.__file__)]
    fingerprint=hashlib.sha256((item['photo_sha256']+item['ply_sha256']).encode()+b''.join(p.read_bytes() for p in code_files)).hexdigest()
    marker=folder/'complete.json'
    if verified_cache(folder,fingerprint):
        print('Cached',item['label'],flush=True);return
    shared=cache_root/fingerprint if cache_root else None
    if shared and verified_cache(shared,fingerprint):
        shutil.copytree(shared,folder,dirs_exist_ok=True);print('Reused verified source',item['label'],flush=True);return
    marker.unlink(missing_ok=True)
    photo=cv.imdecode(np.fromfile(item['photo'],np.uint8),cv.IMREAD_COLOR)
    if photo is None or abs(photo.shape[1]/photo.shape[0]-2)>.01:raise ValueError('Expected 2:1 panorama')
    a,normalization=normalize(read_gaussians(item['ply']));write_json(folder/'normalization.json',normalization)
    np.save(folder/'original.npy',a);scene=load_scene(folder/'original.npy')
    faces=[prep.render.photo_view(photo,b,prep.SIZE) for b in prep.FACES]
    def draw(R,C):
        results=[]
        with torch.no_grad():
            for b in prep.FACES:
                basis=R.T@b;image,alpha,_=render(scene,dict(viewmat=viewmat(C,basis).tolist()),prep.SIZE,depth=True)
                image=image.cpu().numpy();alpha=alpha.cpu().numpy()
                results.append(((image[:,:,:3][:,:,::-1].clip(0,1)*255).astype('u1'),image[:,:,3],alpha,basis,C))
        return results
    guesses=[np.zeros(3),np.median(a[::10,:3],axis=0)];failure=[]
    for guess in guesses:
        try:
            R,C,coarse=prep.calibration(faces,draw(np.eye(3),guess))
            if coarse['held_inliers']<12 or coarse['supported_faces']<2:raise ValueError('Too few verified camera features')
            break
        except (ValueError,cv.error,TypeError) as error:failure.append(str(error))
    else:raise ValueError(f"{item['label']}: camera calibration failed; check image/PLY pairing and texture. {failure}")
    R,C,report=prep.calibration(faces,draw(R,C));print('Camera',item['id'],report,flush=True)
    if report['held_inliers']<12 or report['supported_faces']<2:raise ValueError('Insufficient independent camera evidence')
    for k,(rgb,depth,alpha,basis,center) in enumerate(draw(R,C)):
        cv.imwrite(str(folder/f'source-{k}.jpg'),faces[k]);cv.imwrite(str(folder/f'render-{k}.jpg'),rgb)
        np.savez_compressed(folder/f'depth-{k}.npz',depth=depth,alpha=alpha,basis=basis)
    del scene;torch.cuda.empty_cache()
    cov,rgb,alpha=prep.gaussian_parameters(a);maximum=np.zeros(len(a));projected=np.zeros(len(a));contributed=np.zeros(len(a))
    for k,b in enumerate(prep.FACES):
        peak,total,visible=prep.visibility.contribution((a[:,:3]-C).astype('d'),cov,alpha,R.T@b,512)
        maximum=np.maximum(maximum,peak);projected+=total;contributed+=visible
        print('Visibility',item['id'],k+1,'/ 6 cube faces',flush=True)
    keep=(maximum>.003)|(contributed/np.maximum(projected,1e-9)>.02)
    np.save(folder/'visible.npy',a[keep]);np.save(folder/'visible-mask.npy',keep)
    np.savez(folder/'cache.npz',R=R,C=C)
    report.update(input=len(a),retained=int(keep.sum()),removed=int((~keep).sum()),photo_shape=list(photo.shape[:2]))
    write_json(folder/'camera.json',report)
    names=['original.npy','normalization.json','visible.npy','visible-mask.npy','cache.npz','camera.json']+[f'{kind}-{k}.{ext}' for k in range(6) for kind,ext in [('depth','npz'),('source','jpg'),('render','jpg')]]
    artifacts={name:digest(folder/name) for name in names}
    write_json(folder/'complete.json',dict(calibrated=True,source_visibility=True,fingerprint=fingerprint,artifacts=artifacts))
    if shared:shutil.copytree(folder,shared,dirs_exist_ok=True)
    print('Prepared',item['id'],report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stations',nargs='+');p.add_argument('--cache-root',type=Path)
    args=p.parse_args();meta=initialize(args.inputs,args.out,args.stations)
    cv.setNumThreads(4);torch.set_num_threads(4)
    for item in meta['stations']:prepare(item,args.out,args.cache_root)
