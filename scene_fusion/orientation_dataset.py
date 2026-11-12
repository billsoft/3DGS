"""Same optical centers, multiple rotations/FOVs; no synthesized target images."""
import argparse,json
from pathlib import Path
import numpy as np,cv2 as cv
from scipy.spatial.transform import Rotation
from core import load,prep,write_json,viewmat
from photo_recolor import sample_erp

def main(a):
    a.out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.seed)
    cams=np.load(a.scene/'aligned-cameras.npz');meta=load(a.scene)
    prior=json.loads((a.scene/'data-512/dataset.json').read_text());old=np.load(a.scene/'data-512/targets.npy')
    views=[] if a.held else prior['views'].copy();images=[] if a.held else [x for x in old]
    size=512;y,x=np.indices((size,size))
    for s,item in enumerate(meta['stations']):
        photo=cv.imdecode(np.fromfile(item['photo'],np.uint8),1)[:,:,::-1]
        for face,b in enumerate(prep.FACES):
            for repeat in range(1 if a.held else 2):
                # Both small tilts and roll matter to checking view consistency.
                tilt=Rotation.from_euler('xyz',[rng.uniform(-25,25),rng.uniform(-25,25),rng.uniform(-180,180)],degrees=True).as_matrix()
                basis=cams['Q'][s]@b@tilt;center=cams['C'][s];fov=float(rng.uniform(75,115))
                focal=size/(2*np.tan(np.deg2rad(fov)/2));rays=np.stack([(x+.5-size/2)/focal,(y+.5-size/2)/focal,np.ones_like(x)],-1)
                points=rays.reshape(-1,3)@basis.T+center
                rgb=sample_erp(photo,points,center,cams['Q'][s]).reshape(size,size,3)
                views.append(dict(name=f'rotation-{s}-{face}-{repeat}',center=center.tolist(),basis=basis.tolist(),viewmat=viewmat(center,basis).tolist(),source=s,role='train',fov=fov))
                images.append(rgb)
    np.save(a.out/'targets.npy',np.stack(images).astype('u1'))
    prior.update(size=size,views=views,projection_seed=a.seed,held_projection=a.held)
    write_json(a.out/'dataset.json',prior);print('views',len(views),'held projection',a.held,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=4927);p.add_argument('--held',action='store_true');main(p.parse_args())
