"""Compare actual browser frames to calibrated photos at original stations.

The output is appearance disagreement, not a geometric ground truth metric.
Uses the captured browser camera matrix, including actual roll and FOV.
"""
import argparse,json
from pathlib import Path
import numpy as np,cv2 as cv
from core import load

def run(a):
    pose=json.loads(a.pose.read_text('utf8'));meta=load(a.source)
    cam=np.load(a.source/'aligned-cameras.npz');i=a.station
    if np.linalg.norm(np.array(pose['center'])-cam['C'][i])>1e-5:raise ValueError('Photo reference requires exact original station')
    im=cv.imdecode(np.fromfile(meta['stations'][i]['photo'],np.uint8),1)
    h,w=pose['height'],pose['width'];y,x=np.mgrid[:h,:w];f=h/2/np.tan(np.deg2rad(pose['fov'])/2)
    direction=np.stack([(x+.5-w/2)/f,-(y+.5-h/2)/f,-np.ones_like(x)],-1)
    matrix=np.array(pose['matrixWorld']).reshape(4,4).T
    local=direction@matrix[:3,:3].T@cam['Q'][i]
    u=np.arctan2(-local[:,:,1],local[:,:,0])/(2*np.pi)+.5
    v=np.arctan2(np.linalg.norm(local[:,:,:2],axis=-1),local[:,:,2])/np.pi
    ref=cv.remap(im,(u*im.shape[1]-.5).astype('f'),(v*im.shape[0]-.5).astype('f'),cv.INTER_LINEAR,borderMode=cv.BORDER_WRAP)
    a.out.parent.mkdir(parents=True,exist_ok=True);cv.imwrite(str(a.out.with_suffix('.png')),ref)
    records=[]
    for path in a.image:
        photo=cv.imread(str(path));assert photo.shape==ref.shape
        delta=abs(photo.astype(float)-ref)/255
        records.append(dict(path=str(path),mae=float(delta.mean()),p95_channel_error=float(np.quantile(delta,.95)),
            pixels_error_over_0_15=int((delta.mean(2)>.15).sum())))
    a.out.write_text(json.dumps(dict(station=i,records=records,warning='Original-photo appearance discrepancy, including residual calibration and view-dependent appearance; not measured geometric accuracy.'),indent=2),encoding='utf8');print(records)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--station',type=int,required=True);p.add_argument('--pose',type=Path,required=True);p.add_argument('--image',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
