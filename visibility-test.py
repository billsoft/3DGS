import numpy as np,json,importlib.util
from numba import njit
from scipy.spatial.transform import Rotation
from pathlib import Path
s=importlib.util.spec_from_file_location('r','calibration-render.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
@njit
def contribution(p,cov,alpha,basis,size):
 q=p@basis;order=np.argsort(q[:,2]);T=np.ones((size,size),np.float32)
 maximum=np.zeros(len(p),np.float32);total=np.zeros(len(p),np.float32);seen=np.zeros(len(p),np.float32)
 focal=size*.5
 for k in order:
  x,y,z=q[k]
  if z<.1:continue
  u=x/z*focal+size*.5;v=y/z*focal+size*.5
  j=np.zeros((2,3));j[0,0]=focal/z;j[0,2]=-focal*x/z**2;j[1,1]=focal/z;j[1,2]=-focal*y/z**2
  cc=j@(basis.T@cov[k]@basis)@j.T;cc[0,0]+=.3;cc[1,1]+=.3
  det=cc[0,0]*cc[1,1]-cc[0,1]**2
  if det<=0:continue
  rx=min(120.,3*np.sqrt(cc[0,0]));ry=min(120.,3*np.sqrt(cc[1,1]))
  if u+rx<0 or u-rx>=size or v+ry<0 or v-ry>=size:continue
  for yy in range(max(0,int(v-ry)),min(size,int(v+ry)+1)):
   for xx in range(max(0,int(u-rx)),min(size,int(u+rx)+1)):
    dx=xx+.5-u;dy=yy+.5-v;power=(cc[1,1]*dx*dx-2*cc[0,1]*dx*dy+cc[0,0]*dy*dy)/det
    if power>9:continue
    a=min(.99,alpha[k]*np.exp(-.5*power))
    weight=T[yy,xx]*a;maximum[k]=max(maximum[k],weight);total[k]+=a;seen[k]+=weight;T[yy,xx]*=1-a
 return maximum,total,seen
if __name__=='__main__':
 cameras=json.load(open('calibration/cameras.json'));report={}
 for n in [2,4]:
  a=np.load(f'{n}.npy');rot=Rotation.from_quat(a[:,[11,12,13,10]]).as_matrix()
  cov=(rot*np.exp(2*a[:,None,7:10]))@rot.transpose(0,2,1)
  alpha=1/(1+np.exp(-np.clip(a[:,6],-30,30)))
  cam=cameras[str(n)];R=np.array(cam['world_to_photo']);C=np.array(cam['center'])
  maximum=np.zeros(len(a));total=np.zeros(len(a));seen=np.zeros(len(a))
  for b in r.bases():
   ma,to,se=contribution((a[:,:3]-C).astype('d'),cov,alpha,R.T@b,512)
   maximum=np.maximum(maximum,ma);total+=to;seen+=se
  # Only remove splats that are occluded throughout their footprint, not merely at the center.
  keep=(maximum>.003)|(seen/np.maximum(total,1e-9)>.02)
  np.savez_compressed(f'calibration/visibility-{n}.npz',keep=keep,contribution=maximum,ratio=seen/np.maximum(total,1e-9))
  report[str(n)]=dict(input=len(a),keep=int(keep.sum()),removed=int((~keep).sum()))
  print(n,report[str(n)],flush=True)
 Path('calibration/visibility.json').write_text(json.dumps(report))
