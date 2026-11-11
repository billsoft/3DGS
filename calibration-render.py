"""CPU Gaussian renderer for image-to-model calibration; no point-center z-buffer."""
import numpy as np,cv2 as cv
from scipy.spatial.transform import Rotation
from numba import njit
from pathlib import Path
@njit
def raster(p,cov,col,alpha,basis,size):
 focal=size*.5
 rgb=np.zeros((size,size,3),np.float32);depth=np.zeros((size,size),np.float32);acc=np.zeros((size,size),np.float32)
 q=p@basis
 order=np.argsort(q[:,2])[::-1]
 for k in order:
  x,y,z=q[k]
  if z<.1:continue
  u=x/z*focal+size*.5;v=y/z*focal+size*.5
  j=np.zeros((2,3));j[0,0]=focal/z;j[0,2]=-focal*x/z**2;j[1,1]=focal/z;j[1,2]=-focal*y/z**2
  cc=j@(basis.T@cov[k]@basis)@j.T
  cc[0,0]+=.3;cc[1,1]+=.3
  det=cc[0,0]*cc[1,1]-cc[0,1]**2
  if det<=0:continue
  rx=min(120.,3*np.sqrt(cc[0,0]));ry=min(120.,3*np.sqrt(cc[1,1]))
  if u+rx<0 or u-rx>=size or v+ry<0 or v-ry>=size:continue
  for yy in range(max(0,int(v-ry)),min(size,int(v+ry)+1)):
   for xx in range(max(0,int(u-rx)),min(size,int(u+rx)+1)):
    dx=xx+.5-u;dy=yy+.5-v
    power=(cc[1,1]*dx*dx-2*cc[0,1]*dx*dy+cc[0,0]*dy*dy)/det
    if power>9:continue
    a=min(.99,alpha[k]*np.exp(-.5*power))
    for ch in range(3):rgb[yy,xx,ch]=col[k,ch]*a+rgb[yy,xx,ch]*(1-a)
    depth[yy,xx]=z*a+depth[yy,xx]*(1-a)
    acc[yy,xx]=a+acc[yy,xx]*(1-a)
 return rgb,depth/np.maximum(acc,1e-8),acc
def bases():
 result=[]
 for d in [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]:
  z=np.array(d,float);up=np.array([0,0,1.]) if abs(z[2])<.9 else np.array([0,1.,0])
  x=np.cross(z,up);x/=np.linalg.norm(x);y=np.cross(z,x)
  result.append(np.stack([x,y,z],axis=1))
 return result
def photo_view(photo,b,size):
 yy,xx=np.indices((size,size));q=np.stack([(xx+.5-size/2)/(size/2),(yy+.5-size/2)/(size/2),np.ones_like(xx)],-1)@b.T
 # ERP camera convention used only for this image, not assumed equal to the PLY frame.
 lon=np.arctan2(-q[:,:,1],q[:,:,0]);lat=np.arctan2(np.linalg.norm(q[:,:,:2],axis=2),q[:,:,2])
 return cv.remap(photo,((lon/(2*np.pi)+.5)*photo.shape[1]).astype('f'),(lat/np.pi*photo.shape[0]).astype('f'),cv.INTER_LINEAR,borderMode=cv.BORDER_WRAP)
