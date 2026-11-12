"""Smooth the displacement field, not the starting surface or its texture.

Geometry weights deliberately ignore RGB: a richly textured curved surface still
needs coherent motion. Differing normals and separated surfaces stop propagation.
"""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix,diags

parser=argparse.ArgumentParser()
parser.add_argument('--initial',type=Path,required=True)
parser.add_argument('--candidate',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
a=np.load(args.initial);b=np.load(args.candidate)
assert a.shape==b.shape
tree=cKDTree(a[:,:3]);normal=Rotation.from_quat(a[:,[11,12,13,10]]).as_matrix()[:,:,0]
src=[];dst=[];weights=[]
for start in range(0,len(a),100000):
    ids=np.arange(start,min(start+100000,len(a)))
    dist,ix=tree.query(a[ids,:3],k=9,workers=4)
    s=np.repeat(ids,8);t=ix[:,1:].ravel();d=dist[:,1:].ravel()
    cos=abs((normal[s]*normal[t]).sum(1))
    # Do not connect parallel layers across their normal direction.
    off=abs(((a[t,:3]-a[s,:3])*normal[s]).sum(1))
    w=np.exp(-(d/.025)**2-(off/.004)**2)*np.clip((cos-.85)/.15,0,1)**2
    take=(s<t)&(w>.03)&(d<.04)
    src.append(s[take]);dst.append(t[take]);weights.append(w[take])
s=np.concatenate(src);t=np.concatenate(dst);w=np.concatenate(weights).astype('f')
W=coo_matrix((np.r_[w,w],(np.r_[s,t],np.r_[t,s])),shape=(len(a),len(a))).tocsr()
degree=np.asarray(W.sum(1)).ravel()
W=diags(1/np.maximum(degree,1e-9))@W
original=b[:,:3]-a[:,:3];delta=original.copy()
for _ in range(8):
    delta=.35*original+.65*(W@delta)
    delta[degree==0]=original[degree==0]
b[:,:3]=a[:,:3]+delta
np.save(args.out/'motion-smoothed.npy',b)
report=dict(method='normal/tangent-consistent displacement diffusion; no semantic/color geometry mask',
    edges=len(s),iterations=8,anchor_weight=.35,
    mean_additional_shift=float(np.linalg.norm(delta-original,axis=1).mean()),
    max_additional_shift=float(np.linalg.norm(delta-original,axis=1).max()))
(args.out/'motion-smoothing.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
