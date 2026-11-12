"""Regularize normal displacement from the pre-refit surface, retaining tangential fit.

The reference is the existing room-wide geometric initialization, not a newly
fitted object template. Its errors remain a limitation of this proposal.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix,diags

p=argparse.ArgumentParser();p.add_argument('--initial',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
a=np.load(args.initial);b=np.load(args.candidate);assert a.shape==b.shape
n=Rotation.from_quat(a[:,[11,12,13,10]]).as_matrix().astype('f')[np.arange(len(a)),:,a[:,7:10].argmin(1)]
tree=cKDTree(a[:,:3]);src=[];dst=[];weights=[]
for start in range(0,len(a),60000):
    ids=np.arange(start,min(start+60000,len(a)));d,ix=tree.query(a[ids,:3],k=33,workers=4);d=d[:,1:];ix=ix[:,1:]
    s=np.repeat(ids,32);t=ix.ravel();d=d.ravel();dot=(n[s]*n[t]).sum(1)
    off=abs(((a[t,:3]-a[s,:3])*n[s]).sum(1))
    w=np.exp(-(d/.04)**2-(off/.006)**2)*np.clip((abs(dot)-.9)/.1,0,1)**2
    take=(w>.03)&(d<.07)
    src.append(s[take]);dst.append(t[take]);weights.append((w*np.sign(dot))[take].astype('f'))
s=np.concatenate(src);t=np.concatenate(dst);w=np.concatenate(weights)
W=coo_matrix((w,(s,t)),shape=(len(a),len(a))).tocsr();degree=np.asarray(abs(W).sum(1)).ravel();W=diags(1/np.maximum(degree,1e-9))@W
original=((b[:,:3]-a[:,:3])*n).sum(1);smooth=original.copy()
for _ in range(8):
    smooth=.15*original+.85*(W@smooth);smooth[degree<2]=original[degree<2]
report=[]
for name,strength in [('mild',.5),('medium',.85)]:
    delta=np.clip(strength*(smooth-original),-.008,.008);result=b.copy();result[:,:3]+=delta[:,None]*n
    assert np.array_equal(result[:,3:],b[:,3:]);np.save(args.out/f'{name}.npy',result)
    report.append(dict(name=name,mean_shift=float(abs(delta).mean()),max_shift=float(abs(delta).max()),
        edges=len(w),iterations=8,anchor=.15,strength=strength,unchanged_colors_opacity_covariance=True))
(args.out/'normal-motion.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
