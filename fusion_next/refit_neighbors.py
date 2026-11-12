"""Local surface graph for coherent color/position changes, without semantic masks."""
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import argparse

parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path('four-room/refit-data'))
folder=parser.parse_args().data;a=np.load(folder/'initial.npy')
tree=cKDTree(a[:,:3]);rgb=np.clip(.5+.2820947918*a[:,3:6],0,1)
norm=Rotation.from_quat(a[:,[11,12,13,10]]).as_matrix()[:,:,0]
chunks=[]
for start in range(0,len(a),100000):
    ids=np.arange(start,min(start+100000,len(a)))
    d,ix=tree.query(a[ids,:3],k=7,workers=4)
    src=np.repeat(ids,6);dst=ix[:,1:].reshape(-1);dist=d[:,1:].reshape(-1)
    aligned=abs((norm[src]*norm[dst]).sum(1))
    diff=np.linalg.norm(rgb[src]-rgb[dst],axis=1)
    weight=np.exp(-(dist/.025)**2-(diff/.08)**2)*np.clip((aligned-.8)/.2,0,1)**2
    keep=(src<dst)&(weight>.05)&(dist<.04)
    chunks.append((src[keep].astype('i'),dst[keep].astype('i'),weight[keep].astype('f')))
np.savez_compressed(folder/'neighbors.npz',src=np.concatenate([v[0] for v in chunks]),
    dst=np.concatenate([v[1] for v in chunks]),weight=np.concatenate([v[2] for v in chunks]))
print('Surface graph edges',sum(len(v[0]) for v in chunks),flush=True)
