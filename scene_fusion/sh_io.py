"""Full SH Gaussian PLY IO. Never route these models through DC-only SPLAT."""
from pathlib import Path
import numpy as np
from input_io import FIELDS,header

def write_sh_ply(path,base,sh):
    if base.shape!=(len(sh),14) or sh.ndim!=3 or sh.shape[2]!=3 or sh.shape[1] not in [4,9,16]:raise ValueError('Invalid SH model shape')
    if not np.isfinite(base).all() or not np.isfinite(sh).all():raise ValueError('Nonfinite SH model')
    # INRIA f_rest ordering is channel-major (all R, then all G, then all B).
    names=FIELDS+[f'f_rest_{i}' for i in range((sh.shape[1]-1)*3)]
    rows=np.concatenate([base,sh[:,1:,:].transpose(0,2,1).reshape(len(base),-1)],axis=1).astype('<f4')
    rows[:,3:6]=sh[:,0,:]
    text='ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(base))+'\n'+''.join('property float '+n+'\n' for n in names)+'end_header\n'
    with Path(path).open('wb') as f:f.write(text.encode('ascii'));rows.tofile(f)

def read_sh_ply(path):
    with Path(path).open('rb') as f:
        fmt,count,props=header(f)
        if fmt!='binary_little_endian':raise ValueError('SH roundtrip reader expects binary little endian')
        rows=np.fromfile(f,dtype=np.dtype([(n,'<'+t) for n,t in props]),count=count)
    if len(rows)!=count:raise ValueError('Truncated SH model')
    names=[n for n,t in props];rest=[n for n in names if n.startswith('f_rest_')]
    if len(rest) not in [9,24,45] or set(rest)!={f'f_rest_{i}' for i in range(len(rest))}:raise ValueError('Invalid SH fields')
    base=np.stack([rows[n] for n in FIELDS],axis=1);K=len(rest)//3+1
    sh=np.empty((count,K,3),'f');sh[:,0]=base[:,3:6];sh[:,1:]=np.stack([rows[f'f_rest_{i}'] for i in range(len(rest))],1).reshape(count,3,K-1).transpose(0,2,1)
    return base,sh
