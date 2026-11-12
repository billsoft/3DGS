"""Validated degree-zero Gaussian PLY IO and scale-independent initialization."""
from pathlib import Path
import numpy as np

FIELDS='x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3'.split()
TYPES={'float':'f4','float32':'f4','double':'f8','float64':'f8','uchar':'u1','uint8':'u1','char':'i1','int8':'i1','short':'i2','int16':'i2','ushort':'u2','uint16':'u2','int':'i4','int32':'i4','uint':'u4','uint32':'u4'}

def header(stream):
    if stream.readline().strip()!=b'ply':raise ValueError('Not a PLY file')
    fmt=None;count=None;props=[];element=None
    for _ in range(10000):
        line=stream.readline()
        if not line:raise ValueError('Truncated PLY header')
        tokens=line.decode('ascii').split()
        if not tokens:continue
        if tokens[0]=='end_header':break
        if tokens[0]=='format':fmt=tokens[1]
        if tokens[0]=='element':
            element=tokens[1]
            if count is None and element!='vertex' and int(tokens[2])>0:raise ValueError('Vertex data must precede other nonempty PLY elements')
            if element=='vertex':count=int(tokens[2])
        if tokens[0]=='property' and element=='vertex':
            if tokens[1]=='list' or tokens[1] not in TYPES:raise ValueError('Unsupported vertex property type')
            props.append((tokens[2],TYPES[tokens[1]]))
    else:raise ValueError('PLY header exceeds limit')
    if fmt not in {'ascii','binary_little_endian','binary_big_endian'} or count is None or count<1:raise ValueError('Invalid PLY format or vertex count')
    if len({n for n,t in props})!=len(props):raise ValueError('Duplicate PLY vertex properties')
    return fmt,count,props

def read_gaussians(path):
    with Path(path).open('rb') as f:
        fmt,count,props=header(f);names=[n for n,t in props]
        missing=set(FIELDS)-set(names)
        if missing:raise ValueError(f'{Path(path).name}: not a supported Gaussian PLY; missing {sorted(missing)}')
        if fmt=='ascii':
            values=np.loadtxt(f,max_rows=count,ndmin=2)
            if values.shape!=(count,len(props)):raise ValueError('Wrong ASCII PLY vertex shape')
            columns={n:values[:,i] for i,n in enumerate(names)}
        else:
            endian='<' if fmt=='binary_little_endian' else '>'
            data=np.fromfile(f,dtype=np.dtype([(n,endian+t) for n,t in props]),count=count)
            if len(data)!=count:raise ValueError('Truncated binary PLY vertex data')
            columns={n:data[n] for n in names}
        sh=[n for n in names if n.startswith('f_rest_')]
        if any(np.any(columns[n]!=0) for n in sh):
            raise ValueError('Nonzero higher-order SH is not supported by this exporter; explicit conversion is required (no silent coefficient loss)')
        a=np.stack([columns[n] for n in FIELDS],axis=1).astype('f')
    if not np.isfinite(a).all():raise ValueError('Non-finite Gaussian values')
    q=np.linalg.norm(a[:,10:14],axis=1)
    if np.any(q<1e-8):raise ValueError('Zero Gaussian quaternion')
    a[:,10:14]/=q[:,None]
    if np.max(a[:,7:10])>70 or np.min(a[:,7:10])<-70:raise ValueError('Unusable Gaussian scale range')
    return a

def normalize(a):
    sample=a[::max(1,len(a)//100000),:3].astype('d');center=np.median(sample,axis=0)
    radius=float(np.median(np.linalg.norm(sample-center,axis=1)))
    if not np.isfinite(radius) or radius<1e-12:raise ValueError('Degenerate scene extent')
    # Preserve the usual source origin; remove very large coordinate offsets.
    origin=center if np.linalg.norm(center)>radius*3 else np.zeros(3)
    scale=5./radius;b=a.copy();b[:,:3]=(a[:,:3].astype('d')-origin)*scale;b[:,7:10]+=np.log(scale)
    if not np.isfinite(b).all():raise ValueError('Scene normalization overflow')
    return b,dict(origin=origin.tolist(),scale=scale,robust_radius=radius,normalized_radius=5.,units='arbitrary normalized units')
