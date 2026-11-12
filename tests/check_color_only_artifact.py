"""Verify a real training output did not change accepted geometry or colors."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--protected',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
x,y=np.load(a.before),np.load(a.after);mask=np.load(a.protected)
assert x.shape==y.shape and np.isfinite(y).all() and mask.shape==(len(x),)
assert np.array_equal(x[:,:3],y[:,:3]);assert np.array_equal(x[:,6:10],y[:,6:10])
rot_error=0.
for start in range(0,len(x),30000):
    ix=slice(start,start+30000);R=Rotation.from_quat(x[ix][:,[11,12,13,10]]).as_matrix();S=Rotation.from_quat(y[ix][:,[11,12,13,10]]).as_matrix();rot_error=max(rot_error,float(abs(R-S).max()))
assert rot_error<2e-6
color_error=float(abs(x[mask,3:6]-y[mask,3:6]).max()) if mask.any() else 0.
assert color_error<2e-6
report=dict(points=len(x),positions_scales_opacity_exact=True,max_rotation_matrix_error=rot_error,
            protected_points=int(mask.sum()),max_protected_SH_error=color_error,
            candidate_sha256=hashlib.sha256(a.after.read_bytes()).hexdigest(),passed=True)
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
