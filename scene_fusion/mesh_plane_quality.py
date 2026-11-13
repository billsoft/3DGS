"""Diagnose surface agreement with estimated planes, never geometric ground truth."""
import argparse,json
from pathlib import Path
import numpy as np

def run(a):
    planes=json.loads(a.planes.read_text('utf8'));tolerance=4*planes['spacing'];records=[]
    for path in a.mesh:
        z=np.load(path);p=z['p'];tri=z['tri'];center=p[tri].mean(1)
        cross=np.cross(p[tri[:,1]]-p[tri[:,0]],p[tri[:,2]]-p[tri[:,0]]);area=np.linalg.norm(cross,axis=1);normal=cross/np.maximum(area[:,None],1e-12)
        rows=[]
        for i,plane in enumerate(planes['planes']):
            n=np.array(plane['normal']);distance=abs(center@n+plane['offset']);cos=abs(normal@n)
            valid=(distance<tolerance)&(cos>.8)&(area>1e-12)
            if not valid.any():continue
            w=area[valid];angle=np.arccos(np.clip(cos[valid],0,1))
            rows.append(dict(plane=i,area=float(w.sum()/2),faces=int(valid.sum()),distance_rms=float(np.sqrt(np.average(distance[valid]**2,weights=w))),normal_rms_degrees=float(np.rad2deg(np.sqrt(np.average(angle**2,weights=w))))))
        records.append(dict(mesh=str(path),planes=rows));print(path.parent.name,rows[:3],flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(dict(records=records,tolerance=tolerance,warning='Estimated planes are priors, not ground truth; compare area/support as well as residuals.'),indent=2),encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--planes',type=Path,required=True);p.add_argument('--mesh',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
