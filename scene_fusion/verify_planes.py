"""Check bounded planar quads using a held-out camera before final fitting.

Corner support alone cannot rule out a composite contour spanning several
surfaces. This experimental verifier is not a production geometry certificate.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from core import prep,write_json

SIGNS=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*.5

def rectangle(params):
    frame=Rotation.from_rotvec(params[3:6]).as_matrix()
    return params[:3]+(SIGNS*np.exp(params[6:8]))@frame[:,:2].T

def initial(points):
    center=points.mean(0);u=points[1]-points[0];v=points[3]-points[0];width=np.linalg.norm(u);height=np.linalg.norm(v);u/=width;n=np.cross(u,v);n/=np.linalg.norm(n);v=np.cross(n,u)
    return np.r_[center,Rotation.from_matrix(np.stack([u,v,n],axis=1)).as_rotvec(),np.log([width,height])]

def run(scene,observations,out,free_quads=False,held_mean=3.5,held_max=6.):
    data=json.loads(observations.read_text(encoding='utf8'));obs=data['observations'];S=data['size'];cams=np.load(scene/'aligned-cameras.npz');C,Q=cams['C'],cams['Q'];planes=[];diagnostics=[]
    def project(points,o):
        q=(points-C[o['camera']])@Q[o['camera']]@prep.FACES[o['face']];uv=q[:,:2]/np.maximum(q[:,2,None],1e-6)*S/2+S/2-.5
        return uv,q[:,2]
    def correspond(points):
        found={}
        for k,o in enumerate(obs):
            projected,z=project(points,o)
            if z.min()<=.1:continue
            actual=np.array(o['uv']);best=None
            for reverse in [False,True]:
                for shift in range(4):
                    order=np.roll(np.arange(4)[::-1] if reverse else np.arange(4),shift);err=np.linalg.norm(projected-actual[order],axis=1)
                    if err.mean()<5 and err.max()<10 and (best is None or err.mean()<best[0]):best=(float(err.mean()),dict(observation=k,order=order.tolist(),camera=o['camera']))
            if best and (o['camera'] not in found or best[0]<found[o['camera']][0]):found[o['camera']]=best
        return [v[1] for v in found.values()]
    def fit(params,matches):
        def residual(x):
            points=geometry(x);return np.concatenate([(project(points,obs[m['observation']])[0]-np.array(obs[m['observation']]['uv'])[m['order']]).ravel() for m in matches])
        return least_squares(residual,params,loss='soft_l1',f_scale=1.5,max_nfev=150).x
    def geometry(x):
        if not free_quads:return rectangle(x)
        points=x.reshape(4,3);center=points.mean(0);normal=np.linalg.svd(points-center)[2][-1]
        return points-((points-center)@normal)[:,None]*normal
    for pair in sorted(data['pairs'],key=lambda e:e['score']):
        points=np.array(pair['points']);center=points.mean(0);extent=np.linalg.norm(points-center,axis=1).mean()
        if any(np.linalg.norm(center-np.mean(p['corners'],axis=0))<extent*.15 and abs(np.dot(pair['normal'],p['normal']))>.98 for p in planes):continue
        matches=correspond(points)
        record=dict(pair=[pair['a'],pair['b']],cameras=[m['camera'] for m in matches],center=center.tolist());diagnostics.append(record)
        if len(matches)<3:continue
        params=points.ravel() if free_quads else initial(points);trial=fit(params,matches[:-1]);held=matches[-1];err=np.linalg.norm(project(geometry(trial),obs[held['observation']])[0]-np.array(obs[held['observation']]['uv'])[held['order']],axis=1)
        record.update(held_mean=float(err.mean()),held_max=float(err.max()))
        if err.max()>held_max or err.mean()>held_mean:continue
        final=fit(trial,matches);corners=geometry(final);errors=[]
        for m in matches:errors.extend(np.linalg.norm(project(corners,obs[m['observation']])[0]-np.array(obs[m['observation']]['uv'])[m['order']],axis=1).tolist())
        if max(errors)>7 or np.mean(errors)>3:continue
        normal=np.linalg.svd(corners-corners.mean(0))[2][-1]
        planes.append(dict(corners=corners.tolist(),normal=normal.tolist(),width=float(np.linalg.norm(corners[1]-corners[0])),height=float(np.linalg.norm(corners[3]-corners[0])),cameras=[m['camera'] for m in matches],observations=matches,held_camera=held['camera'],held_mean_px=float(err.mean()),held_max_px=float(err.max()),mean_px=float(np.mean(errors)),max_px=float(max(errors))))
    write_json(out,dict(planes=planes,diagnostics=diagnostics,source='multi-camera quadrilateral corners',semantic_labels=False,manual_coordinates=False,size=S,free_quads=free_quads,held_mean_limit=held_mean,held_max_limit=held_max));print('Verified planes',len(planes),flush=True)
    for i,p in enumerate(planes):print(i,'cams',p['cameras'],'center',np.round(np.mean(p['corners'],axis=0),2),'size',round(p['width'],2),round(p['height'],2),'held',round(p['held_mean_px'],2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--observations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--free-quads',action='store_true');p.add_argument('--held-mean',type=float,default=3.5);p.add_argument('--held-max',type=float,default=6.);a=p.parse_args();run(a.scene,a.observations,a.out,a.free_quads,a.held_mean,a.held_max)
