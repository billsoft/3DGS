"""Spherical photo bundle adjustment; generated depth is initialization only.

Keep one panorama pose and its longest baseline length fixed (gauge). Hold out
whole feature tracks, not observations of tracks seen in training.
"""
import sys,json,importlib.util,argparse
from pathlib import Path
import numpy as np, cv2 as cv
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from pose_graph import bundle_gate,validated_update
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('fusion',ROOT/'four-fuse.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();D=args.out;OUT=D/'photo-bundle';OUT.mkdir(exist_ok=True)
scene=json.loads((D/'scene.json').read_text(encoding='utf8'));photos={s['id']:Path(s['photo']) for s in scene['stations']}
reg=json.loads((D/'registration.json').read_text(encoding='utf8'));ns=reg['stations'];ms=np.array(reg['matrices'])
Cs=[];Qs=[];fs=[]
for n,m in zip(ns,ms):
    cache=OUT/f'features-{n}.npz'
    # Whole-stage checkpoints handle reuse, including feature-code changes.
    im=cv.imdecode(np.fromfile(photos[n],np.uint8),1)
    im=cv.resize(im,(3840,1920));uv,desc=f.prep.features(im,cv.SIFT_create(nfeatures=30000,contrastThreshold=.012))
    lon=(uv[:,0]/3840-.5)*2*np.pi;lat=uv[:,1]/1920*np.pi
    ray=np.stack([np.cos(lon)*np.sin(lat),-np.sin(lon)*np.sin(lat),np.cos(lat)],1)
    np.savez(cache,uv=uv,desc=desc,ray=ray)
    fs.append(np.load(cache));c=np.load(D/n/'cache.npz');s=np.cbrt(np.linalg.det(m[:3,:3]))
    Cs.append(m[:3,:3]@c['C']+m[:3,3]);Qs.append(m[:3,:3]/s@c['R'].T)
C0=np.array(Cs);Q0=np.array(Qs)
baseline=float(np.max(np.linalg.norm(C0-C0[0],axis=1)))
if not np.isfinite(baseline) or baseline<=1e-10:raise ValueError('Camera baseline is degenerate')
# Translation and point parameters are normalized by the scene's own baseline.
origin=C0[0].copy();C0=(C0-origin)/baseline

def triangulate(obs,C,Q):
    cams=np.array([a for a,b in obs]);r=np.array([fs[a]['ray'][b] for a,b in obs])
    v=np.einsum('nij,nj->ni',Q[cams],r);A=np.eye(3)[None]-v[:,:,None]*v[:,None,:]
    p=np.linalg.lstsq(A.sum(0),np.einsum('nij,nj->ni',A,C[cams]).sum(0),rcond=None)[0]
    d=p-C[cams];t=np.einsum('ij,ij->i',d,v);pred=d/np.linalg.norm(d,axis=1)[:,None]
    err=np.linalg.norm(pred-v,axis=1)
    angle=np.rad2deg(np.arccos(np.clip(v@v.T,-1,1))).max()
    return p,err,t,angle

parent={}
def find(a):
    parent.setdefault(a,a)
    if parent[a]!=a:parent[a]=find(parent[a])
    return parent[a]
for i in range(len(ns)):
    for j in range(i+1,len(ns)):
        ab=f.prep.matches(fs[i]['desc'],fs[j]['desc'],.76,True)
        ba=dict(f.prep.matches(fs[j]['desc'],fs[i]['desc'],.76,True));count=0
        for a,b in ab:
            if ba.get(b)!=a:continue
            obs=[(i,int(a)),(j,int(b))];p,e,t,angle=triangulate(obs,C0,Q0)
            if e.max()<.025 and t.min()>.02 and angle>1.5 and t.max()<100:
                parent[find(obs[1])]=find(obs[0]);count+=1
        print('pair',ns[i],ns[j],count,flush=True)
groups={}
for a in list(parent):groups.setdefault(find(a),[]).append(a)
tracks=[];points=[]
for obs in groups.values():
    if len(set(a for a,b in obs))!=len(obs):continue
    p,e,t,angle=triangulate(obs,C0,Q0)
    if e.max()<.025 and t.min()>.02 and angle>1.5 and t.max()<100:tracks.append(obs);points.append(p)
points=np.array(points);rng=np.random.default_rng(20261111);hold=rng.random(len(tracks))<.2
if len(tracks)<30:raise ValueError('Too few verified multiview tracks for bundle adjustment')
train=np.flatnonzero(~hold);obs_c=[];obs_p=[];obs_r=[]
for k,idx in enumerate(train):
    for c,b in tracks[idx]:obs_c.append(c);obs_p.append(k);obs_r.append(fs[c]['ray'][b])
oc=np.array(obs_c);op=np.array(obs_p);rr=np.array(obs_r)
if np.any(np.bincount(oc,minlength=len(ns))<12):raise ValueError('A camera has insufficient bundle observations')
gauge_peer=int(np.argmax(np.linalg.norm(C0-C0[0],axis=1)))
base=np.linalg.norm(C0[gauge_peer]-C0[0]);N=len(train);camera_parameters=6*(len(ns)-1)
if base<1e-4:raise ValueError('Camera baseline is degenerate; distinct capture positions are required')
def unpack(x):
    C=C0.copy();C[1:]+=x[:camera_parameters].reshape(len(ns)-1,6)[:,3:]
    Q=Q0.copy();Q[1:]=Rotation.from_rotvec(x[:camera_parameters].reshape(len(ns)-1,6)[:,:3]).as_matrix()@Q0[1:]
    return C,Q,x[camera_parameters:].reshape(-1,3)
def residual(x):
    C,Q,P=unpack(x);d=P[op]-C[oc];d/=np.linalg.norm(d,axis=1)[:,None]
    ray=np.einsum('nij,nj->ni',Q[oc],rr)
    return np.r_[((d-ray)*weights[:,None]).ravel(),(np.linalg.norm(C[gauge_peer]-C[0])-base)*10]
sp=lil_matrix((3*len(oc)+1,camera_parameters+3*N),dtype=int)
for k,(c,p) in enumerate(zip(oc,op)):
    sp[3*k:3*k+3,camera_parameters+3*p:camera_parameters+3*p+3]=1
    if c:sp[3*k:3*k+3,6*(c-1):6*c]=1
sp[-1,6*(gauge_peer-1)+3:6*gauge_peer]=1
x=np.r_[np.zeros(camera_parameters),points[train].ravel()]
weights=np.ones(len(oc))
print('tracks',len(tracks),'train',N,'hold',int(hold.sum()),flush=True)
sol=least_squares(residual,x,jac_sparsity=sp.tocsr(),loss='soft_l1',f_scale=.002,max_nfev=500,verbose=1,ftol=1e-5,tr_options={'atol':1e-8,'btol':1e-8})
# Estimate robust track weights from TRAINING residuals only. A coherent track
# stays one unit; its held-out siblings can never leak into the fitted weights.
rounds=[]
for iteration in range(2):
    cc,qq,pp=unpack(sol.x);dd=pp[op]-cc[oc];dd/=np.maximum(np.linalg.norm(dd,axis=1)[:,None],1e-12)
    errors=np.linalg.norm(dd-np.einsum('nij,nj->ni',qq[oc],rr),axis=1)
    track_error=np.array([np.sqrt(np.mean(errors[op==k]**2)) for k in range(N)])
    med=np.median(track_error);sigma=max(1./(3840/(2*np.pi)),med+2.5*1.4826*np.median(abs(track_error-med)))
    track_weight=1/(1+(track_error/sigma)**2)
    weights=np.sqrt(track_weight[op])
    counts=np.bincount(oc,weights=track_weight[op],minlength=len(ns))
    if counts.min()<12:raise ValueError('Robust bundle has insufficient effective camera support')
    rounds.append(dict(angular_scale=float(sigma),downweighted_tracks=int((track_weight<.5).sum()),effective_observations=counts.tolist()))
    sol=least_squares(residual,sol.x,jac_sparsity=sp.tocsr(),loss='soft_l1',f_scale=.002,max_nfev=500,verbose=1,ftol=1e-5,tr_options={'atol':1e-8,'btol':1e-8})
C,Q,P=unpack(sol.x)
def held_errors(cc,qq):
    result=[[] for _ in ns]
    for k in np.flatnonzero(hold):
        _,es,depths,_=triangulate(tracks[k],cc,qq)
        for (camera,_),error,depth in zip(tracks[k],es,depths):result[camera].append(float(error*3840/(2*np.pi)) if depth>0 else float('inf'))
    return result
try:
    C,Q,selection=validated_update(C0,Q0,C,Q,held_errors,bool(sol.success))
except ValueError as error:
    (OUT/'report.json').write_text(json.dumps(dict(acceptance=dict(passed=False,reasons=[str(error)])),indent=2))
    raise
# Backtracking also restores the exact scale gauge, even at fraction 1.
# Retriangulate to keep depth anchors in the selected cameras' coordinate frame.
P=np.array([triangulate(tracks[k],C,Q)[0] for k in train])
report={'tracks':len(tracks),'train_tracks':N,'held_tracks':int(hold.sum()),'gauge':'first camera fixed; longest initial baseline from it fixed; scale is arbitrary','gauge_peer':ns[gauge_peer], 'optimizer_success':bool(sol.success),'optimizer_message':sol.message}
report['update_selection']=selection
report['validation_role']='Whole withheld tracks are used for acceptance/backtracking, not optimization residuals; these scores are model-selection validation, not an unbiased test.'
for label,cc,qq in [('before',C0,Q0),('after',C,Q)]:
    for split,sel in [('train',~hold),('held',hold)]:
        es=[]
        for k in np.flatnonzero(sel):
            p,e,t,ang=triangulate(tracks[k],cc,qq);es.extend(e*3840/(2*np.pi))
        report[label+'_'+split]={'median_equivalent_px':float(np.median(es)),'p90_equivalent_px':float(np.quantile(es,.9)),'observations':len(es)}
before_cam=held_errors(C0,Q0);after_cam=held_errors(C,Q)
report['acceptance']=bundle_gate(before_cam,after_cam,bool(sol.success))
report['robust_rounds']=rounds
report['per_camera_held']=[dict(id=n,observations=len(a),before_median=float(np.median(b)),after_median=float(np.median(a)),before_p90=float(np.quantile(b,.9)),after_p90=float(np.quantile(a,.9))) for n,b,a in zip(ns,before_cam,after_cam) if len(a)]
report['camera_translation_change_model_units']=(np.linalg.norm(C-C0,axis=1)*baseline).tolist()
report['camera_rotation_change_deg']=np.rad2deg(Rotation.from_matrix(Q@Q0.transpose(0,2,1)).magnitude()).tolist()
(OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if not report['acceptance']['passed']:raise ValueError('Camera validation rejected: '+'; '.join(report['acceptance']['reasons']))
np.savez(OUT/'bundle.npz',C=C*baseline+origin,Q=Q,C0=C0*baseline+origin,Q0=Q0,points=P*baseline+origin,train=train,hold=hold)
(OUT/'tracks.json').write_text(json.dumps(tracks))
