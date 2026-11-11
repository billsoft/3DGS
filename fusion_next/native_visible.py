"""Native owner fusion with conservative one-sided self-occlusion rejection."""
from common import *
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

def run():
    source=ROOT/'four-room/fusion-native-fidelity';labels=np.load(ROOT/'four-room/fusion-territory/ownership.npz')
    tree=cKDTree(labels['center']);parts=[];report=[]
    for i,station in enumerate(stations):
        a=rm.f.prep.read_gs(source/f'native-{station}.ply');_,face=tree.query(a[:,:3])
        cam=rm.Camera(station,np.linalg.inv(cams[i].inv));cam.maps=[]
        for k in range(6):
            d=np.load(rm.room/station/f'depth-{k}.npz');cam.maps.append((d['depth'],d['alpha']))
        err,conf,dep=cam.sample(cam.project(a[:,:3]));ray=a[:,:3]-C[i];ray/=np.maximum(np.linalg.norm(ray,axis=1)[:,None],1e-9)
        rotation=Rotation.from_quat(a[:,[11,12,13,10]]).as_matrix();local=np.einsum('ni,nij->nj',ray,rotation)
        sigma=np.sqrt(np.sum(local**2*np.exp(2*a[:,7:10]),axis=1))
        hidden=(conf>.98)&(err<-(.04+.01*dep+4*sigma))
        keep=(labels['owner'][face]==i)&~hidden
        parts.append(a[keep]);report.append(dict(station=station,core_points=int((labels['owner'][face]==i).sum()),kept=int(keep.sum()),hidden_removed=int(((labels['owner'][face]==i)&hidden).sum())))
    result=np.concatenate(parts);rm.f.write(OUT/'fused.ply',result);rm.f.to_splat(OUT/'fused.splat',result)
    write_json('native-visible-report.json',dict(sources=report,output_points=len(result),note='One-sided depth rejection at four radial sigmas. Kept Gaussian shape, opacity and colors are untouched. Ownership still depends on existing calibrated surface and does not guarantee native-view fidelity.'))
    write_json('fusion.json',dict(centers=C.tolist(),up=rm.meta['up'],stations=stations))

if __name__=='__main__':run()
