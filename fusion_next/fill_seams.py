"""Fill under-covered narrow seams using the same supported common surface."""
from common import *
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

def run():
    native=rm.f.prep.read_gs(OUT/'anchored-photo-fused.ply')
    surface=rm.f.prep.read_gs(OUT/'fused.ply');labels=np.load(OUT/'ownership.npz')
    effective=np.load(OUT/'effective-ownership.npz')['owner']
    _,face=cKDTree(labels['center']).query(surface[:,:3])
    candidates=np.flatnonzero((labels['distance'][face]<.055)&(effective[face]<4))
    tree=cKDTree(native[:,:3]);rot=Rotation.from_quat(native[:,[11,12,13,10]]).as_matrix()
    inv=(rot*np.exp(-2*native[:,None,7:10]))@rot.transpose(0,2,1)
    opacity=1/(1+np.exp(-np.clip(native[:,6],-30,30)));fills=[];coverage_values=[]
    for start in range(0,len(candidates),10000):
        ids=candidates[start:start+10000];pos=surface[ids,:3]
        _,neighbor=tree.query(pos,k=min(16,len(native)))
        delta=pos[:,None]-native[neighbor,:3]
        squared=np.einsum('nki,nkij,nkj->nk',delta,inv[neighbor],delta)
        alpha=opacity[neighbor]*np.exp(-.5*np.maximum(squared,0))
        coverage=1-np.prod(1-np.clip(alpha,0,.999999),axis=1)
        use=coverage<.85;fill=surface[ids[use]].copy()
        required=np.clip(1-.02/np.maximum(1-coverage[use],1e-6),.1,.98)
        fill[:,6]=np.log(required/(1-required));fills.append(fill);coverage_values.extend(coverage[use].tolist())
    fill=np.concatenate(fills) if fills else np.empty((0,14),'f')
    result=np.concatenate([native,fill]);rm.f.write(OUT/'hybrid-fused.ply',result);rm.f.to_splat(OUT/'hybrid-fused.splat',result)
    write_json('seam-fill-report.json',dict(native_points=len(native),candidate_points=len(candidates),added_points=len(fill),output_points=len(result),band_model_units=.055,median_estimated_coverage_before=float(np.median(coverage_values)) if coverage_values else None,note='Coverage estimate evaluates local 3D Gaussian support, not an exact screen-space raster. Supplemental points come only from supported common-surface seam bands, never unobserved backsides.'))
    print('seam support',len(native),len(fill),len(result),flush=True)

if __name__=='__main__':run()
