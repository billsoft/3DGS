"""Use camera territories as proposals, never as permission to delete coverage.

The accepted model is the fallback everywhere. Only supported source changes
are painted on exactly the same Gaussians; this cannot repair wrong geometry.
"""
from common import *
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from coverage_policy import replacement_gate, assert_coverage_unchanged
import argparse, shutil, hashlib

def run():
    parser=argparse.ArgumentParser();parser.add_argument('--minimum-gain',type=float,default=.9)
    parser.add_argument('--evidence-distance',type=float,default=.08)
    args=parser.parse_args()
    baseline=ROOT/'four-room/baselines/2026-11-11-fusion-next'
    proposal=ROOT/'four-room/fusion-territory'
    if OUT.resolve() in [baseline.resolve(),proposal.resolve(),(ROOT/'four-room/fusion-next').resolve()]:
        raise ValueError('Use a separate GS_OUTPUT_DIR for coverage-safe experiments')
    for name in ['appearance-report.json','fusion.json']:
        shutil.copy2(baseline/name,OUT/name)
    old=np.load(baseline/'ownership.npz');new=np.load(proposal/'ownership.npz')
    effective=np.load(proposal/'effective-ownership.npz')['owner']
    geo=np.load(baseline/'geometry.npz');tri=geo['tri'];center=old['center']
    if not np.array_equal(center,new['center']):raise ValueError('Proposal geometry does not match baseline')
    corr=np.load(baseline/'correspondences.npz');reliable=(corr['quality']>.4)&~corr['held']
    evidence,_=cKDTree(corr['target'][reliable]).query(center)
    n=len(center);old_owner=old['owner'];territory_owner=new['owner']
    # A territory winner is only a tie-breaker among similarly good views.
    # The previous experiment forced poorer observations into camera cores.
    proposed=new['scores'].argmax(1)
    valid=territory_owner<4;index=np.flatnonzero(valid)
    near_best=new['scores'][index,territory_owner[index]]>=args.minimum_gain*new['scores'][index].max(1)
    index=index[near_best];proposed[index]=territory_owner[index]
    face_ok=np.zeros(n,bool)
    images=photos()
    for i in range(4):
        ids=np.flatnonzero((proposed==i)&(old_owner<4)&(old_owner!=i)&(effective<4))
        err,conf,dep=cams[i].sample(cams[i].project(center[ids]))
        previous_score=new['scores'][ids,old_owner[ids]]
        diff=np.linalg.norm(new['colors'][ids,i]-old['colors'][ids,old_owner[ids]],axis=1)
        face_ok[ids]=replacement_gate(previous_score,new['scores'][ids,i],diff,evidence[ids],err,dep,conf,args.minimum_gain,args.evidence_distance)
    a,b=adjacency(tri);length=np.linalg.norm(center[a]-center[b],axis=1)
    graph=coo_matrix((np.r_[length,length],(np.r_[a,b],np.r_[b,a])),shape=(n,n)).tocsr()
    final_owner=np.where(face_ok,proposed,old_owner)
    boundary=np.unique(np.r_[a[final_owner[a]!=final_owner[b]],b[final_owner[a]!=final_owner[b]]])
    distance=dijkstra(graph,directed=False,indices=boundary,min_only=True,limit=.08) if len(boundary) else np.full(n,np.inf)
    native=rm.f.prep.read_gs(baseline/'fused.ply');result=native.copy()
    _,face=cKDTree(center).query(native[:,:3]);selected=np.flatnonzero(face_ok[face])
    changed=np.zeros(len(native),bool);weights=np.zeros(len(native),'f')
    for i in range(4):
        ids=selected[proposed[face[selected]]==i]
        for start in range(0,len(ids),50000):
            batch=ids[start:start+50000];pos=native[batch,:3];f=face[batch]
            target=correct_rgb(sample(images[i],uv(project(pos,i)))[:,::-1]/255,i)
            old_rgb=np.clip(native[batch,3:6]*.2820947918+.5,0,1)
            err,conf,dep=cams[i].sample(cams[i].project(pos))
            safe=(np.linalg.norm(target-old_rgb,axis=1)<.14)&(conf>.8)&(abs(err)<.04+.01*dep)
            # Full replacement in the interior; smooth blending is restricted
            # to a narrow, color-consistent transition, never the entire region.
            t=np.clip(distance[f]/.035,0,1);w=t*t*(3-2*t)
            rgb=old_rgb*(1-w[:,None])+target*w[:,None]
            use=safe&(w>0)
            result[batch[use],3:6]=(rgb[use]-.5)/.2820947918
            changed[batch[use]]=True;weights[batch[use]]=w[use]
        print('source',stations[i],'eligible',len(ids),flush=True)
    assert_coverage_unchanged(native,result)
    rm.f.write(OUT/'fused.ply',result);rm.f.to_splat(OUT/'fused.splat',result)
    diagnostic=result.copy();palette=np.array([[.95,.27,.23],[.2,.8,.35],[.25,.5,.98],[.98,.7,.15]])
    diagnostic[:,3:6]=(.65-.5)/.2820947918
    diagnostic[changed,3:6]=(palette[proposed[face[changed]]]-.5)/.2820947918
    rm.f.to_splat(OUT/'owners.splat',diagnostic)
    np.savez_compressed(OUT/'source-updates.npz',face=face,changed=changed,weight=weights,owner=final_owner)
    write_json('coverage-report.json',dict(method='immutable baseline coverage / quality-gated camera source proposals',baseline_sha256=hashlib.sha256((baseline/'fused.ply').read_bytes()).hexdigest(),points=len(result),proposed_faces=int((old_owner!=proposed).sum()),accepted_faces=int(face_ok.sum()),updated_points=int(changed.sum()),minimum_quality_ratio=args.minimum_gain,maximum_evidence_distance=args.evidence_distance,geometry_opacity_covariance_identical=True,deleted_points=0,added_points=0,note='Coverage equality follows from identical ordered geometry and opacity, not an estimated kNN support score. Only colors change. Existing holes and geometric errors cannot be repaired by this pass.'))
    print('updated',int(changed.sum()),'of',len(native),flush=True)

if __name__=='__main__':run()
