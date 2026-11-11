"""Rollback proposed colors that worsen credible original photo observations."""
from common import *
from coverage_policy import assert_coverage_unchanged,photometric_accept
import shutil

def run():
    source=ROOT/'four-room/fusion-coverage-wide'
    baseline=ROOT/'four-room/baselines/2026-11-11-fusion-next'
    if OUT.resolve() in [source.resolve(),baseline.resolve(),(ROOT/'four-room/fusion-next').resolve()]:
        raise ValueError('Use a separate output directory')
    for name in ['appearance-report.json','fusion.json']:
        shutil.copy2(baseline/name,OUT/name)
    before=rm.f.prep.read_gs(baseline/'fused.ply');candidate=rm.f.prep.read_gs(source/'fused.ply')
    assert_coverage_unchanged(before,candidate)
    updates=np.load(source/'source-updates.npz');ids=np.flatnonzero(updates['changed'])
    face=updates['face'][ids];labels=np.load(ROOT/'four-room/fusion-territory/ownership.npz')
    old_rgb=np.clip(before[ids,3:6]*.2820947918+.5,0,1);new_rgb=np.clip(candidate[ids,3:6]*.2820947918+.5,0,1)
    scores=labels['scores'][face];delta=np.zeros((len(ids),4));valid=np.zeros((len(ids),4),bool)
    images=photos()
    for i in range(4):
        pos=before[ids,:3];err,conf,dep=cams[i].sample(cams[i].project(pos))
        valid[:,i]=(scores[:,i]>.3*scores.max(1))&(scores[:,i]>1e-6)&(conf>.8)&(abs(err)<.04+.01*dep)
        observed=correct_rgb(sample(images[i],uv(project(pos,i)))[:,::-1]/255,i)
        delta[:,i]=abs(new_rgb-observed).mean(1)-abs(old_rgb-observed).mean(1)
    # Every credible view has a vote; no camera wins just by being closer.
    keep,mean,worst=photometric_accept(delta,valid)
    candidate[ids[~keep],3:6]=before[ids[~keep],3:6]
    assert_coverage_unchanged(before,candidate)
    rm.f.write(OUT/'fused.ply',candidate);rm.f.to_splat(OUT/'fused.splat',candidate)
    diagnostic=candidate.copy();diagnostic[:,3:6]=(.65-.5)/.2820947918;diagnostic[ids[keep],3:6]=(np.array([.2,.9,.35])-.5)/.2820947918
    rm.f.to_splat(OUT/'owners.splat',diagnostic)
    np.savez_compressed(OUT/'rollback.npz',ids=ids,kept=keep,valid=valid,mean_delta=mean,worst_delta=worst)
    write_json('coverage-report.json',dict(method='immutable baseline coverage / per-point multi-view photometric rollback',points=len(candidate),proposed_updated_points=len(ids),updated_points=int(keep.sum()),rolled_back_points=int((~keep).sum()),geometry_opacity_covariance_identical=True,deleted_points=0,added_points=0,minimum_credible_views=2,maximum_per_view_mae_increase=.005,required_mean_mae_improvement=.0005,note='Uses original input photos as optimization constraints, not held-out truth. Original holes and geometry errors remain. View-dependent reflection and imperfect source geometry can prevent any reliable source replacement.'))
    print('kept',int(keep.sum()),'rolled back',int((~keep).sum()),flush=True)

if __name__=='__main__':run()
