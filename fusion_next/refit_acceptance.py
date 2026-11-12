"""Check an exported candidate at both resolutions; visual review remains required."""
from pathlib import Path
import json, hashlib, argparse
import numpy as np


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--result',default='refit-coherent-hd')
    folder=Path('four-room')/parser.parse_args().result
    candidate=np.load(folder/'optimized.npy')
    initial=np.load('four-room/refit-data/initial.npy',mmap_mode='r')
    assert candidate.shape==initial.shape and np.isfinite(candidate).all()
    assert np.array_equal(candidate[:,6],initial[:,6])
    movement=np.linalg.norm(candidate[:,:3]-initial[:,:3],axis=1)
    scale=np.exp(candidate[:,7:10]-initial[:,7:10])
    assert movement.max()<=.045001 and scale.min()>=.79999 and scale.max()<=1.25001
    checksum=hashlib.sha256((folder/'optimized.npy').read_bytes()).hexdigest()
    results=[]
    for name,size in [('eval',1024),('eval-low',512)]:
        report=json.loads((folder/name/'evaluation.json').read_text())
        assert report['candidate_sha256']==checksum, 'Evaluation is not for the current model'
        s=report['summary']
        assert s['candidate_mean_ssim']>s['baseline_mean_ssim']
        assert s['improved_ssim_views']>=.9*s['source_views']
        assert s['worst_ssim_delta']>=-.01
        assert max(r['new_holes'] for r in report['views'])<=size*size*.0005
        results.append(dict(size=size,summary=s))
    export=json.loads((folder/'export-check.json').read_text())
    assert export['ply_roundtrip_exact'] and export['opacity_unchanged']
    result=dict(numeric_checks_passed=True,visual_review_required=True,
        evaluated_array_sha256=checksum,ply_sha256=hashlib.sha256((folder/'fused.ply').read_bytes()).hexdigest(),
        splat_sha256=hashlib.sha256((folder/'fused.splat').read_bytes()).hexdigest(),
        max_movement=float(movement.max()),scale_ratio_range=[float(scale.min()),float(scale.max())],
        resolutions=results,note='Fits reuse source images; no independent novel-view ground truth.')
    (folder/'acceptance.json').write_text(json.dumps(result,indent=2))
    print('Both resolutions and exact PLY export passed; visual review still required.')


if __name__=='__main__':run()
