"""Automatic regression gate, not a claim of perfect geometry or novel-view truth."""
import argparse,json
from pathlib import Path
import numpy as np
from core import write_json
from run_state import digest

def assess(bundle,native,evaluation):
    reasons=[];n=native['summary'];e=evaluation['summary'];rows=native['views']
    if not rows or n['native_high_alpha_pixels']<=0 or not np.isfinite([n['fused_ssim'],n['native_ssim'],e['baseline_mean_ssim']]).all():reasons.append('Missing or non-finite fidelity evidence')
    if any(not np.isfinite([r['native']['ssim'],r['fused']['ssim']]).all() or r['native']['mask_pixels']<=0 for r in rows):reasons.append('Invalid source-direction evidence')
    if bundle['held_tracks']<8:reasons.append('Fewer than 8 held-out geometry tracks')
    before=bundle['before_held']['median_equivalent_px'];after=bundle['after_held']['median_equivalent_px']
    if not np.isfinite(after) or after>max(before*1.25,before+.5) or after>3:reasons.append('Held-out camera alignment error too high or regressed')
    if n['fused_ssim']<n['native_ssim']-.02:reasons.append('Mean source fidelity fell by more than 0.02 SSIM')
    if any(r['fused']['ssim']<r['native']['ssim']-.08 for r in rows):reasons.append('A source direction lost more than 0.08 SSIM')
    loss=n['newly_low_alpha_pixels']/max(1,n['native_high_alpha_pixels'])
    if loss>.001:reasons.append('More than 0.1% of native high-coverage pixels became low-alpha')
    if any(r['fused']['holes']/max(1,r['native']['mask_pixels'])>.01 for r in rows):reasons.append('A source direction lost more than 1% coverage')
    if not np.isfinite(e['candidate_mean_ssim']) or e['candidate_mean_ssim']<e['baseline_mean_ssim']-.005:reasons.append('Fitting score regressed from the initialization')
    return dict(passed=not reasons,reasons=reasons,native_coverage_loss_fraction=loss,
        limitations='Automatic regression checks only; unseen regions, reflections and intermediate-view geometry still require visual review.')

def run(out,candidate):
    native=json.loads((out/'fit-hd/native-1024/audit.json').read_text(encoding='utf8'));evaluation=json.loads((out/'fit-hd/eval-1024/evaluation.json').read_text(encoding='utf8'))
    # Folder names are stable stage roles; actual resolution is in quality metadata.
    bundle=json.loads((out/'photo-bundle/report.json').read_text(encoding='utf8'));result=assess(bundle,native,evaluation)
    sha=digest(candidate)
    if evaluation.get('candidate_sha256')!=sha or native.get('candidate_sha256')!=sha:
        result['passed']=False;result['reasons'].append('Validation belongs to a different candidate file')
    result['candidate_sha256']=sha;write_json(out/'quality.json',result);print(json.dumps(result),flush=True)
    if not result['passed']:raise ValueError('Quality gate rejected candidate: '+'; '.join(result['reasons']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args();run(a.out,a.candidate)
