"""Verify the fixed-center shape experiment and retain aggregate evidence only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def run():
    p = argparse.ArgumentParser()
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    old, new = np.load(a.initial), np.load(a.candidate)
    assert old.shape == new.shape and new.shape[1] == 14
    assert np.isfinite(new).all()
    assert np.array_equal(old[:, :3], new[:, :3]), 'Centers changed'
    norm = np.linalg.norm(new[:, 10:14], axis=1)
    assert np.max(np.abs(norm - 1)) < 1e-5
    ratio = np.exp(new[:, 7:10] - old[:, 7:10])
    assert ratio.min() >= .79999 and ratio.max() <= 1.25001
    assert new[:, 6].min() >= -2 and new[:, 6].max() <= 8
    q0 = old[:, 10:14] / np.linalg.norm(old[:, 10:14], axis=1)[:, None]
    q1 = new[:, 10:14] / norm[:, None]
    angle = np.degrees(2*np.arccos(np.clip(np.abs((q0*q1).sum(1)), 0, 1)))
    alpha0, alpha1 = 1/(1+np.exp(-old[:, 6])), 1/(1+np.exp(-new[:, 6]))
    result = dict(passed=True, points=len(new), centers_exactly_unchanged=True,
        finite=True, quaternion_max_norm_error=float(np.max(np.abs(norm-1))),
        scale_ratio_range=[float(ratio.min()), float(ratio.max())],
        rotation_degrees_p50_p90_p99=np.percentile(angle, [50,90,99]).tolist(),
        mean_absolute_opacity_change=float(np.abs(alpha1-alpha0).mean()),
        initial_opacity_logits_outside_bounds=int(((old[:,6]<-2)|(old[:,6]>8)).sum()),
        candidate_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    run()
