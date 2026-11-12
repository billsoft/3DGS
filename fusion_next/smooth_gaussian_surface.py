"""Conservative local point-to-plane smoothing with edge and layer guards.

No RGB averaging, object detection, hole filling, or point deletion. Neighborhoods
must agree in normal and have tangent support in two directions. The candidate
still requires multi-view coverage and visual evaluation before publication.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    a = np.load(args.input)
    pos = a[:, :3]
    frames = Rotation.from_quat(a[:, [11,12,13,10]]).as_matrix().astype('f')
    axis = a[:,7:10].argmin(1)
    normal = frames[np.arange(len(a)), :, axis]
    tree = cKDTree(pos)
    shift = np.zeros_like(pos)
    support = np.zeros(len(a), bool)
    for start in range(0, len(a), 60000):
        ids = np.arange(start, min(start+60000, len(a)))
        distance, ix = tree.query(pos[ids], k=17, workers=4)
        distance, ix = distance[:,1:], ix[:,1:]
        delta = pos[ix]-pos[ids,None]
        n = normal[ids]
        signed = np.einsum('bkj,bj->bk', delta, n)
        alignment = np.abs(np.einsum('bkj,bj->bk', normal[ix], n))
        spacing = np.maximum(np.median(distance,axis=1), .0001)
        valid = (alignment>.9)&(distance<.04)&(np.abs(signed)<np.minimum(.008,spacing[:,None]*.6))
        weight = valid*np.exp(-(distance/(spacing[:,None]+1e-8))**2)
        weight *= np.clip((alignment-.9)/.1,0,1)**2
        tangent = delta-signed[:,:,None]*n[:,None]
        cov = np.einsum('bk,bki,bkj->bij',weight,tangent,tangent)
        eig = np.linalg.eigvalsh(cov)
        # Do not move thin rods, sharp boundaries, or unsupported silhouettes.
        mean_tangent = np.einsum('bk,bki->bi',weight,tangent)/(weight.sum(1)[:,None]+1e-9)
        balanced = np.linalg.norm(mean_tangent,axis=1)<spacing*.25
        trusted = (valid.sum(1)>=6)&(eig[:,1]>.12*eig[:,2])&balanced
        # Robust local normal offset: weighted median rather than averaging layers.
        order = np.argsort(signed,axis=1)
        sorted_weight = np.take_along_axis(weight,order,axis=1)
        median_idx = (np.cumsum(sorted_weight,axis=1)<weight.sum(1)[:,None]*.5).sum(1).clip(0,15)
        offset = np.take_along_axis(signed,order,axis=1)[np.arange(len(ids)),median_idx]
        offset = np.clip(offset,-np.minimum(.002,spacing*.2),np.minimum(.002,spacing*.2))
        shift[ids] = np.where(trusted[:,None], offset[:,None]*n, 0)
        support[ids] = trusted
        if start%600000==0: print('Neighborhoods',start,'/',len(a),flush=True)
    reports = []
    for name,strength in [('mild',.35),('medium',.7)]:
        b=a.copy();b[:,:3]+=strength*shift
        np.save(args.out/f'{name}.npy',b)
        moved=np.linalg.norm(b[:,:3]-a[:,:3],axis=1)
        assert np.array_equal(a[:,3:],b[:,3:])
        reports.append(dict(candidate=name,strength=strength,points=len(a),
            supported=int(support.sum()),moved=int((moved>0).sum()),
            max_shift=float(moved.max()),mean_shift=float(moved.mean()),
            colors_opacity_covariance_exactly_unchanged=True))
    (args.out/'smoothing.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(reports),flush=True)


if __name__=='__main__': run()
