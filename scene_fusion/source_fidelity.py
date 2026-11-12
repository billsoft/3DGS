"""Directional and local source checks; averages cannot hide damaged regions."""
import numpy as np
import cv2 as cv

def local_fidelity(native,candidate,photo,mask,tile=128):
    if native.shape!=candidate.shape or photo.shape!=native.shape or mask.shape!=native.shape[:2]:raise ValueError('Mismatched comparison arrays')
    sigma=max(1.,native.shape[0]/128.)
    errors=[abs(im-photo) for im in [native,candidate]]
    blotches=[abs(cv.GaussianBlur((im-photo).astype('f'),(0,0),sigma)) for im in [native,candidate]]
    rows=[]
    for y in range(0,len(mask),tile):
        for x in range(0,mask.shape[1],tile):
            sl=np.s_[y:y+tile,x:x+tile];m=mask[sl]
            if m.mean()<.9:continue
            before,after=[float(e[sl][m].mean()) for e in errors];bb,ba=[float(e[sl][m].mean()) for e in blotches]
            rows.append(dict(x=x,y=y,native_mae=before,candidate_mae=after,native_blotch=bb,candidate_blotch=ba,
                regressed=after>before+.003 or ba>bb+.002))
    return dict(native_blotch=float(blotches[0][mask].mean()),candidate_blotch=float(blotches[1][mask].mean()),
                checked_tiles=len(rows),regressed_tiles=sum(r['regressed'] for r in rows),tiles=rows)

def strict_source_gate(records):
    failed=[]
    for r in records:
        reasons=[]
        if r['fused'].get('holes',0)>0:reasons.append('lost native coverage')
        if r['fused']['ssim']<r['native']['ssim']-.002:reasons.append('directional SSIM')
        if r['fused']['mae']>r['native']['mae']+.0005:reasons.append('directional RGB error')
        local=r['local_fidelity']
        if local['candidate_blotch']>local['native_blotch']+.0005:reasons.append('low-frequency color blotches')
        if local['regressed_tiles']:reasons.append('local patches')
        if reasons:failed.append(dict(view=r['view'],reasons=reasons,regressed_tiles=local['regressed_tiles']))
    return dict(passed=bool(records) and not failed,failed_directions=failed,total_directions=len(records),
        note='Explicit per-direction and local-patch tolerances, plus visual review; not a pixel-exact guarantee.')
