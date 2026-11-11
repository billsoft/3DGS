"""Robust per-camera RGB gain/offset, validated on held correspondence blocks."""
from common import *
from scipy.optimize import least_squares
def run():
    corr=np.load(OUT/'correspondences.npz');images=photos((3840,1920));pair=corr['pair'];a=np.zeros((len(pair),3));b=a.copy()
    for i in range(4):
        for slot,array in [(0,a),(1,b)]:
            ids=np.flatnonzero(pair[:,slot]==i);array[ids]=sample(images[i],uv(project(corr['target'][ids],i)))[:,::-1]/255
    good=(a.min(1)>.04)&(b.min(1)>.04)&(a.max(1)<.96)&(b.max(1)<.96)&(np.linalg.norm(a-b,axis=1)<.20)
    train=good&~corr['held'];held=good&corr['held']
    def unpack(v):
        gain=np.vstack([np.ones(3),v[:9].reshape(3,3)]);offset=np.vstack([np.zeros(3),v[9:].reshape(3,3)]);return gain,offset
    start=np.r_[np.ones(9),np.zeros(9)]
    def differences(v):
        gain,offset=unpack(v);return a*gain[pair[:,0]]+offset[pair[:,0]]-b*gain[pair[:,1]]-offset[pair[:,1]]
    def residual(v):return np.r_[differences(v)[train].ravel(),.2*(v-start)]
    result=least_squares(residual,start,bounds=(np.r_[np.full(9,.8),np.full(9,-.06)],np.r_[np.full(9,1.2),np.full(9,.06)]),loss='soft_l1',f_scale=.015,max_nfev=80)
    before=np.mean(abs(differences(start)[held]),axis=1);after=np.mean(abs(differences(result.x)[held]),axis=1)
    accepted=bool(len(before)>50 and np.median(after)<np.median(before)*.97 and np.quantile(after,.9)<np.quantile(before,.9)*1.02)
    gain,offset=unpack(result.x if accepted else start)
    report=dict(accepted=accepted,gain=gain.tolist(),offset=offset.tolist(),held=int(held.sum()),median_before=float(np.median(before)),median_after=float(np.median(after)),p90_before=float(np.quantile(before,.9)),p90_after=float(np.quantile(after,.9)),note='Bounded RGB appearance compensation, no sharpening or synthetic texture')
    write_json('appearance-report.json',report);print(report,flush=True)
if __name__=='__main__':run()
