"""Check original-coordinate and aligned-coordinate renders at all source faces."""
from common import *

def run():
    source=ROOT/'four-room/fusion-native-fidelity';records=[]
    for station in stations:
        cache=np.load(rm.room/station/'cache.npz');a=rm.f.prep.read_gs(rm.inputs/f'{station}.ply')
        cov,rgb,opacity=rm.f.prep.gaussian_parameters(a)
        for face,b in enumerate(rm.f.prep.FACES):
            im,_,alpha=rm.f.prep.render.raster((a[:,:3]-cache['C']).astype('d'),cov,rgb,opacity,cache['R'].T@b,384)
            im=np.uint8(np.clip(im*255,0,255));aligned=np.load(source/f'{station}-native-{face}.npz')
            records.append(dict(station=station,face=face,mean_rgb_difference_255=float(abs(im.astype('f')-aligned['rgb']).mean()),mean_alpha_difference=float(abs(alpha-aligned['alpha']).mean())))
        print('original coordinate equivalence',station,flush=True)
    write_json('alignment-validation.json',dict(views=records,passed=all(r['mean_rgb_difference_255']<.5 and r['mean_alpha_difference']<.001 for r in records),note='Sim(3) transform invariance, not independent confirmation of inter-camera registration accuracy. Small numerical rasterization differences are allowed.'))

if __name__=='__main__':run()
