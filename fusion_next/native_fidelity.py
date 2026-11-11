"""Source-camera fidelity tests and an unmodified-Gaussian ownership ablation."""
from common import *
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
from skimage.metrics import structural_similarity

def run():
    labels=np.load(ROOT/'four-room/fusion-territory/ownership.npz');tree=cKDTree(labels['center'])
    parts=[];sources=[];counts=[]
    for i,station in enumerate(stations):
        original=rm.f.prep.read_gs(rm.inputs/f'{station}.ply')
        aligned=rm.f.transform(original,np.linalg.inv(cams[i].inv))
        rm.f.write(OUT/f'native-{station}.ply',aligned);rm.f.to_splat(OUT/f'native-{station}.splat',aligned)
        _,face=tree.query(aligned[:,:3]);keep=labels['owner'][face]==i
        parts.append(aligned[keep]);sources.append(aligned)
        counts.append(dict(station=station,original=len(aligned),retained=int(keep.sum())))
    merged=np.concatenate(parts);rm.f.write(OUT/'fused.ply',merged);rm.f.to_splat(OUT/'fused.splat',merged)
    baseline=rm.f.prep.read_gs(ROOT/'four-room/baselines/2026-11-11-coverage-guard/fused.ply')
    size=384;records=[]
    for i,station in enumerate(stations):
        cropped=rm.f.prep.read_gs(ROOT/f'four-room/fusion-territory/cropped-{station}.ply')
        sheet=Image.new('RGB',(size*4,(size+26)*6),(20,24,30));draw=ImageDraw.Draw(sheet)
        reference=[]
        for col,(name,a) in enumerate([('native',sources[i]),('cropped',cropped),('accepted',baseline),('native_owner',merged)]):
            cov,rgb,opacity=rm.f.prep.gaussian_parameters(a)
            for face,b in enumerate(rm.f.prep.FACES):
                im,_,alpha=rm.f.prep.render.raster((a[:,:3]-C[i]).astype('d'),cov,rgb,opacity,Q[i]@b,size)
                im=np.uint8(np.clip(im*255,0,255))
                np.savez_compressed(OUT/f'{station}-{name}-{face}.npz',rgb=im,alpha=alpha)
                sheet.paste(Image.fromarray(im),(col*size,face*(size+26)+26));draw.text((col*size+4,face*(size+26)+5),f'{station}/{face} {name}',fill='white')
                if col==0:reference.append((im,alpha))
                else:
                    ref,ra=reference[face];mask=ra>.95
                    _,ssim=structural_similarity(cv.cvtColor(ref,cv.COLOR_RGB2GRAY),cv.cvtColor(im,cv.COLOR_RGB2GRAY),data_range=255,full=True)
                    records.append(dict(station=station,face=face,variant=name,source_supported_pixels=int(mask.sum()),new_hole_pixels=int((mask&(alpha<.5)).sum()),source_mask_mae=float(abs(im.astype('f')-ref)[mask].mean()/255),source_mask_ssim=float(ssim[mask].mean())))
            print('rendered',station,name,flush=True)
        sheet.save(OUT/f'station-{station}.jpg',quality=95)
    write_json('fidelity-report.json',dict(sources=counts,views=records,note='Reference is the complete original Gaussian model rendered at its own calibrated camera. No source crop, footprint shrink, meshing, recoloring or nonlinear warp is applied to the reference. Static candidate only assigns whole Gaussians to surface ownership; not presumed successful.'))
    write_json('fusion.json',dict(centers=C.tolist(),up=rm.meta['up'],stations=stations))

if __name__=='__main__':run()
