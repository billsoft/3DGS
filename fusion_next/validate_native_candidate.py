"""Evaluate a static native candidate against complete native camera references."""
from common import *
from PIL import Image,ImageDraw
from skimage.metrics import structural_similarity

def run():
    source=ROOT/'four-room/fusion-native-fidelity';size=384
    a=rm.f.prep.read_gs(OUT/'fused.ply');cov,rgb,opacity=rm.f.prep.gaussian_parameters(a);records=[]
    for i,station in enumerate(stations):
        sheet=Image.new('RGB',(size*3,(size+26)*6),(20,24,30));draw=ImageDraw.Draw(sheet)
        for face,b in enumerate(rm.f.prep.FACES):
            im,_,alpha=rm.f.prep.render.raster((a[:,:3]-C[i]).astype('d'),cov,rgb,opacity,Q[i]@b,size);im=np.uint8(np.clip(im*255,0,255))
            original=np.load(source/f'{station}-native-{face}.npz');accepted=np.load(source/f'{station}-accepted-{face}.npz');ref=original['rgb'];mask=original['alpha']>.95
            _,ssim=structural_similarity(cv.cvtColor(ref,cv.COLOR_RGB2GRAY),cv.cvtColor(im,cv.COLOR_RGB2GRAY),data_range=255,full=True)
            records.append(dict(station=station,face=face,new_hole_pixels=int((mask&(alpha<.5)).sum()),source_mask_mae=float(abs(im.astype('f')-ref)[mask].mean()/255),source_mask_ssim=float(ssim[mask].mean())))
            for col,(name,picture) in enumerate([('native',ref),('accepted',accepted['rgb']),('candidate',im)]):
                sheet.paste(Image.fromarray(picture),(col*size,face*(size+26)+26));draw.text((col*size+4,face*(size+26)+5),f'{station}/{face} {name}',fill='white')
        sheet.save(OUT/f'station-{station}.jpg',quality=95);print('validated',station,flush=True)
    write_json('native-validation.json',dict(views=records))

if __name__=='__main__':run()
