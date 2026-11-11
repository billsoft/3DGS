"""Fixed-camera baseline comparison plus unsharpened original-photo references."""
import os
from pathlib import Path
os.environ.setdefault('GS_OUTPUT_DIR',str(Path(__file__).resolve().parents[1]/'four-room/fusion-detail'))
from common import *
from PIL import Image,ImageDraw
from skimage.metrics import structural_similarity

def run():
    baseline=ROOT/'four-room/baselines/2026-11-11-fusion-next'
    up=np.array(rm.meta['up']);size=640
    details=json.loads((ROOT/'four-room/surface-fusion-v4/inspection-views.json').read_text())
    views=[]
    for view in details:
        cam=np.array(view['position']);z=np.array(view['target'])-cam;z/=np.linalg.norm(z);x=np.cross(z,up);x/=np.linalg.norm(x)
        views.append((view['name'],cam,np.stack([x,np.cross(z,x),z],1),None))
    ims=photos()
    selections=[(0,k) for k in range(6)]+[(2,4)]
    if os.environ.get('GS_QUICK_EVAL')=='1':selections=[(0,1),(0,2),(0,4)]
    for i,face in selections:
        local=rm.f.prep.FACES[face]
        ref=rm.f.prep.render.photo_view(ims[i],local,size)[:,:,::-1]
        views.append((f'source {stations[i]} face {face}',C[i],Q[i]@local,ref))
    for col,(label,folder) in enumerate([('accepted baseline',baseline),('edge detail',OUT)]):
        a=rm.f.prep.read_gs(folder/'fused.ply');cov,rgb,opacity=rm.f.prep.gaussian_parameters(a)
        for row,(name,cam,basis,ref) in enumerate(views):
            im,_,alpha=rm.f.prep.render.raster((a[:,:3]-cam).astype('d'),cov,rgb,opacity,basis,size)
            im=np.uint8(np.clip(im*255,0,255));Image.fromarray(im).save(OUT/f'eval-{col}-{row}.jpg',quality=97)
            np.savez_compressed(OUT/f'eval-{col}-{row}.npz',rgb=im,alpha=alpha)
            print(label,name,flush=True)
        del a,cov,rgb,opacity
    close=Image.new('RGB',(1280,675*len(details)),(20,24,30));draw=ImageDraw.Draw(close)
    for row in range(len(details)):
        for col,label in enumerate(['accepted baseline','edge detail']):
            close.paste(Image.open(OUT/f'eval-{col}-{row}.jpg'),(col*640,row*675+35));draw.text((col*640+10,row*675+10),label+' / '+views[row][0],fill='white')
    close.save(OUT/'details.jpg',quality=97)
    photosheet=Image.new('RGB',(1920,675*(len(views)-len(details))),(20,24,30));draw=ImageDraw.Draw(photosheet);metrics=[]
    for row,(name,_,_,ref) in enumerate(views):
        before=np.load(OUT/f'eval-0-{row}.npz');after=np.load(OUT/f'eval-1-{row}.npz')
        record=dict(view=name,baseline_holes=float((before['alpha']<.05).mean()),detail_holes=float((after['alpha']<.05).mean()))
        if ref is not None:
            y=(row-len(details))*675
            for col,(label,im) in enumerate([('original panorama',ref),('accepted baseline',before['rgb']),('edge detail',after['rgb'])]):
                photosheet.paste(Image.fromarray(im),(col*640,y+35));draw.text((col*640+10,y+10),label+' / '+name,fill='white')
            mask=(before['alpha']>.95)&(after['alpha']>.95)
            mask=cv.erode(mask.astype('u1'),np.ones((9,9),'u1'))>0
            record['common_coverage']=float(mask.mean())
            for label,data in [('baseline',before),('detail',after)]:
                gray=cv.cvtColor(data['rgb'],cv.COLOR_RGB2GRAY);truth=cv.cvtColor(ref,cv.COLOR_RGB2GRAY)
                _,ssim=structural_similarity(gray,truth,data_range=255,full=True)
                record[label+'_mae']=float(np.abs(data['rgb'].astype('f')-ref)[mask].mean()/255)
                record[label+'_ssim']=float(ssim[mask].mean())
        metrics.append(record)
    photosheet.save(OUT/'photo-compare.jpg',quality=97)
    write_json('detail-validation.json',dict(views=metrics,note='Source-view photo metrics use common high-alpha coverage. They do not measure unseen novel-view correctness; brightness/geometry mismatch remains.'))

if __name__=='__main__':run()
