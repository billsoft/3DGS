"""Same-camera photo/native/baseline/recolor/split comparisons across object types."""
from common import *
from PIL import Image,ImageDraw
from skimage.metrics import structural_similarity

def run():
    size=480;up=np.array(rm.meta['up']);views=[]
    for i,v in enumerate(json.loads((ROOT/'four-room/surface-fusion-v4/inspection-views.json').read_text())[:2]):
        z=np.array(v['target'])-C[i];z/=np.linalg.norm(z);x=np.cross(z,up);x/=np.linalg.norm(x)
        views.append((v['name'],i,np.stack([x,np.cross(z,x),z],1)))
    for name,i,face in [('window 2',0,2),('window 3',1,2),('table 2',0,5),('board 3',1,1)]:views.append((name,i,Q[i]@rm.f.prep.FACES[face]))
    images=photos();yy,xx=np.indices((size,size));rays=np.stack([(xx+.5-size/2)/(size/2),(yy+.5-size/2)/(size/2),np.ones_like(xx)],-1)
    references=[];rows=[]
    for name,i,basis in views:references.append(sample(images[i],uv((rays@basis.T@Q[i]).reshape(-1,3))).reshape(size,size,3)[:,:,::-1])
    data={}
    for label in ['native','baseline','recolor','split']:
        cache={}
        for row,(name,i,basis) in enumerate(views):
            path=(ROOT/f'four-room/fusion-native-fidelity/native-{stations[i]}.ply') if label=='native' else ROOT/('four-room/baselines/2026-11-11-coverage-guard/fused.ply' if label=='baseline' else f'four-room/footprint-{label}/fused.ply')
            if path not in cache:
                a=rm.f.prep.read_gs(path);cache={path:(a,*rm.f.prep.gaussian_parameters(a))}
            a,cov,rgb,opacity=cache[path]
            picture,_,alpha=rm.f.prep.render.raster((a[:,:3]-C[i]).astype('d'),cov,rgb,opacity,basis,size)
            picture=np.uint8(np.clip(picture*255,0,255));data[label,row]=(picture,alpha)
            print(label,name,flush=True)
    for row,(name,i,basis) in enumerate(views):
        sheet=Image.new('RGB',(size*5,size+30),(20,24,30));draw=ImageDraw.Draw(sheet)
        before=data['baseline',row][1];mask=before>.95;record=dict(view=name,station=stations[i],models={})
        for col,label in enumerate(['photo','native','baseline','recolor','split']):
            picture=references[row] if label=='photo' else data[label,row][0]
            sheet.paste(Image.fromarray(picture),(col*size,30));draw.text((col*size+6,8),label,fill='white')
            if label!='photo':
                _,sim=structural_similarity(cv.cvtColor(references[row],cv.COLOR_RGB2GRAY),cv.cvtColor(picture,cv.COLOR_RGB2GRAY),data_range=255,full=True)
                record['models'][label]=dict(photo_mae=float(abs(picture.astype('f')-references[row])[mask].mean()/255),photo_ssim=float(sim[mask].mean()),new_hole_pixels=int((mask&(data[label,row][1]<.5)).sum()))
        sheet.save(OUT/f'comparison-{row}.jpg',quality=97);rows.append(record)
    write_json('validation.json',dict(views=rows,note='Same camera and 90-degree FOV. All methods use the baseline high-alpha mask, including newly missing pixels. Photos are input constraints, not independent novel-view ground truth.'))

if __name__=='__main__':run()
