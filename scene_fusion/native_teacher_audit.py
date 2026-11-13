"""Separate native renderer mismatch, coordinate conversion and student loss."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,cv2 as cv
from core import gs,write_json
from input_io import read_gaussians

def tensor_model(a,torch,quantized=False):
    colors=np.clip(a[:,3:6]*.2820947918+.5,0,1)
    opacity=a[:,6].copy()
    if quantized:
        colors=np.floor(colors*255)/255
        alpha=np.floor(255/(1+np.exp(-np.clip(opacity,-30,30))))/255
        alpha=np.clip(alpha,1e-7,1-1e-7);opacity=np.log(alpha/(1-alpha))
    return {k:torch.tensor(v.copy(),device='cuda') for k,v in dict(means=a[:,:3],colors=colors,opacity=opacity,logs=a[:,7:10],quats=a[:,10:14]).items()}

def compare(a,b):
    delta=np.abs(a-b);chroma=(a-a.mean(2,keepdims=True))-(b-b.mean(2,keepdims=True))
    h,w=a.shape[:2];tiles=np.array([[delta[y:y+32,x:x+32].mean() for x in range(0,w-31,32)] for y in range(0,h-31,32)])
    y,x=np.unravel_index(tiles.argmax(),tiles.shape)
    return dict(mae=float(delta.mean()),chroma_mae=float(abs(chroma).mean()),worst32_mae=float(tiles[y,x]),worst32_xy=[int(x*32),int(y*32)])

def run(a):
    import torch
    from refit_render import render,load_scene
    from PIL import Image,ImageDraw
    torch.set_num_threads(4)
    native=json.loads((a.native/'native.json').read_text('utf8'))
    captured=json.loads((a.out/'captures.json').read_text('utf8'));records=[]
    student=load_scene(a.student/'optimized.npz') if a.student else None
    for source in sorted({r['source'] for r in captured['records']}):
        s=native['stations'][source];path=a.native/s['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=s['sha256']:raise ValueError('Native baseline changed')
        raw=read_gaussians(path);world=gs.transform(raw,np.array(s['world_from_raw']))
        local_model=tensor_model(raw,torch);world_model=tensor_model(world,torch);quantized=tensor_model(world,torch,True)
        for record in [r for r in captured['records'] if r['source']==source]:
            size=record['width'];assert size==record['height']
            vm=lambda key:dict(viewmat=(np.diag([1,-1,-1,1])@np.array(record[key]).reshape(4,4).T).tolist(),fov=record['fov'])
            browser=cv.imread(str(a.out/(record['name']+'.png')))[:,:,::-1]/255.
            with torch.no_grad():
                local=render(local_model,vm('localInverse'),size)[0].clamp(0,1).cpu().numpy()
                image=render(world_model,vm('worldInverse'),size)[0].clamp(0,1).cpu().numpy()
                quant=render(quantized,vm('worldInverse'),size)[0].clamp(0,1).cpu().numpy()
                pred=render(student,vm('worldInverse'),size)[0].clamp(0,1).cpu().numpy() if student else image
            result=dict(name=record['name'],browser_vs_local=compare(browser,local),local_vs_world=compare(local,image),browser_vs_teacher=compare(browser,image),browser_vs_quantized=compare(browser,quant),browser_vs_student=compare(browser,pred),teacher_vs_student=compare(image,pred))
            records.append(result)
            sheet=Image.new('RGB',(size*4,size+24));draw=ImageDraw.Draw(sheet)
            for i,(label,img) in enumerate([('native browser',browser),('offscreen teacher',image),('existing single PLY',pred),('browser/student error x5',abs(browser-pred)*5)]):
                sheet.paste(Image.fromarray(np.uint8(np.clip(img,0,1)*255)),(i*size,24));draw.text((i*size+5,5),label,fill='white')
            sheet.save(a.out/(record['name']+'-comparison.jpg'),quality=95)
            print(result,flush=True)
        del local_model,world_model,quantized;torch.cuda.empty_cache()
    write_json(a.out/'audit.json',dict(core=captured['core'],student=str(a.student),views=records,mean={k:float(np.mean([r[k]['mae'] for r in records])) for k in ('browser_vs_local','local_vs_world','browser_vs_teacher','browser_vs_quantized','browser_vs_student','teacher_vs_student')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--student',type=Path);run(p.parse_args())
