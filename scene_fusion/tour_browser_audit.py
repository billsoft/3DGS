"""Check exported PLY and actual browser camera renders against the tour teacher."""
import argparse,json
from pathlib import Path
import numpy as np,torch,cv2 as cv
from PIL import Image,ImageDraw
from core import write_json
from refit_render import load_scene,render
from tour_distill import tour_weights

def run(out):
    meta=json.loads((out/'data/dataset.json').read_text('utf8'));student=load_scene(out/'optimized.npz')
    teachers=[load_scene(out/'data'/f'teacher-{i}.npy') for i in range(len(meta['centers']))];records=[]
    for path in sorted((out/'browser').glob('*.json')):
        m=json.loads(path.read_text('utf8'))
        if 'projection' not in m:continue
        w,h=m['width'],m['height']
        if w!=h:raise ValueError('Audit expects square capture')
        p=np.array(m['projection']).reshape(4,4).T;v=np.diag([1,-1,-1,1])@np.array(m['worldInverse']).reshape(4,4).T
        view=dict(viewmat=v.tolist(),fov=float(np.rad2deg(2*np.arctan(1/p[0,0]))))
        with torch.no_grad():
            rgb,_,_=render(student,view,w);teacher=torch.zeros_like(rgb)
            weights=tour_weights(m['position'],np.array(meta['centers']),meta['core'])
            for i,weight in enumerate(weights):
                if weight:teacher+=weight*render(teachers[i],view,w)[0]
        rgb=rgb.clamp(0,1).cpu().numpy();teacher=teacher.clamp(0,1).cpu().numpy()
        browser=cv.imread(str(path.with_suffix('.png')))[:,:,::-1]/255.
        record=dict(view=path.stem,browser_gpu_mae=float(abs(browser-rgb).mean()),browser_teacher_mae=float(abs(browser-teacher).mean()),gpu_teacher_mae=float(abs(rgb-teacher).mean()))
        records.append(record)
        sheet=Image.new('RGB',(w*3,h+26));draw=ImageDraw.Draw(sheet)
        for i,(label,img) in enumerate([('44% teacher, actual camera',teacher),('student GPU',rgb),('exported PLY in browser',browser)]):
            sheet.paste(Image.fromarray(np.uint8(img*255)),(i*w,26));draw.text((i*w+4,6),label,fill='white')
        sheet.save(out/'browser'/(path.stem+'-parity.jpg'),quality=96)
    if not records:raise ValueError('No captured browser camera frames to audit')
    write_json(out/'browser/parity.json',dict(views=records,renderer_mae_mean=float(np.mean([r['browser_gpu_mae'] for r in records]))));print(records,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
