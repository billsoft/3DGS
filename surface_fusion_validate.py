from pathlib import Path
import json,importlib.util,argparse
import cv2 as cv,numpy as np
from PIL import Image,ImageDraw
s=importlib.util.spec_from_file_location('f','four-fuse.py');f=importlib.util.module_from_spec(s);s.loader.exec_module(f)
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('four-room/surface-fusion-v2'));parser.add_argument('--baseline',type=Path,default=Path('four-room/photo-surfaces.ply'));args=parser.parse_args()
D=Path('four-room');O=args.out;meta=json.loads((D/'fusion.json').read_text());C=np.array(meta['centers']);up=np.array(meta['up'])
size=640;report={};sheet=Image.new('RGB',(1280,2700),(18,22,30));draw=ImageDraw.Draw(sheet)
for col,(label,path) in enumerate([('baseline: '+args.baseline.parent.name,args.baseline),('new: '+O.name,O/'fused.ply')]):
    a=f.prep.read_gs(path);cov,rgb,opacity=f.prep.gaussian_parameters(a);stats=[]
    for i in range(4):
        center=(C[i]+C[(i+1)%4])/2;z=np.array([4,0,-.6])-center;z/=np.linalg.norm(z);x=np.cross(z,up);x/=np.linalg.norm(x);b=np.stack([x,np.cross(z,x),z],1)
        im,dep,alpha=f.prep.render.raster((a[:,:3]-center).astype('d'),cov,rgb,opacity,b,size)
        im=np.uint8(np.clip(im*255,0,255));Image.fromarray(im).save(O/f'{col}-mid-{i+1}.jpg',quality=95)
        draw.text((col*640+12,i*675+10),label+f' / midpoint {i+1}',fill='white');sheet.paste(Image.fromarray(im),(col*640,i*675+35))
        stats.append(dict(midpoint=i+1,low_alpha_fraction=float((alpha<.05).mean())));print(label,stats[-1],flush=True)
    report[label]=stats
sheet.save(O/'compare.jpg',quality=95);(O/'validation.json').write_text(json.dumps(report,indent=2))
