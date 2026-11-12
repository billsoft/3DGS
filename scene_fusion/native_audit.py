"""Compare final fusion with each full original PLY at its calibrated source camera.

This catches holes that an already-cropped fusion baseline would hide. Photos
are fitting references, not held-out novel-view truth. Uses a fixed native mask.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from core import load,gs,write_json
from refit_render import load_scene,load_dataset,render
from refit_evaluate import metrics

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--size',type=int,default=512)
args=p.parse_args();args.report.mkdir(parents=True,exist_ok=True);scene=load(args.out);cams=np.load(args.out/'aligned-cameras.npz');meta,targets=load_dataset(args.out/f'data-{args.size}')
fused=load_scene(args.candidate);records=[]
for i,item in enumerate(scene['stations']):
    path=args.out/'aligned'/(item['id']+'-full.npy')
    np.save(path,gs.transform(np.load(args.out/item['id']/'original.npy'),cams['matrices'][i]))
    native=load_scene(path)
    with torch.no_grad():
        for idx,v in enumerate(meta['views']):
            if v['source']!=i:continue
            raw,ra,_=render(native,v,args.size);result,fa,_=render(fused,v,args.size)
            raw=raw.clamp(0,1).cpu().numpy();result=result.clamp(0,1).cpu().numpy();ra=ra.cpu().numpy();fa=fa.cpu().numpy();photo=targets[idx]/255
            before=metrics(raw,photo,ra,ra);after=metrics(result,photo,fa,ra)
            records.append(dict(view=v['name'],native=before,fused=after))
            sheet=Image.new('RGB',(args.size*3,args.size+30));draw=ImageDraw.Draw(sheet)
            for col,(title,im) in enumerate([('photo',photo),('original PLY',raw),('fused',result)]):
                sheet.paste(Image.fromarray((im*255).astype('u1')),(col*args.size,30));draw.text((col*args.size+6,8),title,fill='white')
            sheet.save(args.report/(v['name']+'.jpg'),quality=96)
    del native;torch.cuda.empty_cache()
summary=dict(views=len(records),native_ssim=float(np.mean([r['native']['ssim'] for r in records])),
    fused_ssim=float(np.mean([r['fused']['ssim'] for r in records])),
    newly_low_alpha_pixels=sum(r['fused']['holes'] for r in records),
    native_high_alpha_pixels=sum(r['native']['mask_pixels'] for r in records))
write_json(args.report/'audit.json',dict(summary=summary,views=records,candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),size=args.size,note='Fixed full-native alpha mask; photo fitting diagnostic, no independent novel-view reference.'))
print(json.dumps(summary),flush=True)
