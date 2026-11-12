"""Fail before GPU work for invalid pairings, photos or Gaussian data."""
import argparse
from pathlib import Path
from PIL import Image
from core import initialize,write_json
from input_io import read_gaussians,normalize

def run(inputs,out,stations=None):
    meta=initialize(inputs,out,stations);rows=[]
    for item in meta['stations']:
        with Image.open(item['photo']) as photo:
            w,h=photo.size
            if abs(w/h-2)>.01:raise ValueError(f"{item['label']}: panorama must have 2:1 width/height, got {w}x{h}")
            photo.verify()
        a=read_gaussians(item['ply']);_,scale=normalize(a)
        rows.append(dict(id=item['id'],label=item['label'],width=w,height=h,points=len(a),normalization=scale))
    write_json(Path(out)/'preflight.json',dict(passed=True,count=len(rows),points=sum(r['points'] for r in rows),sources=rows))
    print('Validated',len(rows),'panorama/PLY pairs',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stations',nargs='+');a=p.parse_args();run(a.inputs,a.out,a.stations)
