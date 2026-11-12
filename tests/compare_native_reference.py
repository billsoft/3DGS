"""Pixel regression check against independently loaded original PLY renders."""
import argparse,json
from pathlib import Path
from PIL import Image
import numpy as np

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--actual',default='browser');p.add_argument('--out',type=Path);a=p.parse_args()
meta=json.loads((a.root/'native.json').read_text());rows=[]
for i in range(len(meta['stations'])):
    for face in [3,4,5]:
        name=f'source-{i+1}-face-{face}.png'
        multi=np.array(Image.open(a.root/a.actual/name)).astype(float)[:,:,:3]
        single=np.array(Image.open(a.root/'standalone'/name)).astype(float)[:,:,:3]
        d=abs(multi-single)
        rows.append(dict(image=name,mae_255=float(d.mean()),max_difference=float(d.max()),fraction_pixels_over_5=float((d.max(2)>5).mean())))
passed=bool(rows) and all(r['mae_255']<.1 and r['fraction_pixels_over_5']<.001 for r in rows)
result=dict(passed=passed,images=len(rows),maximum_mae_255=max(r['mae_255'] for r in rows),maximum_fraction_pixels_over_5=max(r['fraction_pixels_over_5'] for r in rows),thresholds=dict(mae_255=.1,fraction_pixels_over_5=.001),rows=rows,note='Independent single original PLY reference; not a comparison with panorama photographs. Small differences are compatible with sort quantization/ties.')
(a.out or a.root/'standalone/comparison.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}));raise SystemExit(0 if passed else 1)
