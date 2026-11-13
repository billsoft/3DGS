"""Additional deterministic camera FOVs for native-tour distillation ablations."""
import argparse,json
from pathlib import Path
from core import write_json

def detail_views(meta,train_fov=50,check_fov=60):
    if not 10<=train_fov<=120 or not 10<=check_fov<=120 or train_fov==check_fov:
        raise ValueError('Distinct camera FOVs in [10,120] required')
    views=[]
    for v in meta['views']:
        if v['role']!='train' or v['kind'] not in ('anchor','near'):continue
        # tour_refine alternates the extra list into train/test roles.
        for role,fov in [('train',train_fov),('test',check_fov)]:
            views.append(dict(v,name=v['name']+f'-detail-{fov}',role=role,fov=fov))
    if not views:raise ValueError('No source/near-source training views')
    return dict(views=views,train_fov=train_fov,check_fov=check_fov,note='Synthetic native-PLY supervision; held FOV is not a new real observation.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();write_json(a.out,detail_views(json.loads(a.dataset.read_text('utf8'))))
