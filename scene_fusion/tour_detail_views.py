"""Additional deterministic camera FOVs for native-tour distillation ablations."""
import argparse,json
from pathlib import Path
import numpy as np
from core import write_json,viewmat

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

def transition_views(meta):
    core=float(meta['core']);centers=np.asarray(meta['centers']);result=[]
    if not .1<=core<.5:raise ValueError('Transition sampler expects 0.1 <= core < 0.5')
    for a,b in meta['midpoint_edges']:
        faces=[v for v in meta['views'] if v['kind']=='anchor' and v.get('source')==a]
        for train,check in [(core-.01,core-.015),(core,core-.003),(1-core,1-core+.003),(1-core+.01,1-core+.015)]:
            for face,v in enumerate(faces):
                for role,t in [('train',train),('test',check)]:
                    c=centers[a]*(1-t)+centers[b]*t;basis=np.array(v['basis'])
                    result.append(dict(name=f'boundary-{a+1}-{b+1}-{t:.3f}-{face}',center=c.tolist(),basis=basis.tolist(),viewmat=viewmat(c,basis).tolist(),source=None,role=role,kind='route',fov=90))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--transitions',action='store_true')
    a=p.parse_args();meta=json.loads(a.dataset.read_text('utf8'));result=detail_views(meta)
    if a.transitions:result['views']+=transition_views(meta)
    write_json(a.out,result)
