"""Inspect unchanged alpha AND errors on actually modified source-view pixels."""
from common import *

def run():
    report=json.loads((OUT/'detail-validation.json').read_text());ims=photos();records=[]
    for row,item in enumerate(report['views']):
        before=np.load(OUT/f'eval-0-{row}.npz');after=np.load(OUT/f'eval-1-{row}.npz')
        alpha_difference=float(np.max(abs(before['alpha']-after['alpha'])))
        changed=np.max(abs(before['rgb'].astype('i')-after['rgb'].astype('i')),axis=2)>2
        r=dict(view=item['view'],maximum_alpha_difference=alpha_difference,new_hole_pixels=int(((before['alpha']>.95)&(after['alpha']<.5)).sum()),changed_pixels=int(changed.sum()))
        if item['view'].startswith('source '):
            _,station,_,face=item['view'].split();i=stations.index(station)
            truth=rm.f.prep.render.photo_view(ims[i],rm.f.prep.FACES[int(face)],640)[:,:,::-1]
            valid=changed&(before['alpha']>.95)&(after['alpha']>.95)
            delta=(abs(after['rgb'].astype('f')-truth)-abs(before['rgb'].astype('f')-truth)).mean(2)/255
            r.update(evaluated_changed_pixels=int(valid.sum()),changed_pixel_mae_delta=float(delta[valid].mean()) if valid.any() else None)
        records.append(r)
    visible_regression=[r['view'] for r in records if r.get('evaluated_changed_pixels',0)>=50 and r.get('changed_pixel_mae_delta',0)>.003]
    write_json('coverage-audit.json',dict(views=records,coverage_pass=all(r['maximum_alpha_difference']<1e-6 and r['new_hole_pixels']==0 for r in records),changed_region_regressions=visible_regression,note='Positive changed-pixel MAE delta means worse photo agreement. Source photos are inputs, not held-out novel-view truth. Original holes remain.'))
    print(json.dumps(records,indent=2),flush=True)

if __name__=='__main__':run()
