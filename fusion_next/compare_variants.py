"""Compare variants on exactly the same source-view pixels, not different masks."""
from common import *
from skimage.metrics import structural_similarity

def run():
    folders={'surface':OUT,'raw':OUT/'raw-eval','anchored':OUT/'anchored-eval','painted':OUT/'painted-eval','hybrid':OUT/'hybrid-eval'}
    reports={name:json.loads((folder/'detail-validation.json').read_text()) for name,folder in folders.items()}
    indices={name:{r['view']:i for i,r in enumerate(report['views'])} for name,report in reports.items()}
    shared=set.intersection(*[set(i) for i in indices.values()]);images=photos();rows=[]
    for view in sorted(shared):
        if not view.startswith('source '):continue
        _,station,_,face=view.split();source=stations.index(station);face=int(face)
        truth=rm.f.prep.render.photo_view(images[source],rm.f.prep.FACES[face],640)[:,:,::-1]
        data={name:np.load(folder/f'eval-1-{indices[name][view]}.npz') for name,folder in folders.items()}
        data['baseline']=np.load(OUT/f'eval-0-{indices["surface"][view]}.npz')
        mask=np.logical_and.reduce([d['alpha']>.95 for d in data.values()]);mask=cv.erode(mask.astype('u1'),np.ones((9,9),'u1'))>0
        row=dict(view=view,common_coverage_fraction=float(mask.mean()),models={})
        for name,d in data.items():
            _,ssim=structural_similarity(cv.cvtColor(d['rgb'],cv.COLOR_RGB2GRAY),cv.cvtColor(truth,cv.COLOR_RGB2GRAY),data_range=255,full=True)
            row['models'][name]=dict(ssim=float(ssim[mask].mean()),mae=float(np.abs(d['rgb'].astype('f')-truth)[mask].mean()/255),holes=float((d['alpha']<.05).mean()))
        rows.append(row)
    write_json('variants-report.json',dict(views=rows,note='All variants and baseline use an identical eroded high-alpha pixel mask per source view. Scores do not certify novel-view geometry or invisible surfaces.'))
    print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':run()
