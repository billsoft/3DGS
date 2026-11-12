"""Compare poses on identical tracks unused for fitting in BOTH runs."""
import argparse,json
from pathlib import Path
import numpy as np


def run(old,new,out):
    bs=[np.load(p/'photo-bundle/bundle.npz') for p in (old,new)]
    ts=[json.loads((p/'photo-bundle/tracks.json').read_text()) for p in (old,new)]
    ns=json.loads((old/'registration.json').read_text())['stations']
    if ns!=json.loads((new/'registration.json').read_text())['stations']:raise ValueError('Station order differs')
    fs=[np.load(old/'photo-bundle'/f'features-{n}.npz') for n in ns]
    for f,n in zip(fs,ns):
        z=np.load(new/'photo-bundle'/f'features-{n}.npz')
        if not np.array_equal(f['uv'],z['uv']) or not np.array_equal(f['ray'],z['ray']):raise ValueError('Feature identities differ')
    def key(t):return tuple(sorted(map(tuple,t)))
    held=[{key(t) for t,h in zip(tracks,b['hold']) if h} for tracks,b in zip(ts,bs)]
    training=[{o for t,h in zip(tracks,b['hold']) if not h for o in key(t)} for tracks,b in zip(ts,bs)]
    shared=sorted(t for t in held[0]&held[1] if not any(o in training[0] or o in training[1] for o in t))
    if len(shared)<8:raise ValueError('Insufficient common untouched tracks')
    results=[]
    for b in bs:
        errors=[[] for n in ns]
        for t in shared:
            cams=np.array([c for c,f in t]);ray=np.array([fs[c]['ray'][f] for c,f in t])
            v=np.einsum('nij,nj->ni',b['Q'][cams],ray);A=np.eye(3)-v[:,:,None]*v[:,None,:]
            P=np.linalg.lstsq(A.sum(0),np.einsum('nij,nj->ni',A,b['C'][cams]).sum(0),rcond=None)[0]
            d=P-b['C'][cams];front=(d*v).sum(1)>0
            e=np.linalg.norm(d/np.linalg.norm(d,axis=1)[:,None]-v,axis=1)*3840/(2*np.pi)
            for c,x,ok in zip(cams,e,front):errors[c].append(float(x) if ok else 1e6)
        def stats(a):return dict(median=float(np.median(a)),p90=float(np.quantile(a,.9)),observations=len(a))
        results.append(dict(all=stats(sum(errors,[])),cameras={n:stats(e) for n,e in zip(ns,errors) if e}))
    result=dict(common_untouched_tracks=len(shared),unit='equivalent pixels at 3840 panorama width; angular consistency, not geometry ground truth',
                validation_role='Not used in optimization residuals; may be used in automatic update selection, so not an unbiased test set',old=results[0],new=results[1])
    out.mkdir(exist_ok=True,parents=True);(out/'comparison.json').write_text(json.dumps(result,indent=2))
    rows=''.join(f'<tr><td>{n}</td><td>{results[0]["cameras"][n]["median"]:.3f}</td><td>{results[1]["cameras"][n]["median"]:.3f}</td><td>{results[0]["cameras"][n]["p90"]:.3f}</td><td>{results[1]["cameras"][n]["p90"]:.3f}</td></tr>' for n in ns if n in results[0]['cameras'])
    # Remove the arbitrary common similarity only for the layout illustration.
    a=bs[1]['C'];b=bs[0]['C'];ac=a-a.mean(0);bc=b-b.mean(0)
    u,s,vt=np.linalg.svd(ac.T@bc);fix=np.diag([1,1,np.linalg.det(u@vt)])
    scale=np.sum(s*fix.diagonal())/np.sum(ac*ac);aligned=ac@(u@fix@vt)*scale+b.mean(0)
    _,_,axes=np.linalg.svd(bc,full_matrices=False)
    pos=[bc@axes[:2].T,(aligned-b.mean(0))@axes[:2].T]
    extent=max(np.max(np.abs(pos)),1e-9);colors=['#92a4b5','#69e3c8'];shapes=[]
    for k,p in enumerate(pos):
        for i,(x,y) in enumerate(p):
            xx=450+x/extent*350;yy=210-y/extent*160
            shapes.append(f'<circle cx="{xx}" cy="{yy}" r="{9 if k==0 else 4}" fill="{colors[k]}"/><text x="{xx+12}" y="{yy+(20 if k else -8)}" fill="{colors[k]}" font-size="16">{ns[i]}</text>')
    plot='<svg viewBox="0 0 900 420" style="width:100%;background:#101c27">'+''.join(shapes)+'</svg>'
    verdict=('共同特征的中位误差和 P90 均未上升；仍需验证实际漫游画质。' if all(results[1]['all'][k]<=results[0]['all'][k] for k in ['median','p90'])
             else '共同特征的指标并非全部改善，不能认定全面优于旧版。')
    html=f'''<!doctype html><meta charset="utf-8"><title>自动机位配准对照</title><style>body{{background:#18232f;color:#eef4fa;font:18px system-ui;max-width:1000px;margin:50px auto;padding:20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:14px;border-bottom:1px solid #526273}}a{{color:#80d4ff}}</style>
    <h1>自动机位配准：相同特征对照</h1><p>两版均未进入优化残差的 {len(shared)} 条完整特征轨迹；可用于自动选择更新幅度，因此属于验证集，不是盲测集。单位：3840 宽全景的等效像素误差，越低越好。</p>
    <table><tr><th>机位</th><th>旧版中位数</th><th>新版中位数</th><th>旧版 P90</th><th>新版 P90</th></tr>{rows}</table>
    <h2>机位布局</h2><p>灰色为旧版，绿色为新版。仅为对照消除了两版任意的公共旋转、平移和尺度，并投影到机位主平面；不代表实测房间尺寸。</p>{plot}
    <p>这是照片射线的一致性验证，不是房间真实尺寸或最终渲染画质验证。新版不会自动替换已认可的漫游和网格。{verdict}</p><a href="comparison.json">完整验证数据</a>'''
    (out/'index.html').write_text(html,encoding='utf8');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.old,a.new,a.out)
