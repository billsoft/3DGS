"""Publish a photograph-refitted single GS only after frozen-geometry checks."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from core import write_json
from sh_io import write_sh_ply,read_sh_ply


def run(a):
    z=np.load(a.out/'optimized.npz');base,sh=z['base'],z['sh']
    old=np.load(a.reference/'optimized.npz')['base']
    if base.shape!=old.shape or not np.array_equal(base[:,:3],old[:,:3]) or not np.array_equal(base[:,6:],old[:,6:]):
        raise ValueError('Appearance fitting changed geometry or opacity')
    evaluation=json.loads((a.out/'evaluation/evaluation.json').read_text('utf8'))
    digest=hashlib.sha256((a.out/'optimized.npz').read_bytes()).hexdigest()
    if evaluation['candidate_sha256']!=digest:raise ValueError('Stale evaluation')
    final=a.out/'final';final.mkdir(exist_ok=True)
    write_sh_ply(final/'fused.ply',base,sh);b,s=read_sh_ply(final/'fused.ply')
    if not np.array_equal(b,base) or not np.array_equal(s,sh):raise ValueError('PLY round trip failed')
    for folder in ['data','data-512']:
        (a.out/folder).mkdir(exist_ok=True)
        shutil.copyfile(a.reference/folder/'dataset.json',a.out/folder/'dataset.json')
    meta=json.loads((a.reference/'fusion.json').read_text('utf8'))
    meta.update(report_page='scene-report.html',evaluation_folder='evaluation',reference_scene=a.reference.name,
        reference_label='之前认可的单一 PLY',preview_title='固定几何 · 原片高清附色试验',
        preview_status='单一 PLY，几何与透明度保持不变',
        preview_note='原始全景图监督颜色；没有照片覆盖层。细小结构和离开机位后的误差仍需检查。')
    write_json(a.out/'fusion.json',meta)
    validation=dict(points=len(base),sh_degree=int(sh.shape[1]**.5)-1,geometry_opacity_exact=True,roundtrip_exact=True,
        model_sha256=digest,ply_sha256=hashlib.sha256((final/'fused.ply').read_bytes()).hexdigest())
    write_json(final/'validation.json',validation)
    write_json(a.out/'result.json',dict(summary='保留已认可的单一 PLY 的全部几何与透明度，只使用原始全景图重新拟合方向相关颜色。',
        limitations='原片参与训练，来源机位指标是拟合度，不是未见视点的准确度保证。高斯大小固定，不能仅靠颜色恢复所有细纹理。',
        comparison_caption='原始照片 / 之前认可的单一 PLY / 固定几何高清附色。均为同一相机视角，后两列是实际高斯渲染。',
        metrics=dict(photographs=evaluation['summary'],export=validation),native_report='evaluation',evaluation_report='evaluation',
        comparisons=[r['view'] for r in evaluation['views'] if 'candidate' in r],
        midpoints=[r['view'] for r in evaluation['views'] if r['role']=='novel']))
    print(json.dumps(validation),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
