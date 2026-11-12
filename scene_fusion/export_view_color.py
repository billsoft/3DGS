"""Export a fixed-geometry SH candidate with explicit source-fidelity results."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from core import write_json
from sh_io import write_sh_ply,read_sh_ply

def run(a):
    with np.load(a.model) as data:base,sh=data['base'],data['sh']
    previous=np.load(a.reference/'final/optimized.npy')
    if sh.shape!=(len(base),9,3):raise ValueError('Browser export requires degree two SH')
    if base.shape!=previous.shape or not np.array_equal(base[:,:3],previous[:,:3]) or not np.array_equal(base[:,6:],previous[:,6:]):raise ValueError('Appearance-only export changed geometry or opacity')
    evaluation=json.loads((a.out/a.evaluation/'evaluation.json').read_text(encoding='utf8'))
    native=json.loads((a.out/a.native/'audit.json').read_text(encoding='utf8'))
    strict=json.loads((a.out/a.native/'strict-source.json').read_text(encoding='utf8'))
    digest=hashlib.sha256(a.model.read_bytes()).hexdigest()
    if evaluation['candidate_sha256']!=digest or native['candidate_sha256']!=digest:raise ValueError('Evaluation does not belong to this model')
    final=a.out/'final';final.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(a.model,final/'optimized.npz');write_sh_ply(final/'fused.ply',base,sh)
    restored,coeff=read_sh_ply(final/'fused.ply')
    if not np.array_equal(restored,base) or not np.array_equal(coeff,sh):raise ValueError('SH PLY round trip failed')
    validation=dict(points=len(base),degree=2,geometry_opacity_exact=True,full_sh_ply_exact=True,sha256=digest)
    if a.protected_appearance:
        mask=np.load(a.protected_appearance)
        if mask.dtype!=bool or mask.shape!=(len(base),):raise ValueError('Invalid protected appearance mask')
        if not np.array_equal(sh[mask,0],previous[mask,3:6]) or np.any(sh[mask,1:]):raise ValueError('Protected appearance changed')
        validation['protected_appearance_points']=int(mask.sum());validation['protected_appearance_exact']=True
    write_json(final/'validation.json',validation)
    for name in ['scene.json','aligned-cameras.npz']:
        shutil.copyfile(a.reference/name,a.out/name)
    (a.out/'data-512').mkdir(exist_ok=True)
    shutil.copyfile(a.reference/'data-512/dataset.json',a.out/'data-512/dataset.json')
    meta=json.loads((a.reference/'fusion.json').read_text(encoding='utf8'))
    meta.update(sh_degree=2,model_folder='final',method='Source panorama supervised SH2 fitting, fixed Gaussian geometry and opacity',
        preview_title=str(len(meta['stations']))+'机位 · 视角相关颜色试验',
        preview_status='二阶球谐颜色',preview_note='使用全部原始机位和上下视向拟合；仍有残留几何噪点，尚未达到所有原机位局部均不退化。',
        reference_scene=a.reference.name,reference_label='上一版固定颜色')
    write_json(a.out/'fusion.json',meta)
    strict_summary={k:strict[k] for k in ['passed','total_directions']};strict_summary['failed_directions']=len(strict['failed_directions']);strict_summary['details']=a.native+'/strict-source.json'
    write_json(a.out/'result.json',dict(summary='同一融合模型的点现在携带二阶球谐颜色，由全部来源全景图监督拟合；播放器实际读取并渲染这些方向系数。',
        limitations='来源照片参与拟合，不是独立验证真值。严格原机位局部保真检查尚未全部通过；地面纹理、遮挡边缘与部分中间视点仍有误差。没有使用照片覆盖层。',
        metrics=dict(evaluation=evaluation['summary'],native=native['summary'],strict_source=strict_summary,export=validation),
        native_report=a.native,evaluation_report=a.evaluation,
        comparisons=['source-'+meta['stations'][0]+'-'+str(k) for k in [3,4,5]],
        midpoints=[r['view'] for r in evaluation['views'] if r['role']=='novel'][:6]))
    print(json.dumps(validation),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--evaluation',required=True);p.add_argument('--native',required=True)
    p.add_argument('--protected-appearance',type=Path)
    run(p.parse_args())
