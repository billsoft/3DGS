"""Publish a separate experiment page, keeping the accepted page unchanged."""
from pathlib import Path
import json,shutil,argparse
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'four-room/fusion-detail'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--serve-root',type=Path);args=parser.parse_args()
    base='/four-room/fusion-detail/'
    frozen='/four-room/baselines/2026-11-11-fusion-next/'
    page=(ROOT/'fusion-next-room.html').read_text(encoding='utf8')
    page=page.replace('/four-room/fusion-next/',base).replace('/four-room/surface-fusion-v3/',frozen)
    page=page.replace('/fusion-next-mesh.html','/fusion-detail-mesh.html').replace('/fusion-next-report.html','/fusion-detail-report.html')
    page=page.replace('V3 对照版','已接受基线').replace('通用接缝版','保边细节试验版').replace('通用对应与接缝融合','保边采样与原图细节')
    page=page.replace('通用优化试验版。弱证据区域保留初值；尚未解决所有缺损与灯具变形。','通用保边细节试验。没有物体专用贴图；未恢复所有缺失几何。')
    page=page.replace('同视点对照</a>','同视点对照</a><a href="'+base+'photo-compare.jpg">原图三方对照</a>')
    insert="""const inspection=await (await fetch('/four-room/fusion-detail/inspection-views.json')).json();
 for(const v of inspection){const button=document.createElement('button');button.textContent='检查 '+v.name;button.onclick=()=>{movement++;viewer.camera.position.fromArray(v.position);viewer.controls.target.fromArray(v.target);viewer.controls.update();viewer.forceRenderNextFrame();};panel.appendChild(button);}
 """
    page=page.replace("viewer.controls.addEventListener('start'",insert+"viewer.controls.addEventListener('start'")
    (ROOT/'fusion-detail-room.html').write_text(page,encoding='utf8')
    mesh=(ROOT/'fusion-next-mesh.html').read_text(encoding='utf8').replace('/four-room/fusion-next/',base).replace('/fusion-next-room.html','/fusion-detail-room.html')
    (ROOT/'fusion-detail-mesh.html').write_text(mesh,encoding='utf8')
    shutil.copy2(ROOT/'four-room/surface-fusion-v4/inspection-views.json',OUT/'inspection-views.json')
    detail=json.loads((OUT/'detail-export-report.json').read_text());smooth=json.loads((OUT/'smooth-report.json').read_text())
    smoothing_note=(f"保边几何平滑：冻结 {smooth['frozen_unstable_vertices']} 个不稳定顶点；最大位移 {smooth['max_candidate_displacement']:.4f} 模型单位，异常面片 {smooth['invalid_faces']}。" if smooth.get('enabled',True) else '最终默认不启用几何平滑，完整保留已接受基线的几何。消融试验中平滑的额外收益不足，故没有将其作为默认改动。可选平滑代码和试验数据仍保留。')
    checks=json.loads((OUT/'detail-validation.json').read_text());rows=[]
    for row in checks['views']:
        if 'baseline_ssim' in row:rows.append(f"<tr><td>{row['view']}</td><td>{row['baseline_ssim']:.4f}</td><td>{row['detail_ssim']:.4f}</td><td>{row['baseline_mae']:.4f}</td><td>{row['detail_mae']:.4f}</td></tr>")
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>通用保边细节试验</title><style>body{{max-width:1000px;margin:32px auto;padding:16px;background:#17212c;color:#eef4fa;font:16px/1.8 system-ui}}a{{color:#8bd6ff}}img{{max-width:100%}}td,th{{padding:8px;border:1px solid #586878}}table{{border-collapse:collapse}}</style>
<h1>通用保边细节试验</h1><p><a href="/fusion-detail-room.html">预览新结果</a> · <a href="/fusion-detail-room.html?version=previous">已接受基线</a> · <a href="{base}fused.ply">下载 PLY</a></p>
<p>所有区域使用同一算法；没有灯具、窗帘的专用分割或合成贴图。初始化仍继承此前的 V3 几何先验。照片负责区域、弱证据回退机制沿用已接受基线。</p>
<ol><li>按原图梯度与投影像素跨度增加采样，每来源样本预算不超过原来的 1.8 倍。输出 {detail['splats']:,} 个高斯。</li><li>在图像边缘法向收缩高斯覆盖，沿边缘方向保持覆盖；{detail['contracted_splats']:,} 个高斯的边缘强度超过阈值。只改变采样与协方差，不给物体套规则形状。</li><li>接缝只渐变调整低频颜色差，高频纹理保留主照片信息；没有反锐化或 AI 补纹理。</li><li>{smoothing_note}</li></ol>
<p>本算法没有重建缺失背面，也没有修复所有错误拓扑。灯具、窗帘仍可能残留锯齿、灰色和形变。高斯增加会提高浏览器内存和加载时间。网格预览用于检查表面，它不包含高斯足迹收缩和局部颜色偏置。</p>
<h2>原图同相机验证</h2><p>在基线和新版都具备高覆盖率的共同像素上比较原图。SSIM 越高越好、平均绝对色差 MAE 越低越好；这些是来源视点指标，不代表所有漫游位置的画质。</p><table><tr><th>视点</th><th>基线 SSIM</th><th>新版 SSIM</th><th>基线 MAE</th><th>新版 MAE</th></tr>{''.join(rows)}</table>
<p><a href="{base}photo-compare.jpg">原图／基线／新版三方对照</a> · <a href="{base}compare.jpg">四个中点对照</a> · <a href="{base}detail-validation.json">验证数据</a> · <a href="{base}smooth-report.json">平滑检查</a></p><p>近景：左边为已接受基线，右边为新试验。</p><img src="{base}details.jpg"></html>'''
    (ROOT/'fusion-detail-report.html').write_text(report,encoding='utf8')
    if args.serve_root:
        target=args.serve_root.resolve()
        if not target.is_dir():raise ValueError('Serve root must exist')
        for name in ['fusion-detail-room.html','fusion-detail-mesh.html','fusion-detail-report.html']:shutil.copy2(ROOT/name,target/name)
    print('Published separate detail experiment',flush=True)

if __name__=='__main__':main()
