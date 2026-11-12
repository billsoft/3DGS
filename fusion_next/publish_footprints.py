"""Publish unsuccessful edge ablations without promoting either model."""
from pathlib import Path
import json,shutil,argparse
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'four-room/footprint-evaluation'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--serve-root',type=Path);args=parser.parse_args()
    report=json.loads((OUT/'validation.json').read_text())
    rows=''.join('<tr><td>'+r['view']+'</td>'+''.join(f"<td>{r['models'][m]['photo_ssim']:.4f}</td>" for m in ['native','baseline','recolor','split'])+'</tr>' for r in report['views'])
    images=''.join(f"<h2>{r['view']}</h2><a href='/four-room/footprint-evaluation/comparison-{i}.jpg'><img src='/four-room/footprint-evaluation/comparison-{i}.jpg'></a>" for i,r in enumerate(report['views']))
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>跨物体边缘退化试验</title><style>body{{max-width:1400px;margin:30px auto;padding:18px;background:#17212c;color:#edf4fa;font:16px/1.8 system-ui}}a{{color:#9cdeff}}img{{width:100%}}td,th{{padding:8px;border:1px solid #526477}}table{{border-collapse:collapse}}</style>
<h1>灯、窗帘、桌沿和白板：同机位消融试验</h1>
<p><strong>两个新候选均未通过视觉验收，没有替换已接受版本。</strong>原始 PLY 的轮廓更接近原片；只改颜色几乎不修复融合轮廓，细分高斯在六个视点的 SSIM 均下降。结果支持把主要排查方向转向融合表面位置、深度不连续及遮挡，而非仅调整颜色和羽化。</p>
<p>所有行从左到右：<b>原图、原始 PLY、已接受融合、边缘重新赋色、边缘细分并赋色</b>。使用相同拍摄点、方向和90度视角。点击图片可看原尺寸。</p>
<p>算法没有灯具、窗帘或桌子专用分支。根据原图在高斯覆盖范围内的颜色变化、可见深度和置信度，选择102,442个跨边缘高斯。赋色组保留全部形状和透明度；细分组每个选中点拆成9个，维持混合分布的一阶、二阶矩，并近似保留光学厚度积分。后者不保证逐像素透明度不变。总点数由3,166,125增至3,985,661，但增加点数没有带来正确轮廓。</p>
<table><tr><th>检查视向</th><th>原始 PLY SSIM</th><th>融合基线</th><th>只赋色</th><th>细分＋赋色</th></tr>{rows}</table>
<p>分数相对原图，固定使用融合基线高覆盖像素；新增空洞也计入误差。它不是未知视点的几何正确率。试验未证明任何平滑方法都无效，只否决本轮两种参数化处理。</p>
<p><a href='/fusion-native-room.html'>已验证的机位保真播放</a> · <a href='/four-room/footprint-evaluation/validation.json'>完整指标</a> · <a href='/four-room/footprint-recolor/footprint-report.json'>赋色参数</a> · <a href='/four-room/footprint-split/footprint-report.json'>细分参数</a></p>
{images}</html>'''
    geometry='<h2>关键定位：绕过高斯，直接渲染带贴图网格</h2><p>下列顺序为原图、原始 PLY、带贴图网格、融合高斯。网格中已存在灯与窗帘的破碎和错层，说明退化在网格及贴图阶段已经发生；本对照尚不能将几何误差与 UV／来源贴图错配完全分离。</p>'
    for i in [0,2,3]:geometry+=f"<a href='/four-room/footprint-evaluation/mesh-diagnostic-{i}.jpg'><img src='/four-room/footprint-evaluation/mesh-diagnostic-{i}.jpg'></a>"
    html=html.replace('<h1>灯、窗帘、桌沿和白板：同机位消融试验</h1>','<h1>灯、窗帘、桌沿和白板：同机位消融试验</h1>'+geometry)
    (ROOT/'footprint-report.html').write_text(html,encoding='utf8')
    if args.serve_root:shutil.copy2(ROOT/'footprint-report.html',args.serve_root/'footprint-report.html')

if __name__=='__main__':main()
