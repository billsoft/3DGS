"""Publish the isolated shape/opacity experiment, preserving accepted URLs."""
from pathlib import Path
import json,shutil
root=Path(__file__).resolve().parents[1];serve=Path('C:/Users/bills/.codex/hyworld21-pair')
page=(root/'refit-room.html').read_text(encoding='utf8')
page=page.replace('/four-room/refit-surface-guard','/four-room/shape-free')
page=page.replace('/four-room/baselines/2026-11-11-coverage-guard','/four-room/baselines/2026-11-12-refit-surface-guard')
page=page.replace('refit-room.html','shape-room.html').replace('refit-report.html','shape-report.html')
page=page.replace('稳形','形状修正').replace('<a href="?version=detail">清晰对照</a>','')
page=page.replace("const detail=query.get('version')==='detail';", "const detail=false;")
(root/'shape-room.html').write_text(page,encoding='utf8')
control=json.loads((root/'four-room/shape-control/eval/evaluation.json').read_text())
candidate=json.loads((root/'four-room/shape-free/eval/evaluation.json').read_text())
rows=''
for title,d in [('已接受基线',None),('继续训练对照',control),('朝向与透明度可调',candidate)]:
    s=candidate['summary'] if d is None else d['summary']
    rows+=f'<tr><td>{title}</td><td>{s["baseline_mean_ssim"] if d is None else s["candidate_mean_ssim"]:.5f}</td><td>{0 if d is None else s["new_holes"]}</td></tr>'
images=''.join(f'<h2>{n}</h2><a href="/four-room/shape-free/eval/{n}.jpg"><img src="/four-room/shape-free/eval/{n}.jpg"></a>' for n in ['detail-0','detail-1','source-3-2','source-2-5','mid-0-room','mid-2-ceiling','mid-3-table'])
report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>高斯形状对照试验</title><style>body{{max-width:1500px;margin:auto;padding:25px;background:#18222d;color:white;font:16px/1.8 system-ui}}a{{color:#9cdeff}}img{{width:100%}}td{{padding:10px;border:1px solid #567}}</style><h1>固定配准，检验高斯形状与透明度</h1>
<p><a href="/shape-room.html">新候选预览</a> · <a href="/shape-room.html?version=baseline">已接受基线</a> · <a href="/four-room/shape-free/fused.ply">下载 PLY</a></p>
<p>两组使用相同初值、相机、照片、训练步数和损失。相机和高斯中心位置均固定。对照组只优化颜色与尺寸；候选额外优化高斯朝向和透明度。没有物体专用规则，也没有补拍或合成图片。</p>
<table><tr><td>版本</td><td>原片平均 SSIM</td><td>新增低透明度像素</td></tr>{rows}</table>
<p>检查26个原机位视向及12个中间视向。照片也用于训练，分数不是独立重建准确率。原有桌底、细杆和部分轮廓问题仍存在；本试验没有重新求解相机，也没有证明配准误差为零。</p>
<p>1024分辨率26个来源方向均改善；512分辨率25/26改善，平均SSIM从0.90126升至0.90758。两档检查的新增低透明度像素均为0，定义为基线alpha大于0.95而候选低于0.5；不代表所有视觉缺陷或已有空洞消失。目视改善较小，灯体厚度、桌边毛糙及部分遮挡缺失仍在。原始机位分数不能替代中间视点真实照片。</p>
<p>图片依次为原图、已接受基线、候选；中间视点只有基线与候选。<a href="/four-room/shape-free/eval/evaluation.json">完整指标</a></p>{images}</html>'''
(root/'shape-report.html').write_text(report,encoding='utf8')
for name in ['shape-room.html','shape-report.html']:shutil.copyfile(root/name,serve/name)
