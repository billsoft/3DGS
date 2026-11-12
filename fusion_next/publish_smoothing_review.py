"""Publish rejected smoothing experiments without replacing the working player."""
import json,shutil
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=root/'experiments/2026-11-12-smoothing';out.mkdir(parents=True,exist_ok=True)
runs=[('局部点到面平滑','surface-smooth/eval-medium'),
      ('法线位移平滑后重新拟合颜色','normal-refit/eval'),
      ('局部朝向异常修正','normal-consensus/eval')]
rows=[]
for title,folder in runs:
    data=json.loads((root/'four-room'/folder/'evaluation.json').read_text())
    (out/(folder.split('/')[0]+'.json')).write_text(json.dumps(data,indent=2))
    s=data['summary'];rows.append(f'<tr><td>{title}</td><td>{s["candidate_mean_ssim"]:.6f}</td><td>{s["new_holes"]}</td></tr>')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>毛刺平滑试验 · 未替换当前版本</title>
<style>body{max-width:1500px;margin:auto;padding:24px;background:#17222c;color:#eee;font:16px/1.8 system-ui}a{color:#9cdeff}img{width:100%}td,th{padding:10px;border:1px solid #567}table{border-collapse:collapse}</style>
<h1>毛刺平滑试验：当前版本保留</h1><p><a href="/shape-room.html">返回当前模型</a></p>
<p>尝试局部点到面平滑、邻域朝向与长度约束、法线位移平滑后重新拟合颜色，以及更局部的朝向异常修正。没有新拟合地球仪球体或添加灯具贴图。位移方法依赖已有几何初始化；该历史初始化本身包含此前物体先验，不能将其描述为纯照片端到端重建。</p>
<p>最保守方法对尖刺改善不明显；较强平滑损失原机位细节并产生新低透明度像素。本轮没有通过发布标准的候选，因此没有替换模型，也没有把失败版本设为默认。</p>
<table><tr><th>方法</th><th>来源视向平均 SSIM</th><th>新增低透明度像素</th></tr><tr><td>当前 shape 版</td><td>0.929739</td><td>—</td></tr>'''+''.join(rows)+'''</table>
<p>量化检查为26个来源方向和12个中间方向，分辨率1024。来源照片参与拟合，分数不是独立场景精度。低透明度像素指基线 alpha&gt;0.95 而候选 alpha&lt;0.5；不代表所有视觉缺陷。朝向与长度约束另做近景目视检查，未做完整38方向量化，未列入表格。</p>
<h2>较强法线位移平滑：左为当前，右为候选（重新拟合颜色前）</h2><img src="/four-room/normal-smooth/close-medium/globe-front.jpg">
<h2>局部朝向异常修正：左为当前，右为候选</h2><img src="/four-room/normal-consensus/close/globe-front.jpg"><img src="/four-room/normal-consensus/close/globe-side.jpg">
<p>两侧仍能看到明显尖刺。本次实验未证明毛刺可以通过简单后处理可靠去除，也未证明以后无法改善。后续应把几何平滑与多视图投影误差、遮挡约束联合求解，而不是把已经拟合好的点云整体抹平。</p></html>'''
(root/'smoothing-review.html').write_text(page,encoding='utf8')
shutil.copyfile(root/'smoothing-review.html',Path('C:/Users/bills/.codex/hyworld21-pair/smoothing-review.html'))
