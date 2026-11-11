"""Publish source-crop, raw merge and common-surface territory comparisons."""
from pathlib import Path
import json,shutil,argparse
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'four-room/fusion-territory'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--serve-root',type=Path);args=parser.parse_args()
    page=(ROOT/'fusion-detail-room.html').read_text(encoding='utf8').replace('fusion-detail','fusion-territory').replace('保边采样与原图细节','相机领地与来源裁切').replace('保边细节试验版','相机领地试验版')
    page=page.replace("const territory=", "const hybrid=new URLSearchParams(location.search).get('view')==='hybrid';\nconst painted=new URLSearchParams(location.search).get('view')==='painted';\nconst anchored=new URLSearchParams(location.search).get('view')==='anchored';\nconst raw=new URLSearchParams(location.search).get('view')==='raw';\nconst source=new URLSearchParams(location.search).get('source');const individual=/^[2-5]$/.test(source||'');\nconst territory=")
    page=page.replace("path:territory?", "path:hybrid?'/four-room/fusion-territory/hybrid-fused.splat':painted?'/four-room/fusion-territory/anchored-photo-fused.splat':anchored?'/four-room/fusion-territory/anchored-core-fused.splat':raw?'/four-room/fusion-territory/raw-core-fused.splat':individual?'/four-room/fusion-territory/cropped-'+source+'.splat':territory?")
    page=page.replace('<div id="positions">','<p><a href="?view=raw">原始 PLY 裁切直拼</a><a href="?source=2">单独 2</a><a href="?source=3">单独 3</a><a href="?source=4">单独 4</a><a href="?source=5">单独 5</a></p><div id="positions">')
    page=page.replace("(previous?'已接受基线':'相机领地试验版')", "(hybrid?'裁切、锚定、附色与窄缝补点':painted?'锚定并原图附色':anchored?'原始高斯射线锚定':raw?'原始 PLY 裁切直拼':individual?'裁切来源 '+source:previous?'已接受基线':'公共表面领地版')")
    page=page.replace('原始 PLY 裁切直拼</a>','原始 PLY 裁切直拼</a><a href="?view=anchored">原始高斯射线锚定</a>')
    page=page.replace('<div id="positions">','<p><a href="?view=painted">锚定并附色</a><a href="?view=hybrid">窄缝补点版</a></p><div id="positions">')
    page=page.replace('通用保边细节试验。没有物体专用贴图；未恢复所有缺失几何。','核心区固定，接缝窄带竞争。直拼原始 PLY 和公共表面重采样是两个独立对照。')
    page=page.replace('const status=document.querySelector', "const download=document.querySelector('a[download]');\nconst model=hybrid?'hybrid-fused':painted?'anchored-photo-fused':anchored?'anchored-core-fused':raw?'raw-core-fused':individual?'cropped-'+source:'fused';\ndownload.href=previous?'/four-room/baselines/2026-11-11-fusion-next/fused.ply':'/four-room/fusion-territory/'+model+'.ply';\nconst status=document.querySelector")
    (ROOT/'fusion-territory-room.html').write_text(page,encoding='utf8')
    mesh=(ROOT/'fusion-detail-mesh.html').read_text(encoding='utf8').replace('fusion-detail','fusion-territory')
    (ROOT/'fusion-territory-mesh.html').write_text(mesh,encoding='utf8')
    shutil.copy2(ROOT/'four-room/surface-fusion-v4/inspection-views.json',OUT/'inspection-views.json')
    territory=json.loads((OUT/'territory-report.json').read_text());crop=json.loads((OUT/'crop-report.json').read_text());export=json.loads((OUT/'export-report.json').read_text())
    rows=''.join(f"<tr><td>{r['station']}</td><td>{r['input_points']:,}</td><td>{r['core_points']:,}</td><td>{r['narrow_overlap_points']:,}</td><td><a href='/four-room/fusion-territory/cropped-{r['station']}.ply'>PLY</a></td></tr>" for r in crop['sources'])
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>相机领地裁切试验</title><style>body{{max-width:1000px;margin:32px auto;padding:16px;background:#17212c;color:#edf4fa;font:16px/1.8 system-ui}}a{{color:#8bd6ff}}img{{max-width:100%}}td,th{{padding:8px;border:1px solid #526477}}table{{border-collapse:collapse}}</style>
<h1>相机领地与原始 PLY 裁切</h1><p><a href='/fusion-territory-room.html'>公共表面领地版</a> · <a href='/fusion-territory-room.html?view=raw'>裁切原始 PLY 后直接拼接</a> · <a href='/fusion-territory-room.html?view=owners'>最终实际来源颜色</a> · <a href='/fusion-territory-room.html?version=previous'>已接受基线</a></p>
<p>先用标定的相机光心、可见性建立表面上的距离分区；核心区锁定，只允许相邻来源在 0.22 模型单位的表面邻接带内优化接缝。核心区越界变更：{territory['core_changes']}。区域不是平均四等分，也不是以点云重心裁切。场景仍沿用已接受的公共几何及标定，尚未从头联合训练原始高斯。</p>
<p>每份原始 PLY 先变换到共同坐标系，再按自身来源深度的可见性、与公共表面的距离，以及所属区域裁切。保留主领地及 0.055 模型单位的窄邻接重叠带，并约束近缝高斯的 3σ 足迹。这是近似的表面归属约束，不是对高斯椭球做精确几何切割。</p>
<table><tr><th>机位</th><th>原始点数</th><th>核心保留</th><th>窄重叠带</th><th>裁切下载</th></tr>{rows}</table>
<p>提供两个明确分开的结果：① <a href='/four-room/fusion-territory/raw-core-fused.ply'>原始高斯核心区直拼</a>，保留来源颜色，合并时不叠加次要重叠点；② <a href='/four-room/fusion-territory/fused.ply'>公共表面附色与重采样</a>，沿用已接受表面，应用新的区域标签与窄缝颜色处理。第二个不是把第一个直接平滑所得。</p>
<p>公共表面版仍有 {export['prior_texture_retained_faces']} 个面保留弱证据初值，领地图用灰色明确标记，不能声称整个结果严格来自四块纯照片领地。没有灯具、桌面、窗帘专用规则。本轮不改变已接受的几何形状。</p>
<p>第三个独立候选是 <a href='/fusion-territory-room.html?view=anchored'>原始高斯射线锚定版</a>：保留原始点的颜色与透明度，把可靠点沿各自拍摄射线投到公共表面，并用投影 Jacobian 变换协方差。它会移动原始高斯，但不修改作为约束的公共表面。超过位移上限或接近掠射的点不强行调整。<a href='/four-room/fusion-territory/anchored-core-fused.ply'>下载 PLY</a> · <a href='/four-room/fusion-territory/anchored-eval/details.jpg'>基线／射线锚定近景</a> · <a href='/four-room/fusion-territory/anchor-report.json'>锚定数据</a>。</p>
<p>进一步对照：<a href='/fusion-territory-room.html?view=painted'>按归属从原片重新附色</a>，只在深度和颜色一致的窄带内按光学厚度混合；<a href='/fusion-territory-room.html?view=hybrid'>窄缝补点版</a>，另对窄带内覆盖不足的位置补充公共表面高斯。补点范围限定在已支持的接缝表面，不补未知背面。<a href='/four-room/fusion-territory/hybrid-fused.ply'>下载补点版 PLY</a> · <a href='/four-room/fusion-territory/hybrid-eval/details.jpg'>近景对照</a> · <a href='/four-room/fusion-territory/hybrid-eval/compare.jpg'>中点对照</a> · <a href='/four-room/fusion-territory/seam-fill-report.json'>补点数量与限制</a>。</p>
<p><a href='/four-room/fusion-territory/original-floor-reference.jpg'>原始全景的桌面参考</a>：桌面横向接缝和部分电线双环在原图中确实存在，不能将它们一概作为重影删除。</p>
<p><a href='/four-room/fusion-territory/details.jpg'>基线／公共表面领地版近景</a> · <a href='/four-room/fusion-territory/raw-eval/details.jpg'>基线／原始 PLY 裁切直拼近景</a> · <a href='/four-room/fusion-territory/photo-compare.jpg'>原图三方对照</a> · <a href='/four-room/fusion-territory/compare.jpg'>四个中点</a></p>
<p><a href='/four-room/fusion-territory/territory-report.json'>分区数据</a> · <a href='/four-room/fusion-territory/crop-report.json'>裁切数据</a> · <a href='/four-room/fusion-territory/detail-validation.json'>来源视点验证</a></p>
<p>研究参考：<a href='https://www.ptgui.com/whatsnew/120000'>PTGui 最优接缝和零重叠融合</a>；<a href='https://github.com/nmoehrle/mvs-texturing'>MVS-Texturing 的视图选择与接缝校色</a>；<a href='https://github.com/GAP-LAB-CUHK-SZ/GaussReg/blob/main/gs_fusion.py'>GaussReg 中按点云中心距离裁切的原始实现</a>。本试验改用相机光心与可见表面，未声称复现 PTGui 私有算法。</p><img src='/four-room/fusion-territory/compare.jpg'></html>'''
    report=report.replace('<h1>相机领地与原始 PLY 裁切</h1>', '<h1>相机领地与原始 PLY 裁切</h1><p><strong>验收结论：本轮为实验结果，没有替换已接受基线。</strong>公共表面版部分桌面细节改善，但灯具破碎仍在；原始高斯裁切、锚定和补点候选在中点视角有明显墙面噪声、投影布空洞及物体形变，不推荐作为主结果。单靠缩小重叠区不能纠正来源几何的不一致。</p><p><a href="/four-room/fusion-territory/variants-report.json">固定相同有效像素的多方案数值对照</a>（仅来源视点，不能替代中点视觉验收）。</p>')
    (ROOT/'fusion-territory-report.html').write_text(report,encoding='utf8')
    if args.serve_root:
        for name in ['fusion-territory-room.html','fusion-territory-mesh.html','fusion-territory-report.html']:shutil.copy2(ROOT/name,args.serve_root/name)
    print('Published territory comparison',flush=True)

if __name__=='__main__':main()
