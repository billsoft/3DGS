from common import *
import hashlib
def run():
    appearance=json.loads((OUT/'appearance-report.json').read_text())
    geometry=json.loads((OUT/'geometry-report.json').read_text());selection=json.loads((OUT/'ownership-report.json').read_text());export=json.loads((OUT/'export-report.json').read_text());flow=json.loads((OUT/'flow-report.json').read_text())
    assert geometry['flipped_faces']==0
    assert selection['final_energy']<=selection['initial_energy']
    assert 0<=export['maximum_secondary_weight']<=.45
    manifest=[]
    for n in stations:
        for name in [f'{n}_7680.jpg',f'{n}.ply']:
            path=rm.inputs/name;manifest.append(dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write_json('inputs.json',manifest)
    page=(ROOT/'surface-sharp-room.html').read_text(encoding='utf8').replace('/four-room/surface-fusion-v3/','/four-room/fusion-next/').replace("previous?'/four-room/surface-fusion-v2/fused.splat'","previous?'/four-room/surface-fusion-v3/fused.splat'")
    page=page.replace("const previous=", "const territory=new URLSearchParams(location.search).get('view')==='owners';\nconst previous=")
    page=page.replace("path:previous?", "path:territory?'/four-room/fusion-next/owners.splat':previous?")
    page=page.replace('清晰度与区域归属优化','通用对应与接缝融合').replace('清晰度优化版','通用接缝版').replace('上一版公共表面','V3 对照版').replace('/surface-sharp-mesh.html','/fusion-next-mesh.html')
    page=page.replace('<div id="positions">','<p><a href="?view=owners">查看三维领地</a><a href="/fusion-next-report.html">算法与验证</a></p><p>领地：红=2，绿=3，蓝=4，黄=5，灰=保留原有效贴图。</p><div id="positions">')
    page=page.replace('同视点对照</a>','同视点对照</a><a href="/four-room/fusion-next/details.jpg">近景对照</a>')
    page=page.replace('部分细杆与椅背仍可能缺失；新版本不是四份透明点云叠加。','通用优化试验版。弱证据区域保留初值；尚未解决所有缺损与灯具变形。')
    (ROOT/'fusion-next-room.html').write_text(page,encoding='utf8')
    meshpage=(ROOT/'surface-sharp-mesh.html').read_text(encoding='utf8').replace('/four-room/surface-fusion-v3/','/four-room/fusion-next/').replace('/surface-sharp-room.html','/fusion-next-room.html').replace('原片清晰贴图网格','通用接缝融合网格')
    (ROOT/'fusion-next-mesh.html').write_text(meshpage,encoding='utf8')
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>通用对应与三维接缝融合</title><style>body{{max-width:1000px;margin:35px auto;padding:20px;font:16px/1.8 system-ui;background:#151e28;color:#e3ebf5}}a{{color:#89d7ff}}img{{max-width:100%}}td,th{{border:1px solid #58616d;padding:9px}}table{{border-collapse:collapse}}</style>
<h1>通用对应与三维接缝融合</h1><p><a href="/fusion-next-room.html">预览</a> · <a href="/fusion-next-room.html?view=owners">三维领地</a> · <a href="/four-room/fusion-next/fused.ply">融合 PLY</a></p>
<p>这次重构没有灯具模板，也没有按物体名称指定区域。初始化沿用 V3 较好的表面，其中包含此前电视、球体、桌面的先验拟合；因此并非完全从原始 PLY 无先验恢复。四张原片和原始高斯深度用于新对应，已有 collider 用于辅助深度置信度。导出的是重新采样的表面高斯，没有进行原始 3DGS 的可微联合训练。</p>
<ol><li>深度重投影后的双向 DIS 光流，筛除遮挡、弱纹理、过曝、双向不一致和三角化不一致点：共 {flow['total']} 个候选对应。</li><li>沿表面邻接图约束局部法向位移，自动冻结会翻转或塌缩的位置；用空间块留出的对应点检验。用于几何检查的留出点 {geometry['held']} 个，法向残差中位数 {geometry['held_median_before']:.5f} → {geometry['held_median_after']:.5f}，90 分位 {geometry['held_p90_before']:.5f} → {geometry['held_p90_after']:.5f}。单位是模型坐标，不是米；对应点也不是真实扫描深度。</li><li>所有大三角面统一细分，再用公共几何射线遮挡检查和生成深度置信度选择可用照片。图优化显式计算接缝颜色代价与区域连贯性，能量 {selection['initial_energy']:.1f} → {selection['final_energy']:.1f}。这个能量下降不等于画质提高同样比例。</li><li>空间归属固定，沿表面邻接关系限制接缝带。只有颜色、深度、密集匹配都支持的部分才羽化：最终 {export['blend_fraction']:.2%} 的高斯发生边界颜色混合，区域内部保持单一来源。</li><li>新证据不足、或新颜色与已有有效贴图严重冲突时保留初值，共 {export['prior_texture_retained_faces']} 个面片；不保留原来纯灰色的未知占位面。领地图的灰色表示这项保留机制，不是新增灰色几何。</li></ol>
<p>四张照片另做受限 RGB 增益与偏置校正，固定第一张作参考，不合成纹理。留出点色差中位数 {appearance['median_before']:.5f} → {appearance['median_after']:.5f}（RGB 范围 0–1）；校正是否通过验证：{appearance['accepted']}。<a href="/four-room/fusion-next/appearance-report.json">颜色校正数据</a></p>
<p>在低纹理、几何连续的可靠接缝上增加局部颜色校正：在同一三维位置比较两张照片，用表面邻接图平滑传播颜色偏置，每通道调整不超过 0.045。原照片的高频纹理不做模糊处理。这项校正写入 PLY；网格预览仍使用原照片贴图。<a href="/four-room/fusion-next/seam-level-report.json">接缝颜色校正数据</a>（这里的接缝残差是拟合指标，不是独立画质评分）。</p>
<p>几何修正验证：{geometry['accepted']}；翻转面片：{geometry['flipped_faces']}。光流在白灯过曝区和弱纹理区仍可能没有可靠对应，故不能承诺灯具已修复。桌底、细杆和未知背面仍受观测不足限制。已有良好结构的保留，部分依赖初始化先验。</p>
<p><a href="/four-room/fusion-next/compare.jpg">四个相同中点对照（左 V3，右通用版）</a> · <a href="/four-room/fusion-next/details.jpg">灯具／电视背后近景</a></p><img src="/four-room/fusion-next/compare.jpg">
<p><a href="/four-room/fusion-next/flow-report.json">对应点报告</a> · <a href="/four-room/fusion-next/geometry-report.json">几何检查</a> · <a href="/four-room/fusion-next/ownership-report.json">归属优化</a> · <a href="/four-room/fusion-next/export-report.json">边界融合</a> · <a href="/four-room/fusion-next/inputs.json">原始输入校验</a></p></html>'''
    (ROOT/'fusion-next-report.html').write_text(report,encoding='utf8');print('published generic fusion',flush=True)
if __name__=='__main__':run()
