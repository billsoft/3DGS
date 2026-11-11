"""Publish the inspected coverage-preserving result without changing baseline pages."""
from pathlib import Path
import json,shutil,argparse
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'four-room/fusion-coverage-guarded'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--serve-root',type=Path);args=parser.parse_args()
    audit=json.loads((OUT/'coverage-audit.json').read_text());report=json.loads((OUT/'coverage-report.json').read_text())
    if not audit['coverage_pass']:raise ValueError('Coverage regression blocks publication')
    if audit['changed_region_regressions']:raise ValueError('Changed-region regression blocks publication')
    page=(ROOT/'fusion-detail-room.html').read_text(encoding='utf8')
    page=page.replace('fusion-detail','fusion-coverage-guarded').replace('保边采样与原图细节','覆盖保护与局部回退').replace('保边细节试验版','覆盖保护修正版')
    page=page.replace('<a href="/fusion-coverage-guarded-mesh.html">网格版</a>','<a href="/fusion-territory-room.html">上一轮领地试验</a>')
    page=page.replace('查看三维领地','查看实际改色范围').replace('领地：红=2，绿=3，蓝=4，黄=5，灰=保留原有效贴图。','绿色=通过检查的局部改色；灰色=保留基线。')
    page=page.replace('通用保边细节试验。没有物体专用贴图；未恢复所有缺失几何。','本轮主要恢复基线完整度，画面接近基线。保留全部覆盖，仅小范围改色；原有灯具破碎和未知背面仍在。')
    page=page.replace('const status=document.querySelector',"if(previous)document.querySelector('a[download]').href='/four-room/baselines/2026-11-11-fusion-next/fused.ply';\nconst status=document.querySelector")
    (ROOT/'fusion-coverage-guarded-room.html').write_text(page,encoding='utf8')
    shutil.copy2(ROOT/'four-room/surface-fusion-v4/inspection-views.json',OUT/'inspection-views.json')
    rows=''.join(f"<tr><td>{r['view']}</td><td>{r['new_hole_pixels']}</td><td>{r['maximum_alpha_difference']:.6f}</td><td>{r.get('changed_pixel_mae_delta','—')}</td></tr>" for r in audit['views'])
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>覆盖保护验证</title><style>body{{max-width:1000px;margin:30px auto;padding:20px;background:#17212c;color:#edf4fa;font:16px/1.8 system-ui}}a{{color:#8bd6ff}}img{{max-width:100%}}td,th{{padding:8px;border:1px solid #526477}}table{{border-collapse:collapse}}</style>
<h1>覆盖保护与逐点回退</h1><p><a href='/fusion-coverage-guarded-room.html'>预览修正版</a> · <a href='/fusion-next-room.html'>已接受基线</a> · <a href='/fusion-territory-room.html'>上一轮领地版</a> · <a href='/four-room/fusion-coverage-guarded/fused.ply'>下载 PLY</a></p>
<p><strong>本轮主要修复流程退化，没有证实画质大幅超越基线。</strong>保留基线全部 {report['points']:,} 个高斯的位置、尺度、旋转及透明度。相机分区只提名颜色来源，不再触发删点。画面接近基线是有意的安全回退，不能把它包装成全新几何重建。</p>
<p>放宽来源接管曾改色 {report['proposed_updated_points']:,} 个点；对每点投回多个可信原始照片后，回退 {report['rolled_back_points']:,} 个，仅保留 {report['updated_points']:,} 个。必须至少两个可信观察平均颜色误差下降，且任何可信观察的退化受限。照片是优化输入，不是独立真值。</p>
<p>固定检查包括两处灯、电视背面、7 个来源视向和4个机位中点。逐像素覆盖检查没有新增空洞，四中点空洞比例与基线一致；电视背面原有缺失和灯具破碎仍存在。这里的保护约束只保证不因本轮删除或缩点变差，不保证全部颜色更好。</p>
<h2>来源与近景覆盖检查</h2><table><tr><th>视点</th><th>新增空洞像素</th><th>最大透明度差</th><th>实际改动像素颜色误差变化</th></tr>{rows}</table><p>最后一列负值更好；未改动或无原片参考的视点不计。整体平均分可能掩盖小范围退化，因此另算实际改动像素。</p>
<p><a href='/four-room/fusion-coverage-guarded/coverage-report.json'>算法参数与回退数量</a> · <a href='/four-room/fusion-coverage-guarded/coverage-audit.json'>逐像素审计</a> · <a href='/four-room/fusion-coverage-guarded/detail-validation.json'>来源图像指标</a> · <a href='/four-room/fusion-coverage-wide/coverage-audit.json'>被否决的放宽版本</a></p>
<h2>基线与修正版：四个中点</h2><img src='/four-room/fusion-coverage-guarded/compare.jpg'><h2>灯与电视背面</h2><img src='/four-room/fusion-coverage-guarded/details.jpg'></html>'''
    (ROOT/'fusion-coverage-guarded-report.html').write_text(html,encoding='utf8')
    if args.serve_root:
        for name in ['fusion-coverage-guarded-room.html','fusion-coverage-guarded-report.html']:shutil.copy2(ROOT/name,args.serve_root/name)
    print('Published inspected coverage candidate',flush=True)

if __name__=='__main__':main()
