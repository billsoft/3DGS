"""Publish source fidelity evidence and the explicitly view-dependent preview."""
from pathlib import Path
import json,shutil,argparse
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'four-room/fusion-native-fidelity'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--serve-root',type=Path);args=parser.parse_args()
    report=json.loads((OUT/'fidelity-report.json').read_text());alignment=json.loads((OUT/'alignment-validation.json').read_text())
    if not alignment['passed']:raise ValueError('Original coordinate fidelity check failed')
    variants={name:[r for r in report['views'] if r['variant']==name] for name in ['accepted','native_owner']}
    variants['previous_cropped_merge']=json.loads((ROOT/'four-room/fusion-native-previous/native-validation.json').read_text())['views']
    variants['native_visible']=json.loads((ROOT/'four-room/fusion-native-visible/native-validation.json').read_text())['views']
    variants['native_fill']=json.loads((ROOT/'four-room/fusion-native-fill/native-validation.json').read_text())['views']
    summary={name:dict(views=len(rows),new_hole_pixels=sum(r['new_hole_pixels'] for r in rows),mean_source_mae=sum(r['source_mask_mae'] for r in rows)/len(rows),mean_source_ssim=sum(r['source_mask_ssim'] for r in rows)/len(rows)) for name,rows in variants.items()}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    names={'accepted':'已接受融合基线','native_owner':'原始高斯按领地直拼','previous_cropped_merge':'上一轮裁切后的完整合并','native_visible':'保留形状＋单向遮挡筛除','native_fill':'原始领地＋有观测支持的补洞'}
    rows=''.join(f"<tr><td>{names[name]}</td><td>{r['new_hole_pixels']:,}</td><td>{r['mean_source_mae']:.4f}</td><td>{r['mean_source_ssim']:.4f}</td></tr>" for name,r in summary.items())
    links=''.join(f"<li>{s}号机位：<a href='/four-room/fusion-native-fidelity/station-{s}.jpg'>原始／单份裁切／基线／原始领地直拼</a> · <a href='/four-room/fusion-native-visible/station-{s}.jpg'>修正后静态候选</a> · <a href='/four-room/fusion-native-previous/station-{s}.jpg'>上一轮完整裁切合并</a></li>" for s in ['2','3','4','5'])
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>原始机位保真测试</title><style>body{{max-width:1100px;margin:30px auto;padding:16px;background:#17212c;color:#edf4fa;font:16px/1.8 system-ui}}a{{color:#9cdeff}}td,th{{padding:8px;border:1px solid #526477}}table{{border-collapse:collapse}}img{{max-width:100%}}</style>
<h1>原始机位保真测试与预览</h1><p><a href='/fusion-native-room.html'>机位保真播放器</a> · <a href='/fusion-coverage-guarded-room.html'>已接受单模型基线</a></p>
<p><strong>静态融合候选仍未达到“各原始机位不退化”，未替换基线。</strong>本轮提供的可用改进是视点相关播放：在拍摄点及小邻域，完整显示该来源原始高斯；中点回到已接受模型。没有给灯或桌子单独贴图，没有把原始高斯改造成公共网格。它不是一个新的统一 PLY。</p>
<p>原始点云仅进行已有相似变换；四机位×六方向的变换前后渲染通过一致性检查。这证明刚性对齐本身没有造成明显源视图损失，不证明四份独立生成几何彼此完全一致。</p>
<h2>24个来源视向的静态融合检查</h2><table><tr><th>版本</th><th>原始有覆盖、融合后缺失的像素总数</th><th>相对原始高斯的平均颜色误差 ↓</th><th>平均 SSIM ↑</th></tr>{rows}</table>
<p>参考是各自完整原始 PLY 的渲染，不是真实扫描。所有方向使用384×384分辨率和相同相机；新增空洞计原始 alpha&gt;0.95、候选 alpha&lt;0.5。单份裁切列只是消融诊断，不能当作完整融合结果；完整旧合并也另行验证。</p><ul>{links}</ul>
<h2>播放器策略与限制</h2><p>相机距离本机位小于最近相邻机位间距的8%时使用完整原始场景；超过24%时使用融合基线，中间做平滑屏幕颜色过渡。同一时刻最多一个原始场景参与显示。两套几何不会合并为新的高斯层，但过渡期间仍可能出现屏幕双影。阈值是保守工程参数，没有证明整个邻域都具备正确几何。</p>
<p>在机位中心原始场景权重为1，在四个相邻中点为0。拍摄点的原有缺陷也会保留；中间区域原有错误仍在。播放器加载两套渲染器，对内存和显卡要求高于单模型预览。</p>
<p><a href='/four-room/fusion-native-fidelity/alignment-validation.json'>24方向变换前后验证</a> · <a href='/four-room/fusion-native-fidelity/summary.json'>静态候选汇总</a> · <a href='/four-room/fusion-native-visible/fused.ply'>未通过验收的静态候选 PLY（仅供诊断）</a></p>
<h2>2号机位：原始／基线／静态候选</h2><img src='/four-room/fusion-native-visible/station-2.jpg'></html>'''
    if (OUT/'browser-auto.png').exists() and (OUT/'browser-baseline.png').exists():
        html=html.replace('<h2>24个来源视向的静态融合检查</h2>',"<h2>实际浏览器同机位对照</h2><p>原有融合模型：</p><img src='/four-room/fusion-native-fidelity/browser-baseline.png'><p>机位保真模式：显示原始高斯，电视支架和桌沿保留完整。</p><img src='/four-room/fusion-native-fidelity/browser-auto.png'><h2>24个来源视向的静态融合检查</h2>")
    html=html.replace("<h2>播放器策略与限制</h2>","<p><a href='/fusion-native-static-room.html'>未通过验收的静态补洞候选预览</a> · <a href='/four-room/fusion-native-fill/compare.jpg'>静态候选四个中点对照</a></p><h2>播放器策略与限制</h2>")
    (ROOT/'fusion-native-report.html').write_text(html,encoding='utf8')
    page=(ROOT/'fusion-coverage-guarded-room.html').read_text(encoding='utf8').replace('fusion-coverage-guarded','fusion-native-fill')
    page=page.replace('覆盖保护与局部回退','原始高斯补洞 · 未通过验收').replace('覆盖保护修正版','静态补洞候选（未通过验收）')
    page=page.replace('<a href="?view=owners">查看实际改色范围</a>','').replace('绿色=通过检查的局部改色；灰色=保留基线。','此静态候选仍有电视错误遮挡，未替换基线。')
    page=page.replace('本轮主要恢复基线完整度，画面接近基线。保留全部覆盖，仅小范围改色；原有灯具破碎和未知背面仍在。','保留原始高斯形状并补充有观测支持的缺口；不保证所有机位完整，电视等区域仍不正确。')
    page=page.replace('/fusion-native-fill-report.html','/fusion-native-report.html').replace('/four-room/fusion-native-fill/photo-compare.jpg','/four-room/fusion-native-fill/station-2.jpg').replace('/four-room/fusion-native-fill/details.jpg','/four-room/fusion-native-fill/station-4.jpg')
    (ROOT/'fusion-native-static-room.html').write_text(page,encoding='utf8')
    shutil.copy2(ROOT/'four-room/surface-fusion-v4/inspection-views.json',ROOT/'four-room/fusion-native-fill/inspection-views.json')
    if args.serve_root:
        for name in ['fusion-native-room.html','native-anchor-policy.js','fusion-native-report.html','fusion-native-static-room.html']:shutil.copy2(ROOT/name,args.serve_root/name)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
