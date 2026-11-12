"""Publish a reviewable static PLY candidate without replacing the accepted URL."""
from pathlib import Path
import argparse,json,shutil,html

ROOT=Path(__file__).resolve().parents[1]


def run():
    parser=argparse.ArgumentParser()
    parser.add_argument('--result',required=True)
    parser.add_argument('--serve-root',type=Path)
    parser.add_argument('--verdict',default='候选版，仍需检查剩余缺陷')
    args=parser.parse_args()
    folder=ROOT/'four-room'/args.result
    evaluation=json.loads((folder/'eval/evaluation.json').read_text())
    training=json.loads((folder/'training.json').read_text())
    prefix='/four-room/'+args.result
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>会议室 · 多视图联合优化</title><style>body{margin:0;background:#111;color:white;font:14px system-ui}#tools{position:absolute;z-index:100;top:12px;left:12px;padding:14px;border-radius:10px;background:#18212eef;max-width:350px}h1{font-size:17px}button{margin:3px;padding:6px}a{color:#9cdeff;margin-right:10px}p{margin:8px 0}</style>
<div id="tools"><h1>会议室 · 多视图联合优化</h1><p><a href="?version=new">稳形候选</a><a href="?version=detail">清晰对照</a><a href="?version=baseline">已接受基线</a><a href="/refit-report.html">同视点对照</a></p><div id="positions"></div><p>W/S 前后 · A/D 左右平移 · Q 上升 / E 下降</p><p id="status" role="status">加载中…</p><p>单个静态融合模型。未知背面和部分细杆仍有缺损；没有使用灯具模板或照片覆盖层。</p><a id="download" download>下载融合 PLY</a></div>
<script type="importmap">{"imports":{"three":"/three.js"}}</script><script type="module">
import {installRoomNavigation} from './room-navigation.js';
import * as THREE from 'three';
import {Viewer,RenderMode,SceneRevealMode} from './gaussian.js';
const query=new URLSearchParams(location.search),previous=query.get('version')==='baseline';
const detail=query.get('version')==='detail';
const base=previous?'/four-room/baselines/2026-11-11-coverage-guard':detail?'/four-room/refit-coherent-hd':'__PREFIX__';
document.querySelector('#download').href=base+'/fused.ply';
try {
const meta=await (await fetch('__PREFIX__/fusion.json')).json();
const centers=meta.centers.map(p=>new THREE.Vector3(...p));
const viewer=new Viewer({cameraUp:meta.up,initialCameraPosition:centers[0].clone().lerp(centers[1],.5).toArray(),initialCameraLookAt:[4,0,-.6],sharedMemoryForWorkers:false,integerBasedSort:true,gpuAcceleratedSort:false,ignoreDevicePixelRatio:true,enableOptionalEffects:true,renderMode:RenderMode.OnChange,sceneRevealMode:SceneRevealMode.Instant});
await viewer.addSplatScenes([{path:base+'/fused.splat',splatAlphaRemovalThreshold:3}],true);viewer.start();
let movement=0;const panel=document.querySelector('#positions');
const button=(name,action)=>{const b=document.createElement('button');b.textContent=name;b.onclick=action;panel.appendChild(b)};
const go=point=>{const token=++movement,from=viewer.camera.position.clone(),start=performance.now();const tick=()=>{if(token!==movement)return;const t=Math.min(1,(performance.now()-start)/1100),e=t*t*t*(10-15*t+6*t*t);const next=from.clone().lerp(point,e),delta=next.clone().sub(viewer.camera.position);viewer.camera.position.copy(next);viewer.controls.target.add(delta);viewer.controls.update();viewer.forceRenderNextFrame();if(t<1)requestAnimationFrame(tick)};requestAnimationFrame(tick)};
centers.forEach((c,i)=>button(meta.stations[i]+'号机位',()=>go(c)));
panel.appendChild(document.createElement('br'));centers.forEach((c,i)=>button(meta.stations[i]+'–'+meta.stations[(i+1)%4]+'中点',()=>go(c.clone().lerp(centers[(i+1)%4],.5))));
const inspection=await (await fetch('__PREFIX__/inspection-views.json')).json();
const inspect=i=>{const v=inspection[i];movement++;viewer.camera.position.fromArray(v.position);viewer.controls.target.fromArray(v.target);viewer.controls.update();viewer.forceRenderNextFrame()};
inspection.forEach((v,i)=>button('检查 '+v.name,()=>inspect(i)));
if(query.has('inspect'))inspect(Number(query.get('inspect')));
installRoomNavigation({camera:viewer.camera,controls:viewer.controls,onMove:()=>{movement++},render:()=>viewer.forceRenderNextFrame()});
viewer.controls.addEventListener('start',()=>{movement++});window.roomViewer=viewer;window.inspectRoom=inspect;
document.querySelector('#status').textContent='已加载 · '+(previous?'已接受基线':detail?'清晰对照版':'稳形联合优化候选');
} catch(e){document.querySelector('#status').textContent='加载失败：'+e.message;console.error(e)}
</script></html>'''.replace('__PREFIX__',prefix)
    summary=evaluation['summary']
    table=''.join(f'<tr><td>{html.escape(r["view"])}</td><td>{r["baseline"]["ssim"]:.4f}</td><td>{r["candidate"]["ssim"]:.4f}</td><td>{r["new_holes"]}</td></tr>' for r in evaluation['views'] if 'baseline' in r)
    selections=['detail-0','detail-1','source-2-2','source-3-2','source-2-5','source-3-1','mid-0-room','mid-1-room','mid-2-ceiling','mid-3-table']
    pictures=''.join(f'<h2>{name}</h2><a href="{prefix}/eval/{name}.jpg"><img loading="lazy" src="{prefix}/eval/{name}.jpg"></a>' for name in selections)
    report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>联合优化实测对照</title><style>body{{max-width:1500px;margin:25px auto;padding:20px;background:#17212c;color:#edf4fa;font:16px/1.8 system-ui}}a{{color:#9cdeff}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #526477;padding:7px}}</style>
<h1>四张原片共同约束最终高斯</h1><p><strong>{html.escape(args.verdict)}</strong></p><p><a href="/refit-room.html">交互预览</a><a href="/refit-room.html?version=baseline">已接受基线</a><a href="{prefix}/fused.ply">下载单个 PLY</a></p>
<p>本轮直接优化已接受模型的颜色、位置与尺寸，保留点数和透明度。没有灯具专用分支、照片覆盖层或视点专用颜色。四张全景图拆成24个训练方向，但仍然只有4个真实拍摄位置。</p>
<p>先用表面邻接约束抑制逐点拟合产生的碎纹，再做高分辨率优化。稳形候选额外平滑位移场：同一表面上的邻近点协同移动，跨表面和法向突变处停止传播；原有纹理和物体表面本身不直接做均值平滑。随后固定位置，微调颜色与尺寸。</p>
<p><a href="/refit-room.html?version=detail">清晰对照版</a>在1024检查下平均 SSIM 为0.9414，但部分曲面近看更有颗粒感。默认稳形版牺牲一些拟合分数以减少不协调位移；两种结果均保留，旧基线未覆盖。</p>
<p>检查26个原机位方向和12个中间视向。相对原片的平均 SSIM：{summary['baseline_mean_ssim']:.4f} → {summary['candidate_mean_ssim']:.4f}；平均 RGB 误差：{summary['baseline_mean_mae']:.4f} → {summary['candidate_mean_mae']:.4f}。SSIM 提升方向 {summary['improved_ssim_views']}/26；固定基线覆盖区域内，新增 alpha&lt;0.5 像素合计 {summary['new_holes']}。</p>
<p>这些照片也用于优化，因此分数是原片拟合检查，不是独立测试集的准确率；中间视点没有真实照片可作真值。不能据此宣称未知背面已恢复。相同视点图片从左到右为原图、基线、候选；中点只有基线和候选。</p>
<p><a href="{prefix}/eval/evaluation.json">逐视点指标</a><a href="{prefix}/training.json">训练参数</a><a href="{prefix}/export-check.json">导出检查</a></p>
{pictures}<h2>逐视点 SSIM</h2><table><tr><th>视向</th><th>基线</th><th>候选</th><th>新增空洞像素</th></tr>{table}</table>
<p>参考实现：<a href="https://github.com/GAP-LAB-CUHK-SZ/GaussReg">GaussReg</a>、<a href="https://github.com/nmoehrle/mvs-texturing">MVS-Texturing</a>、<a href="https://github.com/hbb1/2d-gaussian-splatting">2DGS</a>、<a href="https://github.com/zju3dv/PGSR">PGSR</a>。可微渲染采用 <a href="https://github.com/nerfstudio-project/gsplat">gsplat</a>；本项目没有复制这些项目的受限训练代码，也没有实现完整 PGSR。</p></html>'''
    for name,text in [('refit-room.html',page),('refit-report.html',report)]:
        (ROOT/name).write_text(text,encoding='utf8')
        if args.serve_root:shutil.copy2(ROOT/name,args.serve_root/name)


if __name__=='__main__':run()
