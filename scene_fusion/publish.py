import argparse,json,shutil
from pathlib import Path
from core import ROOT,load,write_json

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--evaluation',required=True);p.add_argument('--native',required=True);p.add_argument('--serve-root',type=Path,required=True)
args=p.parse_args();scene=load(args.out);bundle=json.loads((args.out/'photo-bundle/report.json').read_text(encoding='utf8'));evaluation=json.loads((args.out/args.evaluation/'evaluation.json').read_text(encoding='utf8'));native=json.loads((args.out/args.native/'audit.json').read_text(encoding='utf8'));reg=json.loads((args.out/'registration.json').read_text(encoding='utf8'))
if args.out.resolve()!=(args.serve_root/'four-room/scenes'/args.out.name).resolve():
    raise ValueError('Publish output must be under serve-root/four-room/scenes/<scene-id> (or an existing junction to it)')
views=evaluation['views'];mid=[v['view'] for v in views if v['role']=='novel'];ids=[s['id'] for s in scene['stations']]
comparisons=[f'source-{ids[0]}-3',f'source-{ids[0]}-4',f'source-{ids[len(ids)//2]}-5',f'source-{ids[-1]}-2']
report=dict(summary=f"{len(ids)} 张全景图与 {len(ids)} 份 PLY 从原始输入重新标定、配准、分区和联合拟合；单个静态融合模型，使用独立场景缓存。",
    limitations='仍是实验结果。部分近物体边界、细杆、反射和未观测区域可能有重影或缺损；照片参与拟合，来源 SSIM 不等于新视点准确率。当前场景数量可变，但不同房间仍需通过标定和匹配证据检查。',
    metrics={'来源数量':len(ids),'成功初始化匹配边':sum(e['accepted'] for e in reg['edges'].values()),'多视图轨迹':bundle['tracks'],'留出轨迹':bundle['held_tracks'],
      '留出中位等效像素_前':bundle['before_held']['median_equivalent_px'],'留出中位等效像素_后':bundle['after_held']['median_equivalent_px'],
      '优化前后同掩码拟合指标':evaluation['summary'],'原始PLY固定掩码检查':native['summary']},
    native_report=args.native,evaluation_report=args.evaluation,comparisons=comparisons,midpoints=mid[:6])
write_json(args.out/'result.json',report)
if (args.out/'quality.json').exists():
    report['metrics']['自动质量检查']=json.loads((args.out/'quality.json').read_text(encoding='utf8'))
    write_json(args.out/'result.json',report)
for name in ['scene-room.html','scene-report.html','room-navigation.js','room-sorting.js','room-look.js']:shutil.copyfile(ROOT/name,args.serve_root/name)
print('Published scene',args.out.name,flush=True)
