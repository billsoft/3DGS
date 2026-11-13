"""Versioned panorama + Gaussian PLY -> single GS -> smooth guide -> photo mesh.

All artifacts belong to one run. Existing scene recipes and accepted baselines
are never adopted implicitly. Training fidelity is reported separately from job
completion. The smoke profile tests plumbing, not delivered scene quality.
"""
import argparse,hashlib,html,json,re,shutil,subprocess,sys,time
from pathlib import Path
from core import ROOT,initialize,scene_id,write_json
from run_state import RunState,digest
from pipeline import ps_quote

PRESETS={'adaptive':dict(depth=9,iterations=3,relative_voxel=.004),
         'smooth':dict(depth=8,iterations=12,relative_voxel=.004)}

def make_plan(root,ids,profile='standard',mode='both',conflict_regions=True):
    r=Path(root);c=r/'calibration';native=r/'native';base=r/'distill';single=r/'single'
    raw=r/'depth-raw';depth=r/'depth-smooth';tsdf=r/'tsdf';repair=r/'repair';guide=r/'guide'
    smoke=profile=='smoke';size=128 if smoke else 512;steps=12 if smoke else 1800
    result=[]
    def add(name,script,args,outputs,env='cpu',fresh=()):
        result.append(dict(name=name,script='scene_fusion/'+script+'.py',args=list(map(str,args)),
                           outputs=[str(x) for x in outputs],environment=env,fresh=[str(x) for x in fresh]))
    add('prepare','prepare',['--inputs','@INPUTS@','--out',c,'--cache-root',r.parent/'.source-cache-v2'],
        [c/i/f for i in ids for f in ['complete.json','original.npy','visible.npy','normalization.json','cache.npz','camera.json',*[f'depth-{k}.npz' for k in range(6)]]],'gpu')
    add('registration','registration',['--out',c],[c/'registration.json',*[c/i/'registration-features.npz' for i in ids]])
    add('bundle','bundle',['--out',c],[c/'photo-bundle'])
    add('scales','align_scales',['--out',c],[c/'aligned-cameras.npz',c/'scales.json',*[c/i/'depth-anchors.npz' for i in ids]])
    add('camera-graph','dataset',['--out',c,'--metadata-only','--evidence-graph'],[c/'data-512/dataset.json',c/'camera-graph.json'])
    add('native','native_package',['--scene',c,'--out',native],[native],fresh=[native])
    add('teacher','tour_distill',['prepare','--native',native,'--out',base,'--size',size,'--core','.49','--live-targets'],[base/'data'],'gpu',[base/'data'])
    add('distill','tour_distill',['train','--out',base,'--steps',steps],[base/'optimized.npz',base/'training.json'],'gpu')
    add('detail-views','tour_detail_views',['--dataset',base/'data/dataset.json','--out',r/'detail-views.json','--transitions'],[r/'detail-views.json'])
    add('refine','tour_refine',['--reference',base,'--out',single,'--views',r/'detail-views.json','--sizes',*([128] if smoke else [640,1024,1536]),
        '--steps',12 if smoke else 2400,'--core','.49','--lr-scale','.35','--final-lr-ratio','.1','--patch-weight','.15',
        '--max-points',2200000,'--split-count',300000,'--split-steps',500,1300],
        [single/'optimized.npz',single/'training.json',single/'data'],'gpu',[single])
    add('evaluate','tour_distill',['evaluate','--out',single,'--eval-size',128 if smoke else 512,'--local-checks'],[single/'evaluation'],'gpu',[single/'evaluation'])
    add('export','tour_distill',['export','--out',single,'--native',native],[single/'final',single/'fusion.json',single/'data-512/dataset.json'])
    add('capture','single_gs_mesh',['capture','--source',c,'--model',single/'optimized.npz','--out',raw,'--size',128 if smoke else 768],[raw],'gpu',[raw])
    add('depth-smooth','depth_geometry_smooth',['--source',raw,'--out',depth,'--profile','strong'],[depth],fresh=[depth])
    add('tsdf','single_gs_mesh',['integrate','--source',depth,'--out',tsdf,'--resolution',96 if smoke else 512,'--max-faces',60000 if smoke else 1200000],[tsdf],'mesh',[tsdf])
    add('repair','mesh_surface_repair',['--mesh',tsdf/'surface.npz','--cameras',depth,'--out',repair],[repair],'mesh',[repair])
    add('guide','poisson_surface',['--mesh',repair/'surface.npz','--cameras',depth,'--out',guide,'--depth',6 if smoke else 9,
        '--samples',15000 if smoke else 500000,'--max-faces',60000 if smoke else 1200000,'--point-smooth-iterations',4],[guide],'mesh',[guide])
    modes=list(PRESETS) if mode=='both' else [mode]
    for name in modes:
        opt=PRESETS[name];coarse=r/(name+'-coarse');geom=r/(name+'-geometry');photo=r/name
        extra=['--edge-source',depth,'--reference-mesh',guide/'surface.npz'] if name=='adaptive' else []
        add(name+'-lowpass','coarse_surface',['--guide',guide/'geometry-guide.ply','--cameras',depth,'--out',coarse,
            '--relative-voxel',opt['relative_voxel'],'--depth',6 if smoke else opt['depth'],'--iterations',opt['iterations'],*extra],[coarse],'mesh',[coarse])
        add(name+'-weld','mesh_numeric_weld',['--mesh',coarse/'surface.npz','--cameras',depth,'--out',geom],[geom],'mesh',[geom])
        add(name+'-texture','textured_mesh',['--source',depth,'--geometry',geom/'surface.npz','--out',photo,'--warped','--footprint',
            '--texture-width',1024 if smoke else 7680,'--smoothing',0,'--visibility-cache',geom/'visibility.npz','--coherent-patches',
            *(['--conflict-regions'] if conflict_regions else [])],[photo],'cpu',[photo])
    add('inspection-views','oblique_views',['--source',depth,'--out',r/'inspection-views.json'],[r/'inspection-views.json'])
    add('mesh-audit','mesh_stage_audit',['--cameras',depth,'--mesh',*[r/n/'surface-owned.npz' for n in modes],
        '--out',r/'audit','--views',r/'inspection-views.json','--size',96 if smoke else 384],[r/'audit'],'mesh',[r/'audit'])
    return result

def finish(root,mode,profile):
    modes=list(PRESETS) if mode=='both' else [mode];rid=scene_id(root)
    ev=json.loads((root/'single/evaluation/evaluation.json').read_text('utf8'))
    links=[('原始 PLY 连续漫游','/native-tour-room.html?scene='+rid+'/native&range=0.49'),
           ('单一高斯 PLY 预览','/scene-room.html?scene='+rid+'/single'),
           ('下载单一高斯 PLY','/four-room/scenes/'+rid+'/single/final/fused.ply'),
           ('下载平滑几何引导 PLY（非高斯）','/four-room/scenes/'+rid+'/guide/geometry-guide.ply')]
    for n in modes:
        label='自适应平滑' if n=='adaptive' else '平滑优先'
        links += [(label+'预览','/mesh-photo-room.html?scene='+rid+'/'+n),
                  (label+' OBJ＋贴图','/four-room/scenes/'+rid+'/'+n+'/textured-room.zip')]
    write_json(root/'result.json',dict(version=1,profile=profile,stage_complete=True,visual_acceptance_required=True,
        anchor_fidelity_passed=ev['anchor_fidelity_passed'],geometry_guide_is_gaussian=False,links=links,
        graph=json.loads((root/'calibration/camera-graph.json').read_text('utf8')),
        mesh_audit='audit/audit.json',recipe='recipe.json'))
    items=''.join('<li><a href="'+html.escape(url,quote=True)+'">'+html.escape(label)+'</a></li>' for label,url in links)
    page='<!doctype html><meta charset="utf-8"><title>重建结果</title><style>body{font:18px/1.8 system-ui;margin:40px;background:#14202b;color:white}a{color:#9ddaff}</style>'
    page+='<h1>多全景与 PLY 重建结果</h1><p>'+('流程测试：低分辨率，不作为质量版本。' if profile=='smoke' else '独立重建结果，仍需检查原机位、交接与斜视画面。')+'</p><ul>'+items+'</ul>'
    page+='<p>原机位数值保真门槛：'+('通过' if ev['anchor_fidelity_passed'] else '未通过')+'。完成流程不等于恢复真实形状；未知区域、反射和细结构仍可能有缺陷。</p><a href="result.json">验证与流程记录</a>'
    (root/'index.html').write_text(page,encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--settings',type=Path,default=ROOT/'scene_fusion/settings.local.json');p.add_argument('--mode',choices=['both',*PRESETS],default='both')
    p.add_argument('--profile',choices=['standard','smoke'],default='standard');p.add_argument('--resume',action='store_true')
    p.add_argument('--legacy-texture-partition',action='store_true',help='Compare against the earlier texture partition without conflict-region consolidation')
    p.add_argument('--check',action='store_true');p.add_argument('--plan',action='store_true');a=p.parse_args()
    a.out=a.out.resolve();a.inputs=a.inputs.resolve()
    if a.out==a.inputs or a.inputs in a.out.parents or a.out in a.inputs.parents:raise ValueError('Output must be separate from input')
    if not re.fullmatch('[A-Za-z0-9_-]+',a.out.name):raise ValueError('Output run name must use letters, digits, hyphens or underscores')
    settings=json.loads(a.settings.read_text('utf8'));envs={k:Path(settings[k+'_python']) for k in ('cpu','gpu','mesh')}
    if not all(v.is_file() for v in envs.values()):raise ValueError('Configure CPU, GPU and Open3D mesh Python environments')
    scene=initialize(a.inputs,a.out/'calibration');ids=[s['id'] for s in scene['stations']]
    stages=make_plan(a.out,ids,a.profile,a.mode,not a.legacy_texture_partition)
    for stage in stages:stage['args']=[str(a.inputs) if x=='@INPUTS@' else x for x in stage['args']]
    recipe=dict(version=2,input_scene=scene,profile=a.profile,mode=a.mode,presets=PRESETS,
        texture_conflict_regions=not a.legacy_texture_partition,stages=stages)
    if a.plan:print(json.dumps(recipe,ensure_ascii=False,indent=2));return
    from preflight import run as preflight
    if a.check:preflight(a.inputs,a.out/'calibration');print('INPUT CHECK PASSED');return
    old=a.out/'recipe.json'
    if old.exists() and json.loads(old.read_text('utf8'))!=recipe:raise ValueError('Recipe changed; use a new run output')
    write_json(old,recipe)
    code=[*sorted((ROOT/'scene_fusion').glob('*.py')),*sorted((ROOT/'fusion_next').glob('*.py')),*[ROOT/n for n in ('four-prepare.py','four-fuse.py','run-refit-gpu.ps1','calibration-render.py','visibility-test.py')]]
    fingerprint=hashlib.sha256(''.join(digest(f) for f in code).encode()).hexdigest()
    state=RunState(a.out,fingerprint,a.resume);logs=a.out/'logs';logs.mkdir(exist_ok=True)
    state.stage('preflight',[digest(old)],[a.out/'calibration/preflight.json'],lambda:preflight(a.inputs,a.out/'calibration'))
    shell=shutil.which('pwsh') or shutil.which('powershell')
    for stage in stages:
        argv=[str(ROOT/stage['script']),*stage['args']];env=stage['environment']
        command=[str(envs[env]),*argv]
        if env=='gpu':
            command=[shell,'-NoProfile','-Command','& '+ps_quote(ROOT/'run-refit-gpu.ps1')+' -Python '+ps_quote(envs['gpu'])+' -PythonArguments @('+','.join(map(ps_quote,argv))+')']
        def work(stage=stage,command=command):
            # Preserve failed/outdated attempts rather than overwrite candidate data.
            for folder in map(Path,stage['fresh']):
                if folder.exists():
                    if a.out not in folder.resolve().parents:raise ValueError('Stage output escapes run directory')
                    backup=a.out/'attempts'/str(time.time_ns());backup.mkdir(parents=True)
                    folder.rename(backup/folder.name)
            with (logs/(stage['name']+'.log')).open('w',encoding='utf8') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                tail='\n'.join((logs/(stage['name']+'.log')).read_text('utf8',errors='replace').splitlines()[-12:])
                raise RuntimeError(stage['name']+' failed: '+tail)
        state.stage(stage['name'],[str(envs[env]),*argv],stage['outputs'],work)
    state.stage('delivery',[a.mode,a.profile],[a.out/'result.json',a.out/'index.html'],lambda:finish(a.out,a.mode,a.profile))
    serve=Path(settings['serve_root']).resolve() if settings.get('serve_root') else None
    if serve and (serve/'four-room/scenes').resolve() in a.out.parents:
        for name in ['mesh-photo-room.html','mesh-photo-room.js','scene-room.html','tour-distill-report.html','native-tour-room.html','native-tour.js']:
            if (ROOT/name).resolve()!=(serve/name).resolve():shutil.copyfile(ROOT/name,serve/name)
        print('RESULT http://127.0.0.1:7870/four-room/scenes/'+scene_id(a.out)+'/index.html',flush=True)
    state.complete()

if __name__=='__main__':main()
