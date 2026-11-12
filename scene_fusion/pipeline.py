"""Native Windows scene runner: validated inputs, checkpoints, QA and publication."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
from core import ROOT,initialize
from run_state import RunState,digest

def ps_quote(value):return "'"+str(value).replace("'","''")+"'"

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path)
    p.add_argument('--gpu-python',type=Path,required=True);p.add_argument('--serve-root',type=Path)
    p.add_argument('--stations',nargs='+');p.add_argument('--cache-root',type=Path)
    p.add_argument('--steps',type=int,default=2800);p.add_argument('--hd-steps',type=int,default=1400)
    p.add_argument('--depth-warp',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--check',action='store_true',help='Validate inputs only, without GPU processing')
    a=p.parse_args();a.inputs=a.inputs.resolve();a.gpu_python=a.gpu_python.resolve()
    if a.out is None:
        base=a.serve_root.resolve() if a.serve_root else ROOT
        suffix=hashlib.sha256(str(a.inputs).encode()).hexdigest()[:6]
        a.out=base/'four-room/scenes'/('scene-'+time.strftime('%Y%m%d-%H%M%S')+'-'+suffix)
    a.out=a.out.resolve();a.cache_root=(a.cache_root or a.out.parent/'.source-cache-v2').resolve()
    if a.out==a.inputs or a.inputs in a.out.parents:raise ValueError('Output must be outside the original input directory')
    if a.steps<1 or a.hd_steps<1:raise ValueError('Training steps must be positive')
    shell=shutil.which('pwsh') or shutil.which('powershell')
    if not shell or not a.gpu_python.is_file():raise ValueError('PowerShell or GPU Python environment is missing')
    if a.serve_root and a.out!=(a.serve_root.resolve()/'four-room/scenes'/a.out.name).resolve():raise ValueError('Published output must be under serve-root/four-room/scenes')
    scene=initialize(a.inputs,a.out,a.stations)
    code=list((ROOT/'scene_fusion').glob('*.py'))+list((ROOT/'fusion_next').glob('refit*.py'))+[ROOT/f for f in ['four-prepare.py','four-fuse.py','visibility-test.py','calibration-render.py','scene-room.html','scene-report.html','room-navigation.js','run-refit-gpu.ps1','fusion_next/gaussian_geometry.py']]
    code_hash=hashlib.sha256(''.join(digest(f) for f in sorted(code)).encode()).hexdigest()
    state=RunState(a.out,code_hash,a.resume);logs=a.out/'logs';logs.mkdir(exist_ok=True)
    def execute(name,script,*args,outputs,gpu=False):
        argv=[str(ROOT/script),*[str(x) for x in args]]
        if gpu:
            command='& '+ps_quote(ROOT/'run-refit-gpu.ps1')+' -Python '+ps_quote(a.gpu_python)+' -PythonArguments @('+','.join(map(ps_quote,argv))+')'
            cmd=[shell,'-NoProfile','-Command',command]
        else:cmd=[sys.executable,*argv]
        def work():
            with (logs/(name+'.log')).open('w',encoding='utf8') as log:
                result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                tail='\n'.join((logs/(name+'.log')).read_text(encoding='utf8',errors='replace').splitlines()[-12:])
                raise RuntimeError(f'{name} failed; see {logs/(name+".log")}\n{tail}')
        state.stage(name,argv,[a.out/x for x in outputs],work)
    selection=['--stations',*a.stations] if a.stations else []
    execute('preflight','scene_fusion/preflight.py','--inputs',a.inputs,'--out',a.out,*selection,outputs=['preflight.json'])
    if a.check:state.complete();print('INPUT CHECK PASSED',a.out,flush=True);return
    source_files=[f"{s['id']}/{f}" for s in scene['stations'] for f in ['complete.json','original.npy','visible.npy','cache.npz','camera.json',*[f'depth-{k}.npz' for k in range(6)]]]
    execute('prepare','scene_fusion/prepare.py','--inputs',a.inputs,'--out',a.out,'--cache-root',a.cache_root,*selection,outputs=source_files,gpu=True)
    execute('registration','scene_fusion/registration.py','--out',a.out,outputs=['registration.json'])
    execute('bundle','scene_fusion/bundle.py','--out',a.out,outputs=['photo-bundle/bundle.npz','photo-bundle/tracks.json','photo-bundle/report.json'])
    execute('scales','scene_fusion/align_scales.py','--out',a.out,outputs=['aligned-cameras.npz','scales.json',*[s['id']+'/depth-anchors.npz' for s in scene['stations']]])
    execute('domains','scene_fusion/ownership.py','--out',a.out,'--snapshot','rigid-initial',outputs=['rigid-initial.npy','rigid-initial-meta.json',*[f"aligned/{s['id']}.npy" for s in scene['stations']]])
    if a.depth_warp:
        execute('warp','scene_fusion/warp.py','--out',a.out,outputs=['warp-report.json',*[f"warped/{s['id']}.npy" for s in scene['stations']]])
        execute('warp-depth','scene_fusion/render_warped.py','--out',a.out,outputs=[f"warped-depth/{s['id']}/depth-{k}.npz" for s in scene['stations'] for k in range(6)],gpu=True)
        execute('warp-domains','scene_fusion/ownership.py','--out',a.out,'--warped','--snapshot','warp-initial',outputs=['warp-initial.npy','warp-initial-meta.json'])
    execute('activate','scene_fusion/activate.py','--out',a.out,'--snapshot','warp-initial' if a.depth_warp else 'rigid-initial',outputs=['initial.npy','initial.ply','initial.splat'])
    for size in [512,1024]:
        execute(f'dataset-{size}','scene_fusion/dataset.py','--out',a.out,'--size',size,outputs=[f'data-{size}/{f}' for f in ['initial.npy','targets.npy','dataset.json']])
    execute('neighbors','fusion_next/refit_neighbors.py','--data',a.out/'data-512',outputs=['data-512/neighbors.npz'])
    shutil.copyfile(a.out/'data-512/neighbors.npz',a.out/'data-1024/neighbors.npz')
    train=['--graph-weight','.8','--edge-weight','.3','--move-limit','.08','--shape-free','--opacity-min','-12','--opacity-max','24']
    execute('fit-low','fusion_next/refit_train.py','--data',a.out/'data-512','--out',a.out/'fit-low','--steps',a.steps,'--warmup',min(250,a.steps//4),
        '--color-lr','.003','--position-lr','.00016','--scale-lr','.0005',*train,outputs=['fit-low/optimized.npy','fit-low/training.json'],gpu=True)
    execute('fit-hd','fusion_next/refit_train.py','--data',a.out/'data-1024','--out',a.out/'fit-hd','--resume',a.out/'fit-low/optimized.npy',
        '--steps',a.hd_steps,'--warmup',0,'--color-lr','.001','--position-lr','.00004','--scale-lr','.0002',*train,outputs=['fit-hd/optimized.npy','fit-hd/training.json'],gpu=True)
    candidate=a.out/'fit-hd/optimized.npy'
    for size in [512,1024]:
        execute(f'eval-{size}','fusion_next/refit_evaluate.py','--data',a.out/f'data-{size}','--candidate',candidate,'--out',a.out/f'fit-hd/eval-{size}',
            '--baseline-label','initial fusion',outputs=[f'fit-hd/eval-{size}/evaluation.json'],gpu=True)
        execute(f'native-{size}','scene_fusion/native_audit.py','--out',a.out,'--candidate',candidate,'--report',a.out/f'fit-hd/native-{size}','--size',size,
            outputs=[f'fit-hd/native-{size}/audit.json'],gpu=True)
    execute('quality','scene_fusion/quality.py','--out',a.out,'--candidate',candidate,outputs=['quality.json'])
    execute('export','scene_fusion/export.py','--out',a.out,'--candidate',candidate,outputs=['final/fused.ply','final/fused.splat','final/export-check.json','fusion.json'])
    if a.serve_root:
        execute('publish','scene_fusion/publish.py','--out',a.out,'--evaluation','fit-hd/eval-1024','--native','fit-hd/native-1024','--serve-root',a.serve_root.resolve(),outputs=['result.json'])
    state.complete();print('COMPLETE',a.out,flush=True)
    if a.serve_root:print('PREVIEW http://127.0.0.1:7870/scene-room.html?scene='+a.out.name,flush=True)

if __name__=='__main__':main()
