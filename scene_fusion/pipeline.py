"""Native Windows entry point for variable-count panorama + Gaussian PLY scenes."""
import argparse,json,shutil,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def ps_quote(value):return "'"+str(value).replace("'","''")+"'"

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpu-python',type=Path,required=True);p.add_argument('--serve-root',type=Path)
    p.add_argument('--steps',type=int,default=2800);p.add_argument('--hd-steps',type=int,default=1400)
    p.add_argument('--depth-warp',action='store_true',help='Experimental photo-track depth field; default uses only global similarity alignment')
    p.add_argument('--reuse-calibration',action='store_true',help='Reuses only byte-identical input calibration; later stages rerun')
    a=p.parse_args();a.inputs=a.inputs.resolve();a.out=a.out.resolve();a.gpu_python=a.gpu_python.resolve()
    shell=shutil.which('pwsh') or shutil.which('powershell');timings={}
    def execute(script,*args,gpu=False):
        started=time.monotonic();argv=[str(ROOT/script),*[str(x) for x in args]]
        if gpu:
            command='& '+ps_quote(ROOT/'run-refit-gpu.ps1')+' -Python '+ps_quote(a.gpu_python)+' -PythonArguments @('+','.join(map(ps_quote,argv))+')'
            subprocess.run([shell,'-NoProfile','-Command',command],cwd=ROOT,check=True)
        else:subprocess.run([sys.executable,*argv],cwd=ROOT,check=True)
        timings[script]=timings.get(script,0)+time.monotonic()-started
    # The manifest hash is checked even when prepared source caches exist.
    execute('scene_fusion/prepare.py','--inputs',a.inputs,'--out',a.out,gpu=True)
    for stage,artifact in [('registration.py','registration.json'),('bundle.py','photo-bundle/bundle.npz')]:
        if not(a.reuse_calibration and (a.out/artifact).exists()):execute('scene_fusion/'+stage,'--out',a.out)
    execute('scene_fusion/align_scales.py','--out',a.out)
    execute('scene_fusion/ownership.py','--out',a.out)
    if a.depth_warp:
        execute('scene_fusion/warp.py','--out',a.out)
        execute('scene_fusion/render_warped.py','--out',a.out,gpu=True)
        execute('scene_fusion/ownership.py','--out',a.out,'--warped')
    for size in [512,1024]:execute('scene_fusion/dataset.py','--out',a.out,'--size',size)
    execute('fusion_next/refit_neighbors.py','--data',a.out/'data-512')
    shutil.copyfile(a.out/'data-512/neighbors.npz',a.out/'data-1024/neighbors.npz')
    train=['--graph-weight','.8','--edge-weight','.3','--move-limit','.08','--shape-free','--opacity-min','-12','--opacity-max','24']
    execute('fusion_next/refit_train.py','--data',a.out/'data-512','--out',a.out/'fit-low','--steps',a.steps,'--warmup',250,
        '--color-lr','.003','--position-lr','.00016','--scale-lr','.0005',*train,gpu=True)
    execute('fusion_next/refit_train.py','--data',a.out/'data-1024','--out',a.out/'fit-hd','--resume',a.out/'fit-low/optimized.npy',
        '--steps',a.hd_steps,'--warmup',0,'--color-lr','.001','--position-lr','.00004','--scale-lr','.0002',*train,gpu=True)
    candidate=a.out/'fit-hd/optimized.npy'
    for size in [512,1024]:
        execute('fusion_next/refit_evaluate.py','--data',a.out/f'data-{size}','--candidate',candidate,'--out',a.out/f'fit-hd/eval-{size}',gpu=True)
        execute('scene_fusion/native_audit.py','--out',a.out,'--candidate',candidate,'--report',a.out/f'fit-hd/native-{size}','--size',size,gpu=True)
    execute('scene_fusion/export.py','--out',a.out,'--candidate',candidate)
    if a.serve_root:
        execute('scene_fusion/publish.py','--out',a.out,'--evaluation','fit-hd/eval-1024','--native','fit-hd/native-1024','--serve-root',a.serve_root.resolve())
    (a.out/'run-times.json').write_text(json.dumps(timings,indent=2))
    print('Experimental result exported. Inspect report and source/midpoint views before accepting it.',flush=True)

if __name__=='__main__':main()
