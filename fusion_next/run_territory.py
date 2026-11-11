"""Camera territories and cropped original PLY experiment, isolated from baseline."""
from pathlib import Path
import os,sys,subprocess,shutil,argparse
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--validate',action='store_true')
    parser.add_argument('--serve-root',type=Path)
    args=parser.parse_args()
    if args.serve_root and not args.validate:parser.error('Publishing requires --validate')
    baseline=ROOT/'four-room/baselines/2026-11-11-fusion-next'
    output=ROOT/'four-room/fusion-territory';output.mkdir(parents=True,exist_ok=True)
    for name in ['geometry.npz','ownership.npz','prior.npz','correspondences.npz','seam-offset.npy','appearance-report.json','geometry-report.json','ownership-report.json','flow-report.json','seam-level-report.json']:
        shutil.copy2(baseline/name,output/name)
    env=os.environ.copy();env.update(GS_OUTPUT_DIR=str(output),GS_TERRITORY='1',GS_DETAIL='1',OPENBLAS_NUM_THREADS='1')
    for script in ['ownership.py','crop_sources.py','seam_level.py','anchor_cropped.py','export.py','fill_seams.py']:
        subprocess.run([sys.executable,str(ROOT/'fusion_next'/script)],cwd=ROOT,env=env,check=True)
    if args.validate:
        env.pop('GS_QUICK_EVAL',None)
        commands=[['fusion_next/validate_detail.py'],['surface_fusion_validate.py','--out',str(output),'--baseline',str(baseline/'fused.ply')],['export_surface_bundle.py','--out',str(output)]]
        for command in commands:subprocess.run([sys.executable]+command,cwd=ROOT,env=env,check=True)
    if args.serve_root:
        subprocess.run([sys.executable,'fusion_next/publish_territory.py','--serve-root',str(args.serve_root)],cwd=ROOT,env=env,check=True)
    print('Territory experiment exported',flush=True)

if __name__=='__main__':main()
