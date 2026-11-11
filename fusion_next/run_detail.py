"""Separate experiment; never overwrite the accepted baseline output."""
from pathlib import Path
import os,sys,shutil,subprocess,argparse,json
ROOT=Path(__file__).resolve().parent.parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smooth',action='store_true');parser.add_argument('--output',default='fusion-detail');parser.add_argument('--validate',action='store_true');parser.add_argument('--serve-root',type=Path);args=parser.parse_args()
    baseline=ROOT/'four-room/baselines/2026-11-11-fusion-next'
    if Path(args.output).name!=args.output:raise ValueError('Output must be a folder name')
    if not args.output.startswith('fusion-detail'):raise ValueError('Use a fusion-detail experiment folder; baseline outputs are protected')
    output=ROOT/'four-room'/args.output
    if output.resolve()==baseline.resolve():raise ValueError('Cannot overwrite the accepted baseline')
    output.mkdir(parents=True,exist_ok=True)
    names=['geometry.npz','ownership.npz','prior.npz','correspondences.npz','seam-offset.npy','appearance-report.json','geometry-report.json','ownership-report.json','flow-report.json','seam-level-report.json']
    for name in names:shutil.copy2(baseline/name,output/name)
    if not args.smooth:
        (output/'smooth-report.json').write_text(json.dumps(dict(enabled=False,accepted=False,frozen_unstable_vertices=0,max_candidate_displacement=0,invalid_faces=0,held_median_before=0,held_median_after=0,held_p90_before=0,held_p90_after=0,note='Default preserves accepted geometry exactly. Optional conservative smoothing did not provide a material visual gain in the ablation.'),indent=2),encoding='utf8')
    env=os.environ.copy();env['GS_OUTPUT_DIR']=str(output);env['GS_DETAIL']='1';env['OPENBLAS_NUM_THREADS']='1'
    if args.validate:env.pop('GS_QUICK_EVAL',None)
    scripts=['smooth_surface.py','ownership.py','seam_level.py'] if args.smooth else []
    scripts+=['export.py']
    for script in scripts:subprocess.run([sys.executable,str(ROOT/'fusion_next'/script)],cwd=ROOT,env=env,check=True)
    if args.validate:
        commands=[['fusion_next/validate_detail.py'],['surface_fusion_validate.py','--out',str(output),'--baseline',str(baseline/'fused.ply')],['export_surface_bundle.py','--out',str(output)]]
        for command in commands:subprocess.run([sys.executable]+command,cwd=ROOT,env=env,check=True)
    if args.serve_root:
        if args.output!='fusion-detail' or not args.validate:raise ValueError('Publishing requires the validated fusion-detail output')
        subprocess.run([sys.executable,str(ROOT/'fusion_next/publish_detail.py'),'--serve-root',str(args.serve_root)],cwd=ROOT,env=env,check=True)
    print('Detail experiment exported:',output,flush=True)

if __name__=='__main__':main()
