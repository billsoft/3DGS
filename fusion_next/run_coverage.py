"""Generate proposals, rollback photometric regressions, validate and publish."""
from pathlib import Path
import os,sys,subprocess,argparse
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--serve-root',type=Path);args=parser.parse_args()
    env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env.pop('GS_QUICK_EVAL',None)
    def call(script,*arguments):
        subprocess.run([sys.executable,script,*map(str,arguments)],cwd=ROOT,env=env,check=True)
    env['GS_OUTPUT_DIR']=str(ROOT/'four-room/fusion-coverage-wide')
    call('fusion_next/coverage_safe.py','--minimum-gain','.75','--evidence-distance','.2')
    env['GS_OUTPUT_DIR']=str(ROOT/'four-room/fusion-coverage-guarded')
    call('fusion_next/photometric_guard.py')
    call('fusion_next/validate_detail.py')
    call('fusion_next/audit_coverage.py')
    call('surface_fusion_validate.py','--out',env['GS_OUTPUT_DIR'],'--baseline',ROOT/'four-room/baselines/2026-11-11-fusion-next/fused.ply')
    if args.serve_root:call('fusion_next/publish_coverage.py','--serve-root',args.serve_root)

if __name__=='__main__':main()
