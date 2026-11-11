"""Reproduce native-camera diagnostics, static ablations and preview artifacts."""
from pathlib import Path
import os,sys,shutil,subprocess,argparse
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--serve-root',type=Path);args=parser.parse_args()
    env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1'
    def call(script,*arguments):subprocess.run([sys.executable,script,*map(str,arguments)],cwd=ROOT,env=env,check=True)
    def output(name):env['GS_OUTPUT_DIR']=str(ROOT/'four-room'/name)
    output('fusion-native-fidelity');call('fusion_next/native_fidelity.py');call('fusion_next/validate_native_alignment.py')
    output('fusion-native-visible');call('fusion_next/native_visible.py');call('fusion_next/validate_native_candidate.py')
    output('fusion-native-previous');Path(env['GS_OUTPUT_DIR']).mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'four-room/fusion-territory/raw-core-fused.ply',Path(env['GS_OUTPUT_DIR'])/'fused.ply');call('fusion_next/validate_native_candidate.py')
    output('fusion-native-fill');call('fusion_next/native_coverage_fill.py');call('fusion_next/validate_native_candidate.py')
    call('surface_fusion_validate.py','--out',env['GS_OUTPUT_DIR'],'--baseline',ROOT/'four-room/baselines/2026-11-11-coverage-guard/fused.ply')
    if args.serve_root:call('fusion_next/publish_native.py','--serve-root',args.serve_root)

if __name__=='__main__':main()
