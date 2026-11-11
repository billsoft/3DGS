"""Run the full CPU pipeline from Windows: python fusion_next/pipeline.py."""
import argparse,subprocess,sys,time
from common import *
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-correspondences',action='store_true');args=parser.parse_args();times={}
    stages=[] if args.reuse_correspondences and (OUT/'correspondences.npz').exists() else ['correspondence.py']
    stages+=['appearance.py','geometry.py','ownership.py','seam_level.py','export.py','diagnostics.py']
    for script in stages:
        start=time.monotonic();subprocess.run([sys.executable,str(ROOT/'fusion_next'/script)],cwd=ROOT,check=True);times[script]=time.monotonic()-start
    subprocess.run([sys.executable,str(ROOT/'surface_fusion_validate.py'),'--out',str(OUT),'--baseline',str(BASE/'fused.ply')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'export_surface_bundle.py'),'--out',str(OUT)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'fusion_next/publish.py')],cwd=ROOT,check=True);write_json('run-times.json',times)
if __name__=='__main__':main()
