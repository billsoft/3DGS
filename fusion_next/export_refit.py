"""Export an evaluated optimized array to the existing standalone PLY/player."""
from common import *
import argparse, shutil


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--initial',type=Path,default=ROOT/'four-room/refit-data/initial.npy')
    parser.add_argument('--allow-opacity-change',action='store_true')
    args=parser.parse_args()
    a=np.load(args.candidate)
    initial=np.load(args.initial,mmap_mode='r')
    if a.shape!=initial.shape or not np.isfinite(a).all():
        raise ValueError('Invalid optimized model shape or values')
    if not args.allow_opacity_change and not np.array_equal(a[:,6],initial[:,6]):
        raise ValueError('Unexpected opacity change')
    rm.f.write(OUT/'fused.ply',a);rm.f.to_splat(OUT/'fused.splat',a)
    restored=rm.f.prep.read_gs(OUT/'fused.ply')
    if not np.array_equal(a,restored):raise ValueError('PLY roundtrip changed values')
    write_json('fusion.json',dict(centers=C.tolist(),up=rm.meta['up'],stations=stations,
        splats=len(a),method='bounded multi-view photometric Gaussian refit',
        view_dependent=False,semantic_rules=False,
        note='Single static Gaussian PLY. Four observed cameras do not determine unseen geometry.'))
    shutil.copyfile(ROOT/'four-room/surface-fusion-v4/inspection-views.json',OUT/'inspection-views.json')
    write_json('export-check.json',dict(points=len(a),finite=True,ply_roundtrip_exact=True,
        opacity_unchanged=bool(np.array_equal(a[:,6],initial[:,6])),max_position_change=float(np.linalg.norm(a[:,:3]-initial[:,:3],axis=1).max())))
    print('Exported',len(a),'Gaussians',flush=True)


if __name__=='__main__':run()
