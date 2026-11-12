"""Export one generic scene result, without importing historical room metadata."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from core import load,gs,prep,write_json

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
args=p.parse_args();scene=load(args.out);a=np.load(args.candidate)
if a.ndim!=2 or a.shape[1]!=14 or not np.isfinite(a).all():raise ValueError('Invalid Gaussian array')
if np.any(np.linalg.norm(a[:,10:14],axis=1)<.5):raise ValueError('Invalid rotations')
folder=args.out/'final';folder.mkdir(exist_ok=True);np.save(folder/'optimized.npy',a)
gs.write(folder/'fused.ply',a);gs.to_splat(folder/'fused.splat',a)
if not np.array_equal(prep.read_gs(folder/'fused.ply'),a):raise ValueError('Export roundtrip changed values')
meta=json.loads((args.out/'fusion.json').read_text(encoding='utf8'));meta.update(optimized=True,model_folder='final',splats=len(a),experimental=True)
write_json(args.out/'fusion.json',meta)
write_json(folder/'export-check.json',dict(points=len(a),finite=True,ply_roundtrip_exact=True,
    candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),ply_sha256=hashlib.sha256((folder/'fused.ply').read_bytes()).hexdigest()))
print('Exported',len(a),flush=True)
