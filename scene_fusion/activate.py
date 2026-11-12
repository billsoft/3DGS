"""Activate a chosen immutable initialization snapshot before downstream stages."""
import argparse,shutil,json
from pathlib import Path
import numpy as np
from core import gs,write_json
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--snapshot',required=True);a=p.parse_args()
source=a.out/(a.snapshot+'.npy');shutil.copyfile(source,a.out/'initial.npy');data=np.load(source)
gs.write(a.out/'initial.ply',data);gs.to_splat(a.out/'initial.splat',data)
write_json(a.out/'fusion.json',json.loads((a.out/(a.snapshot+'-meta.json')).read_text(encoding='utf8')))
