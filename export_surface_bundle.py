import json,zipfile,argparse
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('four-room/surface-fusion-v2'));args=parser.parse_args()
O=args.out;meta=json.loads((O/'mesh.json').read_text());groups=meta['groups'];rows=np.fromfile(O/'mesh.bin','<f4').reshape(-1,5)
with (O/'room.obj').open('w') as file:
    file.write('mtllib room.mtl\n');np.savetxt(file,rows[:,:3],fmt='v %.6f %.6f %.6f');np.savetxt(file,np.c_[rows[:,3],1-rows[:,4]],fmt='vt %.7f %.7f')
    for i,g in enumerate(groups):
        file.write(f'usemtl photo{i}\n');ids=np.arange(g['start']+1,g['start']+g['count']+1).reshape(-1,3);np.savetxt(file,np.repeat(ids,2,axis=1),fmt='f %d/%d %d/%d %d/%d')
(O/'room.mtl').write_text(''.join(f'newmtl photo{i}\nKa 1 1 1\nKd 1 1 1\nillum 1\nmap_Kd {g["texture"]}\n\n' for i,g in enumerate(groups)))
with zipfile.ZipFile(O/'textured-room.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['room.obj','room.mtl']+[g['texture'] for g in groups]:z.write(O/name,name)
print('exported',len(rows)//3,'faces, textures',len(groups))
