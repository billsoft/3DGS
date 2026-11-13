"""Integration check on real native/photo pairs and the exported playback package."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from input_io import header,read_gaussians,normalize

def fields(path):
    with Path(path).open('rb') as f:
        fmt,count,props=header(f);offset=f.tell();f.seek(0);prefix=f.read(offset)
        endian='<' if fmt=='binary_little_endian' else '>'
        data=np.fromfile(f,dtype=[(n,endian+t) for n,t in props],count=count)
        return prefix,data,f.read()

scene,out,native=map(Path,sys.argv[1:])
manifest=json.loads((scene/'scene.json').read_text('utf8'))
package=json.loads((out/'native.json').read_text('utf8'))
reference=json.loads((native/'native.json').read_text('utf8'))
for key in ('centers','views','up','midpoint_edges'):
    assert package[key]==reference[key],key
checks=[]
for source,entry,old in zip(manifest['stations'],package['stations'],reference['stations'],strict=True):
    assert source['id']==entry['id']==old['id']
    assert entry['world_from_raw']==old['world_from_raw']
    raw=read_gaussians(source['ply']);normalized,_=normalize(raw)
    assert np.array_equal(normalized,np.load(scene/source['id']/'original.npy')),'stale normalized source'
    before,a,tail=fields(source['ply']);after,b,after_tail=fields(out/entry['file'])
    assert before==after and tail==after_tail and a.dtype==b.dtype and len(a)==len(b)
    dc=['f_dc_0','f_dc_1','f_dc_2']
    for k in a.dtype.names:
        if k not in dc:assert a[k].tobytes()==b[k].tobytes(),k
    assert hashlib.sha256((out/entry['file']).read_bytes()).hexdigest()==entry['sha256']
    assert hashlib.sha256(Path(source['ply']).read_bytes()).hexdigest()==source['ply_sha256']
    changed=np.any(np.stack([a[k]!=b[k] for k in dc]),axis=0)
    checks.append(dict(id=entry['id'],points=len(a),changed_colors=int(changed.sum()),all_non_color_fields_byte_identical=True,cache_matches_source=True,hash_verified=True))
result=dict(passed=True,stations=checks)
(out/'export-check.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result))
