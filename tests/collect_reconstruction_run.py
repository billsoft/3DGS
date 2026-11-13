"""Collect auditable, image-free records from an actual completed run."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from run_state import digest

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
def read(path):return json.loads(path.read_text('utf8'))
state=read(a.run/'run-state.json');recipe=read(a.run/'recipe.json')
assert state['status']=='complete'
for stage in state['stages'].values():
    assert stage['status']=='complete'
    for name,expected in stage['artifacts'].items():
        path=a.run/name
        if path.is_dir() and name in stage.get('directory_files',{}):
            import hashlib
            actual=hashlib.sha256(json.dumps([(f,digest(path/f)) for f in stage['directory_files'][name]],ensure_ascii=False).encode()).hexdigest()
        else:actual=digest(path)
        assert actual==expected,name
for item in recipe['input_scene']['stations']:
    assert digest(item['photo'])==item['photo_sha256']
    assert digest(item['ply'])==item['ply_sha256']
report=dict(profile=recipe['profile'],inputs=recipe['input_scene']['count'],source_hashes_verified=True,
    artifact_hashes_verified=True,stages=[dict(name=k,seconds=v['seconds']) for k,v in state['stages'].items()],
    camera_graph=read(a.run/'calibration/camera-graph.json'),
    bundle_acceptance=read(a.run/'calibration/photo-bundle/report.json')['acceptance'],
    ply=read(a.run/'single/final/validation.json'),meshes={},browser={},
    warning='Smoke run verifies execution, not production image quality. Synthetic two-camera room is not another real captured room.')
for name in ['adaptive','smooth']:
    meta=read(a.run/name/'mesh.json');report['meshes'][name]=dict(triangles=meta['triangles'],texture_size=meta['texture_size'],numeric_weld=meta['numeric_weld'])
    for kind in ['browser','oblique-browser']:
        path=a.run/name/kind/'check.json'
        if path.exists():report['browser'][name+'/'+kind]=read(path)
for name in ['single/browser','native/tour-browser']:
    path=a.run/name/'check.json'
    if path.exists():report['browser'][name]=read(path)
a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
print('Verified',len(state['stages']),'stages and original inputs')
