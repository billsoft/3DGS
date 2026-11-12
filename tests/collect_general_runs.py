"""Verify completed integration outputs; publish aggregate evidence only."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scene_fusion'))
from run_state import digest

p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
rows=[]
for name,count in [('general-two-v2',2),('general-four-v2',4),('general-six-v2',6),('general-synthetic-v2',2)]:
    out=a.base/name
    def read(path):return json.loads((out/path).read_text(encoding='utf8'))
    manifest=read('scene.json');state=read('run-state.json');quality=read('quality.json')
    native=read('fit-hd/native-1024/audit.json');evaluation=read('fit-hd/eval-1024/evaluation.json');bundle=read('photo-bundle/report.json')
    assert manifest['count']==count and state['status']=='complete' and quality['passed'],name
    assert all(s['status']=='complete' for s in state['stages'].values()),name
    sha=digest(out/'fit-hd/optimized.npy')
    assert sha==quality['candidate_sha256']==native['candidate_sha256']==evaluation['candidate_sha256'],name
    assert native['summary']['views']==count*6,name
    for source in manifest['stations']:
        assert digest(source['photo'])==source['photo_sha256'],name+' original photo changed'
        assert digest(source['ply'])==source['ply_sha256'],name+' original PLY changed'
    rows.append(dict(scene=name,synthetic='synthetic' in name,sources=count,labels=[s['label'] for s in manifest['stations']],
        stages=len(state['stages']),quality=quality,native=native['summary'],fit=evaluation['summary'],export=read('final/export-check.json'),
        held_tracks=bundle['held_tracks'],held_before=bundle['before_held'],held_after=bundle['after_held'],
        original_inputs_unchanged=True,browser=read('browser/browser-check.json') if (out/'browser/browser-check.json').exists() else None))
resume=(a.base/'general-synthetic-v2/resume-check.log').read_text(encoding='utf8')
reused=[line[6:] for line in resume.splitlines() if line.startswith('REUSE ')]
assert len(reused)==19 and not any(line.startswith('START ') for line in resume.splitlines())
result=dict(unit_tests=23,resume=dict(passed=True,reused_stages=reused),real_data_note='2,4,6 are subsets of the same six-view real room, not three independent rooms.',
    synthetic_note='Independent procedural room; two cameras, Unicode labels, PLY unit scale factors 0.01 and 100.',runs=rows)
a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print('Verified',len(rows),'complete runs, candidate hashes and immutable original inputs')
