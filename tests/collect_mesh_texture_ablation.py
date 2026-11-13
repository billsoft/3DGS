"""Record independent lamp-edge ablations and verify accepted artifacts unchanged."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SCENES=ROOT/'four-room/scenes'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    presets=json.loads((ROOT/'experiments/2026-11-13-reconstruction-workflow/accepted-presets.json').read_text('utf8'))
    old=json.loads((ROOT/'experiments/2026-11-13-geometry-guide/accepted-baseline.json').read_text('utf8'))
    checked=[]
    for entry in old['artifacts']+[a for c in presets['candidates'] for a in c['artifacts']]:
        path=SCENES/entry['path'];actual=sha(path)
        assert actual==entry['sha256'],f'Accepted artifact changed: {path}'
        checked.append(dict(path=entry['path'],sha256=actual))
    records={}
    for name in ['compare-up4','compare-up1']:
        records[name]=json.loads((SCENES/'mesh-conflict-photo-six-v1/diagnostics'/f'{name}.json').read_text('utf8'))
    artifacts=[]
    for scene in ['mesh-wide-normal-photo-six-v1','mesh-conflict-photo-six-v1']:
        for name in ['mesh.bin','mesh.json','textured-room.zip']:
            path=SCENES/scene/name;artifacts.append(dict(path=f'{scene}/{name}',sha256=sha(path)))
    out=ROOT/'experiments/2026-11-13-texture-diagnostic';out.mkdir(exist_ok=True)
    result=dict(status='both-repair-candidates-rejected; accepted-presets-unchanged',
        verified_baseline_artifacts=checked,experiments=records,artifacts=artifacts,
        limitation='Single-photo self-view projection is diagnostic only; it does not prove geometry or roaming quality.',
        next_target='Local multi-view projection alignment before seam selection, not further global low-pass filtering')
    (out/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('Verified accepted artifacts:',len(checked));print('Saved',out/'evidence.json')
if __name__=='__main__':main()
