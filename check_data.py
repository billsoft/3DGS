"""Check the private prerequisites without importing scene-dependent modules."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parent

def missing_inputs():
    room = ROOT / 'four-room'
    inputs = Path(os.environ.get('GS_INPUT_DIR', str(ROOT / 'data/input')))
    paths = [room / name for name in (
        'registration.json', 'fusion.json', 'photo-bundle/bundle.npz',
        'photo-bundle/depth-report.json', 'surface-fusion-v3/mesh.bin',
        'surface-fusion-v3/mesh.json', 'surface-fusion-v3/fused.ply',
        'surface-fusion-v3/fused.splat', 'surface-fusion-v3/fusion.json',
        'surface-fusion-v4/inspection-views.json')]
    for station in ('2', '3', '4', '5'):
        paths.extend([inputs / f'{station}_7680.jpg', inputs / f'{station}.ply',
                      room / station / 'cache.npz'])
        for face in range(6):
            paths.extend([room / station / f'depth-{face}.npz',
                          room / 'colliders' / f'{station}-{face}.npy'])
    metadata = room / 'surface-fusion-v3/mesh.json'
    if metadata.is_file():
        paths.extend(metadata.parent / g['texture']
                     for g in json.loads(metadata.read_text())['groups'])
    return [p for p in paths if not p.is_file()]

if __name__ == '__main__':
    missing = missing_inputs()
    for path in missing:
        print(f'MISSING: {path}')
    print(f'{len(missing)} missing prerequisites' if missing else 'All required input files exist')
    raise SystemExit(bool(missing))
