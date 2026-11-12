"""Prepare calibrated photo constraints independently of the CUDA environment."""
from common import *
import argparse


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args()
    size = args.size
    images = photos()
    yy, xx = np.indices((size, size))
    rays = np.stack([(xx + .5 - size/2)/(size/2),
                     (yy + .5 - size/2)/(size/2), np.ones_like(xx)], -1)
    views, targets = [], []
    def add(name, center, basis, source, role):
        view = np.eye(4)
        view[:3, :3] = basis.T
        view[:3, 3] = -basis.T @ center
        views.append(dict(name=name, center=center.tolist(), basis=basis.tolist(),
                          viewmat=view.tolist(), source=source, role=role))
        if source is not None:
            tx = uv((rays @ basis.T @ Q[source]).reshape(-1, 3))
            targets.append(sample(images[source], tx).reshape(size, size, 3)[:, :, ::-1])
        else:
            targets.append(np.zeros((size, size, 3), 'u1'))
    for i, station in enumerate(stations):
        for face, b in enumerate(rm.f.prep.FACES):
            add(f'source-{station}-{face}', C[i], Q[i] @ b, i, 'train')
    up = np.array(rm.meta['up'])
    def look(center, target):
        z = target-center; z /= np.linalg.norm(z)
        x = np.cross(z, up); x /= np.linalg.norm(x)
        return np.stack([x, np.cross(z, x), z], 1)
    detail = json.loads((ROOT/'four-room/surface-fusion-v4/inspection-views.json').read_text())
    for i, v in enumerate(detail[:2]):
        add(f'detail-{i}', C[i], look(C[i], np.array(v['target'])), i, 'detail')
    # Midpoints are geometry/coverage guards, not invented reference photographs.
    for i in range(4):
        center = (C[i] + C[(i+1) % 4])/2
        add(f'mid-{i}-room', center, look(center, np.array([4., 0., -.6])), None, 'novel')
        add(f'mid-{i}-ceiling', center, Q[i] @ rm.f.prep.FACES[4], None, 'novel')
        add(f'mid-{i}-table', center, Q[i] @ rm.f.prep.FACES[5], None, 'novel')
    np.save(OUT/'targets.npy', np.stack(targets))
    baseline = ROOT/'four-room/baselines/2026-11-11-coverage-guard/fused.ply'
    np.save(OUT/'initial.npy', rm.f.prep.read_gs(baseline))
    for station in stations:
        np.save(OUT/f'native-{station}.npy', rm.f.prep.read_gs(
            ROOT/f'four-room/fusion-native-fidelity/native-{station}.ply'))
    write_json('dataset.json', dict(size=size, views=views, stations=stations,
        centers=C.tolist(), up=up.tolist(), baseline=str(baseline),
        note='24 cube faces come from four physical cameras, not 24 independent positions.'))
    print('Prepared', len(views), 'views', flush=True)


if __name__ == '__main__':
    run()
