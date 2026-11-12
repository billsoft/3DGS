"""Unit-invariant Sim(3) graph fitting. No capture order or room dimensions."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def apply(m, x):
    return x @ m[:3, :3].T + m[:3, 3]


def pack(m):
    s = np.cbrt(np.linalg.det(m[:3, :3]))
    return np.r_[Rotation.from_matrix(m[:3, :3] / s).as_rotvec(), m[:3, 3], np.log(s)]


def unpack(v):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_rotvec(v[:3]).as_matrix() * np.exp(v[6])
    m[:3, 3] = v[3:6]
    return m


def fit_graph(edges, count):
    if count < 2:
        raise ValueError('At least two cameras are required')
    def confidence(e):
        k = e['train']
        error = np.median(np.linalg.norm(apply(e['m'], e['x'][k])-e['y'][k], axis=1))
        return min(e['cells'], 32) / (1 + error / e['tol'])
    ordered = sorted(edges.items(), key=lambda kv: -confidence(kv[1]))
    poses = {0: np.eye(4)}
    # Prim: grow through the strongest available cut edge, not input numbering.
    while len(poses) < count:
        for (i, j), e in ordered:
            if (i in poses) != (j in poses):
                poses[j if i in poses else i] = (poses[i] @ e['m'] if i in poses
                                                else poses[j] @ np.linalg.inv(e['m']))
                break
        else:
            raise ValueError(f'Disconnected registration graph: {sorted(set(range(count))-set(poses))}')
    unit = np.median([np.cbrt(np.linalg.det(poses[i][:3, :3]))*e['tol'] for (i, j), e in ordered])
    if not np.isfinite(unit) or unit <= 0:
        raise ValueError('Invalid registration scale')
    # Normalize translations too, so optimizer stopping tolerances are unit invariant.
    initial = np.concatenate([pack(poses[i]) for i in range(1, count)]).reshape(-1, 7)
    initial[:, 3:6] /= unit
    initial = initial.ravel()
    def matrices(v):
        ps = v.reshape(-1, 7).copy(); ps[:, 3:6] *= unit
        return [np.eye(4)] + [unpack(p) for p in ps]
    rows = []
    for (i, j), e in ordered:
        k = e['train']; n = int(k.sum())
        # Equal mass per occupied image cell, equal total mass per camera pair.
        cells = e.get('sample_cells')
        w = np.ones(n)
        if cells is not None:
            _, inverse, counts = np.unique(cells[k], axis=0, return_inverse=True, return_counts=True)
            w = 1. / counts[inverse]
        w = np.sqrt(w/w.sum())
        scale = np.cbrt(np.linalg.det(poses[i][:3, :3])) * e['tol']
        rows.append((i, j, e['x'][k], e['y'][k], w, scale))
    switches = np.ones(len(rows)); v = initial.copy(); rounds = []
    def norms(ms):
        return [np.linalg.norm(apply(ms[j], x)-apply(ms[i], y), axis=1)/scale
                for i, j, x, y, w, scale in rows]
    for _ in range(3):
        prior_errors = norms(matrices(v))
        switches = 1/(1+(np.array([np.median(r) for r in prior_errors])/2)**2)
        robust = [1/np.sqrt(1+(r/2)**2) for r in prior_errors]
        def residual(v):
            ms = matrices(v)
            return np.concatenate([((apply(ms[j], x)-apply(ms[i], y))/scale
                                    * (w*np.sqrt(switches[k])*robust[k])[:, None]).ravel()
                                   for k, (i, j, x, y, w, scale) in enumerate(rows)])
        lower = np.full_like(v, -np.inf); upper = -lower
        lower[6::7] = initial[6::7]-np.log(4); upper[6::7] = initial[6::7]+np.log(4)
        sol = least_squares(residual, v, bounds=(lower, upper), loss='linear',
                            max_nfev=120, x_scale='jac')
        if not sol.success or not np.isfinite(sol.x).all():
            raise ValueError('Global registration did not converge: '+sol.message)
        v = sol.x
        medians = np.array([np.median(r) for r in norms(matrices(v))])
        switches = 1/(1+(medians/2)**2)
        rounds.append(dict(cost=float(sol.cost), median_normalized_errors=medians.tolist()))
    report = dict(method='cell-balanced, unit-normalized robust Sim3 graph', rounds=rounds,
                  edge_weights={f'{i}-{j}':float(w) for ((i,j),_),w in zip(ordered,switches)},
                  translation_unit=float(unit))
    return matrices(v), report


def bundle_gate(before, after, success):
    """Held-out observations are used only for acceptance, never fitting weights."""
    reasons = []
    if len(before)!=len(after) or not len(after):
        return dict(passed=False,reasons=['Missing camera validation data'])
    if not success:
        reasons.append('Bundle optimizer did not converge')
    for i, (b, a) in enumerate(zip(before, after)):
        if len(a) < 8 or len(b)!=len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
            reasons.append(f'Camera {i}: insufficient finite held-out evidence')
            continue
        for quantile, ceiling in [(.5, 3.), (.9, 12.)]:
            bv, av = np.quantile(b, quantile), np.quantile(a, quantile)
            if av > max(bv*1.15, bv+.25) or av > ceiling:
                reasons.append(f'Camera {i}: held-out q{quantile} error regressed or too high')
    return dict(passed=not reasons, reasons=reasons)


def estimate_scale(ratios, cells):
    """Robust log-scale with equal image-cell weight and cell bootstrap CI."""
    ratios=np.asarray(ratios);cells=np.asarray(cells)
    good=np.isfinite(ratios)&(ratios>0)
    ratios=ratios[good];cells=cells[good]
    if len(ratios)<12:raise ValueError('Insufficient depth scale evidence')
    _,inverse=np.unique(cells,axis=0,return_inverse=True)
    logs=np.log(ratios)
    values=np.array([np.median(logs[inverse==i]) for i in range(inverse.max()+1)])
    if len(values)<4:raise ValueError('Depth scale evidence is spatially concentrated')
    center=np.median(values)
    for _ in range(8):
        sigma=max(.005,1.4826*np.median(abs(values-center)))
        weights=np.minimum(1,1.5*sigma/np.maximum(abs(values-center),1e-12))
        center=float(np.sum(values*weights)/weights.sum())
    consensus=abs(values-center)<3*sigma
    if consensus.sum()<4 or consensus.mean()<.6:
        raise ValueError('Depth scale is inconsistent: no spatial majority consensus')
    retained=values[consensus]
    center=float(np.median(retained))
    rng=np.random.default_rng(903)
    boot=np.median(rng.choice(retained,(512,len(retained)),replace=True),axis=1)
    interval=np.exp(np.quantile(boot,[.025,.975]));scale=float(np.exp(center))
    spread=float(np.quantile(abs(np.exp(retained-center)-1),.9))
    uncertainty=float((interval[1]-interval[0])/scale)
    if uncertainty>.3 or spread>.5:raise ValueError(f'Depth scale is inconsistent: cells={len(values)}, scale={scale:.4g}, interval={uncertainty:.3f}, spread={spread:.3f}; regenerate depth or add photo overlap')
    return scale,dict(cells=len(values),consensus_cells=int(consensus.sum()),rejected_cells=int((~consensus).sum()),samples=len(ratios),cell_scale_interval95=interval.tolist(),
                      relative_interval_width=uncertainty,relative_spread=spread)


def validated_update(C0,Q0,C,Q,evaluate,success):
    """Backtrack a proposed pose update on validation tracks, never manual poses.

    Validation is for model selection, not an unbiased final benchmark. Keeping
    the initial solution is explicit; insufficient initial evidence still fails.
    """
    before=evaluate(C0,Q0);rotation=Rotation.from_matrix(Q@Q0.transpose(0,2,1)).as_rotvec()
    peer=int(np.argmax(np.linalg.norm(C0-C0[0],axis=1)));base=np.linalg.norm(C0[peer]-C0[0])
    attempts=[]
    for fraction in (1.,.5,.25,.125,0.):
        cc=C0+(C-C0)*fraction
        length=np.linalg.norm(cc[peer]-cc[0])
        if length<1e-12:continue
        cc=cc[0]+(cc-cc[0])*(base/length)
        qq=Rotation.from_rotvec(rotation*fraction).as_matrix()@Q0
        gate=bundle_gate(before,evaluate(cc,qq),success)
        attempts.append(dict(fraction=fraction,**gate))
        if gate['passed']:return cc,qq,dict(fraction=fraction,initial_retained=fraction==0,attempts=attempts)
    raise ValueError('No camera update passed independent-track validation: '+str(attempts))
