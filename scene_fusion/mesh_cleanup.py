"""Conservative removal of small disconnected mesh islands with free-space evidence.

Absence of observations is not evidence of empty space. Keep unknown islands,
thin structures connected to a larger surface, and explicitly protected geometry.
All size gates are relative to this mesh's sampling spacing, not room coordinates.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def remove_unsupported_islands(p, tri, views, spacing, protected=None):
    if not len(tri):
        return np.ones(0, bool), dict(components=0, removed_faces=0, candidates=[])
    protected = np.zeros(len(tri), bool) if protected is None else protected
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    graph = coo_matrix((np.ones(len(edges), bool), (edges[:, 0], edges[:, 1])), shape=(len(p), len(p))).tocsr()
    _, labels = connected_components(graph, directed=False)
    ids, component = np.unique(labels[tri[:, 0]], return_inverse=True)
    count = np.bincount(component)
    area = np.linalg.norm(np.cross(p[tri[:, 1]]-p[tri[:, 0]], p[tri[:, 2]]-p[tri[:, 0]]), axis=1)*.5
    total_area = np.bincount(component, weights=area)
    is_protected = np.bincount(component, weights=protected.astype(int)) > 0
    # A long wire is not a tiny island, even when represented by few faces.
    candidates = np.flatnonzero((count <= 128) & (total_area < 32*spacing**2) & ~is_protected)
    large = ~np.isin(component, candidates)
    stable_vertices = np.unique(tri[large])
    tree = cKDTree(p[stable_vertices]) if len(stable_vertices) else None
    keep = np.ones(len(tri), bool); records = []
    for c in candidates:
        faces = np.flatnonzero(component == c); vertices = p[np.unique(tri[faces])]
        diameter = float(np.linalg.norm(np.ptp(vertices, axis=0)))
        if diameter > 8*spacing:
            continue
        gap = float(tree.query(vertices, k=1)[0].min()) if tree is not None else 0.
        if gap < spacing:
            continue  # Do not widen a crack adjacent to a reliable surface.
        centers = p[tri[faces]].mean(1)
        probes = np.concatenate([centers, vertices])
        support = np.zeros(len(probes), int); free = support.copy(); supported_views = 0
        for view in views:
            delta, confidence, distance = view.sample(probes)
            tolerance = np.maximum(1.5*spacing, .02*distance)
            near = (confidence > .25) & (np.abs(delta) < tolerance)
            contradiction = (confidence > .25) & (delta > 2*tolerance)
            support += near; free += contradiction
            supported_views += near.mean() >= .6
        fraction = float(((free >= 2) & (free > support) & (support < 2)).mean())
        rejected = fraction >= .7 and supported_views < 2
        if rejected:
            keep[faces] = False
        records.append(dict(component=int(c), faces=len(faces), diameter=diameter, gap=gap,
                            supporting_views=int(supported_views), contradictory_fraction=fraction,
                            removed=bool(rejected)))
    return keep, dict(components=len(ids), removed_faces=int((~keep).sum()),
                      removed_components=sum(r['removed'] for r in records), candidates=records,
                      method='Small detached islands, contradicted by at least two confident predicted-depth views; unknown/protected geometry retained')
