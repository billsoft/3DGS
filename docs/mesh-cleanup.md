# Mesh fragments, holes and partial-visibility regression

The native continuous-tour player is still the visual reference. It stores six
complete models (3,051,998 total Gaussians in this experiment), selects one when
stationary, and blends briefly while moving. `native-tour.js` switches to a
single source after 180 ms without translation, with tie hysteresis. The
917,420-point distilled model instead approximates deterministic 44% spatial
blending, without that temporal state. Even at anchor cameras where the teacher
is pure, fitting a smaller common model is not lossless. This explains why the
native player remains sharper; it is not evidence that overlapping six meshes
would improve the reconstruction.

## Diagnosed stages

`mesh_stage_audit.py` casts identical calibrated rays through the pre-texture and
published meshes. It records missed rays, connected components, boundary edges,
and nonmanifold edges. Missed-ray totals include unobserved regions, not just
visible pinholes; they must be read together with actual browser images.

- The original TSDF surface already has 12,830 components and 85,608 boundary
  edges. Its generated depth is not a measured consistent surface.
- The old texture stage increases boundary edges to 121,656. For camera 1's
  downward 512-square view, missed rays increase from 4,570 to 5,407.
- The old small planar-loop repair reduces that view only to 5,401 misses.
- Both raster and initial ray visibility kept only the face centroid's result.
  A hidden centroid deleted a whole triangle even if its corners were visible.
  The corrected rule retains any sampled visible footprint; coverage still
  influences appearance-source quality. It does not make an entirely hidden
  face visible, nor guarantee visibility of every pixel between probes.

## New repair path

`mesh_surface_repair.py` uses Open3D's connected-triangle components and bounded
boundary triangulation. Fragment thresholds combine face count, area and spatial
diameter, all relative to sampling spacing. Long thin structures survive the
diameter gate. Small isolated genuine details can nevertheless be removed;
this is an explicit experimental tradeoff, not semantic noise detection.

Hole search can be widened, but caps wider than six spacings must be planar to
within 0.35 spacing. This avoids flattening large curved gaps. Duplicate faces
and caps that overfill an edge are rejected. Original vertex positions stay
unchanged. Caps are inferred geometry, not recovered measurements.

Exact camera-to-surface rays provide four footprint samples per triangle. The
cached results carry geometry and camera hashes and are rejected after geometry
changes. Baking with the cache requires zero additional smoothing. Source
photographs remain full resolution; existing narrow seam feathering is reused.
Protected cap geometry with no texture support remains visibly neutral rather
than being textured through an occluder.

## Reproduce on Windows

Run repair/audit in the isolated `requirements-mesh.txt` environment. Run baking
in the existing CPU photo environment. Native tour and previous candidates are
not overwritten.

```powershell
python scene_fusion/mesh_surface_repair.py --mesh four-room/scenes/single-gs-mesh-six-v1/surface.npz --cameras four-room/scenes/single-gs-depth-six-v1 --hole-radius 12 --out four-room/scenes/single-gs-repair-six-v5
python scene_fusion/textured_mesh.py --source four-room/scenes/single-gs-depth-six-v1 --geometry four-room/scenes/single-gs-repair-six-v5/surface.npz --warped --footprint --texture-width 7680 --smoothing 0 --visibility-cache four-room/scenes/single-gs-repair-six-v5/visibility.npz --out four-room/scenes/single-gs-photo-mesh-six-v6
python scene_fusion/mesh_stage_audit.py --cameras four-room/scenes/single-gs-depth-six-v1 --mesh four-room/scenes/single-gs-photo-mesh-six-v2/surface-owned.npz four-room/scenes/single-gs-photo-mesh-six-v6/surface-owned.npz --out four-room/scenes/mesh-cleanup-audit-v6
python tests/test_mesh_surface_repair.py
python -m unittest discover -s tests -p test_textured_mesh.py -v
node tests/check_photo_mesh.mjs single-gs-photo-mesh-six-v6
```

The browser check covers 18 source directions, six midpoints and geometry/owner
views. Same-view captures are shown by `mesh-repair-report.html`; the scene's
`repair-comparison.json` supplies the reference scene, image names, metrics and
limitations. No source photographs, models or captures are committed.

This pipeline does not restore missing chair backs or unobserved undersides.
Nor does it prove that all remaining disconnected pieces are artifacts. It
remains a separate preview until the user accepts its visual tradeoffs.

## Primary references

- [Open3D component analysis](https://www.open3d.org/docs/release/tutorial/geometry/mesh.html)
- [Open3D bounded hole filling](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.TriangleMesh.html#open3d.t.geometry.TriangleMesh.fill_holes)
- [CGAL hole filling and fairing](https://doc.cgal.org/latest/PMP_Mesh_repair/index.html): related reference, not a dependency or implemented CGAL pipeline here.
