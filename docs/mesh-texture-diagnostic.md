# Geometry versus projection-boundary ablation, 2026-11-13

The user requested stronger low-pass filtering to remove jagged gray regions near
lamps, with intentional detail loss permitted. No object masks or room coordinates
were introduced. Both accepted presets remain unchanged.

## Results

At source 4, cubemap direction 4, using the real browser camera matrix at 1280×900:

| Experiment | RGB MAE against original panorama |
| --- | ---: |
| Accepted smooth mesh, multi-photo texturing | 0.0150183256 |
| Wider bilateral normal filter and new visibility/texturing | 0.0162178513 |
| Connected projection-conflict texture regions, unchanged geometry | 0.0150179682 |
| Single-photo diagnostic, unchanged accepted geometry | 0.0017688374 |

At source 1 direction 4, baseline MAE is 0.0187139967 and the single-photo
diagnostic is 0.0018987144. Visual inspection shows a substantial disappearance of
the gray jaggies in the diagnostic. The mesh remains identical, so stronger mesh
smoothing is not necessary to remove this particular source-view artifact.

The diagnostic deliberately ignores multi-view visibility; a photograph projected
from its own center can conceal depth errors. This is **not** evidence of correct
geometry, nor a usable one-photo scene texture. The remaining root-cause split
between local pose error, reconstructed depth error, and texture ownership is not
yet resolved. The viewer uses unlit MeshBasicMaterial; these are not real-time
cast shadows.

Both actual repair candidates failed the lamp objective at the initial review.
Subsequently the user inspected the conflict-region candidate across the scene,
preferred its improvements elsewhere, and explicitly accepted it as the default
texture algorithm. `reconstruct.py` recipe version 2 therefore enables this step
for both geometry presets; `--legacy-texture-partition` restores the earlier
partition behavior for comparison in a new output directory. The wider normal
filter is still not promoted. Lamp defects remain unresolved. Existing accepted
model files remain unchanged; the new acceptance manifest records the candidate.

## Reproduction

Use the existing mesh Open3D environment for geometry and fusion environment for
texturing. All paths below are relative to the repository, on native Windows.

```powershell
python scene_fusion/mesh_feature_denoise.py --mesh four-room/scenes/mesh-lowpass-geometry-six-v3/surface.npz --cameras four-room/scenes/geometry-smooth-depth-strong-six-v1 --out four-room/scenes/mesh-wide-normal-geometry-six-v1 --iterations 16 --fit-iterations 32 --cap-ratio 1.5 --normal-bandwidth .35 --min-cosine 0
python scene_fusion/textured_mesh.py --source four-room/scenes/geometry-smooth-depth-strong-six-v1 --geometry four-room/scenes/mesh-wide-normal-geometry-six-v1/surface.npz --out four-room/scenes/mesh-wide-normal-photo-six-v1 --smoothing 0 --texture-width 7680 --footprint --warped --visibility-cache four-room/scenes/mesh-wide-normal-geometry-six-v1/visibility.npz --coherent-patches
python scene_fusion/textured_mesh.py --source four-room/scenes/geometry-smooth-depth-strong-six-v1 --geometry four-room/scenes/mesh-lowpass-geometry-six-v3/surface.npz --out four-room/scenes/mesh-conflict-photo-six-v1 --smoothing 0 --texture-width 7680 --footprint --warped --visibility-cache four-room/scenes/mesh-lowpass-geometry-six-v3/visibility.npz --coherent-patches --conflict-regions
node tests/capture_mesh_diagnostics.mjs mesh-lowpass-photo-six-v4 3 4
```

For the diagnostic, serialize `C` and `Q` from aligned-cameras.npz into a local
JSON object with those keys, then run `tests/capture_mesh_single_photo.mjs SCENE
CAMERA_JSON STATION_INDEX FACE_INDEX`. It uses the selected source's existing
texture image (including its small baked feather bands), rewrites UVs only in the
temporary browser session, and saves a PNG. It does not modify the exported mesh.

The widened-normal experiment preserved connectivity and bounded displacement;
7,010 vertices rolled back to prevent face collapse/flips. The conflict-region
experiment reassigned 2,816 faces in 151 supported small regions, insufficient to
change the inspected lamp. It never expands visibility or fills colors.

## Next algorithmic target, not yet implemented

Test locally aligned multi-view texture coordinates before final view selection
and seam feathering. Require forward/backward correspondence and depth/occlusion
gates, bounded displacement, and held-out original and oblique views. Do not
globally average conflicting colors or force all surfaces to one camera.

Relevant primary sources:

- [MVS-Texturing implementation](https://github.com/nmoehrle/mvs-texturing/blob/master/apps/texrecon/texrecon.cpp): view selection and separate global/local seam leveling.
- [Fu et al., Texture Mapping for 3D Reconstruction With RGB-D Sensor, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Fu_Texture_Mapping_for_CVPR_2018_paper.pdf): geometry/camera errors can require texture-coordinate alignment beyond color correction.

These references were consulted; their code was not integrated in these ablations.
