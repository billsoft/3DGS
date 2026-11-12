# Oblique-view continuity and static PLY fidelity

The native tour remains the reference. This experiment adds two independently
inspectable candidates, without replacing it or the preceding mesh.

## What caused the discrepancy

A rendered Gaussian color does not imply a unique opaque surface. The previous
mesh used expected Gaussian depth, then TSDF integration, simplification,
fragment removal and visibility-based photo ownership. Depth disagreement can
produce disconnected sheets before texturing; removing unsupported faces can
open additional gaps. A front view can hide them behind another sheet. Double
sided materials were already enabled, so backface culling was not the cause.

The new evaluation adds 96 translated, grazing and route views to 18 original
camera directions. Browser captures and raycasts use the same calibrated poses;
the browser check verifies the CV/Three.js camera basis conversion. A ray miss
counts absent geometry, not all dark pixels, wrong textures or wrong shapes.

Tracing source-view ray misses back to the original rendered depth gives direct
stage evidence. Of the published v6 mesh's 18,860 misses in 18 source directions,
14,464 still have finite input depth in range and alpha >= 0.95. The pre-texture
TSDF/decimated mesh already has 19,721 misses, including 14,490 with valid input
depth. Thus most tested losses occur during depth-to-surface conversion, before
photo ownership; they are not explained by missing photographs. This audit
does not separately attribute integration versus simplification, and the input
depth is synthetic rather than a measurement.

## Experiments and decision

All ray counts below use 114 views at 384 square pixels. These views do not prove
correctness at every possible position, and closed inferred surfaces can score
well despite an incorrect shape.

| Candidate | Missed rays | Decision |
|---|---:|---|
| Previous published mesh v6 | 90,849 | Retained baseline |
| Surface-oriented 2DGS, median depth, repaired before texture | 110,364 | Rejected as a hole repair |
| Automatic planar depth consolidation, final textured mesh | 96,550 | Rejected as a hole repair |
| Screened Poisson, final textured mesh | 0 | Separate preview, with reconstruction caveats |

2DGS used the installed gsplat 1.5.3 backend, shortest-axis surface initialization,
source-photo fitting, normal consistency and ray-distortion losses. It reduced
pre-cleanup connected components from 12,830 to 3,533, but increased ray misses.
Its intermediate surface Gaussians are **not** exported as a normal 3DGS PLY.

The planar experiment estimated scene planes using RANSAC, required support
from multiple stations, and moved only high-alpha locally aligned depth samples.
It preserved foreground discontinuities and did not fill unknown depths. This
was insufficient to repair the existing topology and is not a default.

The selected mesh experiment uses oriented samples from the cleaned accepted
TSDF surface, Screened Poisson reconstruction, bounded component removal, exact
ray visibility and original 7680x3840 photographs. It does not discard a validated
surface solely because no source photograph can texture it. Such faces remain
neutral gray rather than receiving a photograph through an occluder.

The result has 1,590,030 faces, including 196,337 without usable source texture
(a face count, not an image-area percentage). Boundary edges fell from 69,586
to zero, but 981 nonmanifold edges remain: this is a renderable surface preview,
not a certified manifold solid or CAD asset. Among rays that already hit the old
mesh, the median across views of median absolute depth change is 0.00114 scene
units; the median across views of the 95th percentile is 0.01094. Thin structures,
lamps, chair backs and inferred unseen regions remain visibly imperfect.

An additional material-coverage audit separates gray fallback from color seams:
at source camera 1 looking upward, only 2/147,456 rays hit an untextured face.
Most visible ceiling blotches therefore are not missing geometry or the neutral
fallback material; projected photo ownership, appearance disagreement and
residual geometry/projection errors remain. Closure alone does not fix them.

## Static PLY refinement

`tour_refine.py` keeps all original teacher models intact and trains a single
standard SH2 PLY at 1024 resolution. It splits large high-error Gaussians in their
longest local axis, increasing capacity from 917,420 to 1,367,420 points. It adds
oblique supervision and uses a 49% spatial core. Exactly 50% would collapse the
smooth transition to a nearest-source boundary; neither reproduces temporal
rest selection and hysteresis in the interactive player.

At the same 1024-resolution views against the **same 49% synthetic teacher**:

| Views | Previous PLY SSIM | New PLY SSIM | Previous MAE | New MAE |
|---|---:|---:|---:|---:|
| 36 original directions | 0.97328 | 0.98142 | 0.00889 | 0.00834 |
| 264 untrained positions or directions | 0.94886 | 0.95547 | 0.01528 | 0.01484 |

34/36 source directions and 246/300 total evaluation views improve in SSIM.
The strict per-source gate still fails. This is an improvement, not lossless
baking of the native tour and not an independent photographic accuracy test.
Some held-out oblique directions share positions with differently oriented
training views. The older route/near-position holdout is still disjoint.

Further 384-resolution evaluation exposed a regression in this HD-only v2:
anchor SSIM 0.96995 → 0.96558 and held-out SSIM 0.93923 → 0.93578. Its HD result
must not be presented as resolution-independent improvement. A subsequent
candidate resumes these points with mixed 384/640/1024 supervision and no more
splits. The changed sampling rate is explicitly tested instead of comparing only
the training resolution. This is multi-resolution fitting with the existing
standard renderer, not an implementation of Mip-Splatting's filters.

The final preview is **tour-distilled-six-v3**, after 3,000 additional mixed-size
steps with the same 1,367,420 points. Against the same original v1 reference:

| Resolution | Source SSIM, old → v3 | Held-out SSIM, old → v3 |
|---|---:|---:|
| 384 | 0.96995 → 0.98220 | 0.93923 → 0.94715 |
| 1024 | 0.97328 → 0.97861 | 0.94886 → 0.95465 |

This trades some v2 HD gain for better cross-resolution behavior. Both scales
improve over v1 on average, but neither passes the strict every-source gate.
The report exposes both resolutions rather than hiding the lower-scale result.

The exported PLY roundtrips exactly, loads independently as one SH2 PLY in the
actual browser, and is checked at seven browser poses against GPU rendering.
No original PLY or photograph is selected or overlaid during its playback.

## Reproduce

Run mesh commands in the isolated `requirements-mesh.txt` environment, photo
baking in the existing CPU photo environment, GPU commands with the native
Windows `run-refit-gpu.ps1` wrapper. Paths below are generated scene folders,
not checked-in assets. Camera count is read from scene metadata.

```powershell
python scene_fusion/oblique_views.py --source four-room/scenes/single-gs-depth-six-v1 --out four-room/scenes/oblique-audit-v1/views.json
python scene_fusion/poisson_surface.py --mesh four-room/scenes/single-gs-repair-six-v5/surface.npz --cameras four-room/scenes/single-gs-depth-six-v1 --out four-room/scenes/poisson-surface-six-v1 --depth 10
python scene_fusion/textured_mesh.py --source four-room/scenes/single-gs-depth-six-v1 --geometry four-room/scenes/poisson-surface-six-v1/surface.npz --out four-room/scenes/poisson-photo-mesh-six-v1 --texture-width 7680 --smoothing 0 --warped --footprint --visibility-cache four-room/scenes/poisson-surface-six-v1/visibility.npz
./run-refit-gpu.ps1 scene_fusion/tour_refine.py --reference four-room/scenes/tour-distilled-six-v1 --out four-room/scenes/tour-distilled-six-v2 --views four-room/scenes/oblique-audit-v1/views.json --steps 4000
./run-refit-gpu.ps1 scene_fusion/tour_distill.py evaluate --out four-room/scenes/tour-distilled-six-v2 --eval-size 1024 --evaluation-folder evaluation-hd
python scene_fusion/tour_distill.py export --out four-room/scenes/tour-distilled-six-v2 --native four-room/scenes/native-source-six-v1 --evaluation-folder evaluation-hd
node tests/check_mesh_oblique.mjs poisson-photo-mesh-six-v1
node tests/check_photo_mesh.mjs poisson-photo-mesh-six-v1
node tests/check_tour_distill_browser.mjs tour-distilled-six-v2
./run-refit-gpu.ps1 scene_fusion/tour_browser_audit.py --out four-room/scenes/tour-distilled-six-v2
```

For the final mixed-resolution candidate, use `tour_refine.py` with the same
reference and views, a fresh `--out`, `--resume` pointing to v2's checkpoint,
`--no-split --sizes 384 640 1024 --steps 3000`. Evaluate v3 independently at 384
and 1024, export it, and repeat the standard-PLY browser/parity checks.

## Primary references and implementation boundary

- [2DGS paper](https://arxiv.org/abs/2403.17888) and
  [official renderer](https://github.com/hbb1/2d-gaussian-splatting/blob/main/gaussian_renderer/__init__.py): surface Gaussians, depth and normal regularization; studied, not vendored.
- [gsplat 2DGS example](https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_trainer_2dgs.py): the installed gsplat implementation is used by the optional surface experiment.
- [PGSR](https://arxiv.org/abs/2406.06521): related plane/depth consistency research; the full PGSR pipeline is not implemented here.
- [DN-Splatter](https://github.com/maturk/dn-splatter): related indoor geometry and depth/normal priors; not a dependency.
- [Mip-Splatting](https://niujinshuchong.github.io/mip-splatting/): documents Gaussian dilation and sampling-rate artifacts. It informs the multi-resolution audit; its renderer and filters are not vendored here.
- [PoissonRecon](https://github.com/mkazhdan/PoissonRecon) and
  [Open3D surface reconstruction](https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html): Screened Poisson is used through Open3D. Its smooth extrapolation is why closure must be checked visually and cannot be equated with measured geometry.

The useful next geometry work is calibrated multi-view surface fitting and
occlusion-aware correspondence. More color averaging, larger overlap, or blind
hole deletion does not resolve conflicting depth. Mesh reconstruction and
single-PLY appearance distillation therefore remain separate validated outputs.
