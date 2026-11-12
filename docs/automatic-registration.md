# Automatic panorama registration and camera anchoring

The runner estimates camera poses from photo correspondences; generated PLY
depth provides an initialization and a separate depth-scale estimate. It does
not assume four/six stations, clockwise capture order, a room dimension, or
manually edited camera coordinates. This revision improves robustness and
failure handling; it does not establish that the current fusion is geometrically
correct everywhere or that registration alone removes holes and floaters.

## Pipeline

1. Mutual SIFT matching and 3D Sim(3) RANSAC reject collinear minimal samples.
   Pair thresholds use the observed point extent rather than an absolute room
   length. Matches occupy image cells; a small textured region must not dominate.
2. A maximum-confidence spanning tree initializes a connected pose graph.
   Translation units and residuals are normalized. Each pair has equal total
   mass and each occupied cell equal mass. Robust point/edge weights reduce
   inconsistent loops. Disconnected graphs and failed optimization stop the run.
3. Spherical photo bundle adjustment fixes the first pose and its longest
   baseline. Scene-normalized coordinates replace fixed-world-distance cutoffs.
   Generated depth is not an observation in this optimization. Two robust
   reweighting passes estimate track noise from training observations only.
4. Whole withheld tracks validate every camera's median/P90 angular consistency,
   positive depth and minimum support. If the full update regresses a camera,
   automatic backtracking tests 1/2, 1/4, 1/8, then the initial solution. It fixes
   the same scale gauge at every step. Reports explicitly record fallback; if
   none passes, no new camera package is produced. This validation is used for
   model selection, so its scores are not an unbiased test benchmark.
5. PLY scale is recovered in log space with equal image-cell influence, robust
   majority consensus, and a cell bootstrap interval. Depth edges are excluded.
   The old clipping around the initial scale is removed. Inconsistent or
   concentrated evidence stops the stage. A single scalar cannot correct local
   depth deformation inside independently generated PLYs.

Algorithm safety thresholds still exist, but they are common defaults, not
per-scene hand-tuned camera offsets or scaling. Arbitrary photo counts do not
guarantee enough overlap, parallax or texture. Sparse repetitive scenes can fail.

## Usage

`run-scene.ps1` uses this calibration automatically. To compute only camera poses
and depth scales before spending GPU time on fusion:

```powershell
.\run-scene.ps1 -Inputs 'C:\path\to\panoramas-and-plys' -CalibrationOnly
```

Use a fresh output directory when testing a new algorithm. Stage checkpoints
include code hashes. Existing accepted PLYs, meshes and previews are not replaced
by the calibration-only command. See `registration.json`, `photo-bundle/report.json`
and `scales.json` in the output directory for errors, automatic update fraction,
per-camera support and scale uncertainty. Relative scale is arbitrary, not metres.

## Verification on 2026-11-12

- Six photos: 992 tracks, 197 withheld; validation median 1.108 to 0.571 equivalent
  pixels at panorama width 3840. Full update accepted.
- Four-photo subset (stations 1, 2, 4, 5): 578 tracks, 118 withheld. Full update
  rejected for one camera's P90; automatic half update passed. Median 0.764 to
  0.546, P90 4.269 to 3.728. No hand-entered pose changes.
- Fair comparison with the previously accepted six-camera calibration uses 80
  common tracks absent from both optimization residuals: median 0.700 to 0.690,
  but P90 2.810 to 3.144. Therefore this is **not a demonstrated overall visual
  improvement over the accepted calibration**. Accepted tours/meshes remain intact.
- Synthetic tests cover 2/4/6/9 cameras, unit scaling, a dense incorrect loop,
  disconnected input, local validation regression, automatic backtracking, and
  concentrated/inconsistent depth-scale evidence. The ordinary runner tests pass.

The four-photo subset is not the historical hand-refined four-photo reconstruction.
The comparison tool verifies feature identity and excludes shared observations
used for training in either run; its withheld tracks may still be used for model
selection. It produces an HTML camera-layout/error report, not a new scene model.

## References

The design follows robust graph optimization and photo bundle-adjustment
principles described by [Open3D multiway registration](https://www.open3d.org/docs/release/tutorial/pipelines/multiway_registration.html)
and [COLMAP documentation](https://colmap.github.io/faq.html).
The implementation here is independent NumPy/SciPy code; it does not import
COLMAP or copy the Open3D pose optimizer. Future work needs broader scenes and
rendered route validation before claiming a general accuracy improvement.
