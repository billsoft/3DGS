# Surface-aware 3D Gaussian color filtering

`scene_fusion/bilateral_appearance.py` implements a conservative graph color
operator on a static SH PLY checkpoint. This is a 3D neighborhood operation,
not screen-space postprocessing, atmospheric dehazing, or a geometry repair.

The neighborhood uses k-nearest Gaussian centers, an adaptive radius capped by
global point spacing, shortest-axis normal agreement, local plane offsets,
Gaussian thickness, DC color similarity and directional SH compatibility.
Nearly isotropic Gaussians have no dependable shortest-axis normal and are
excluded. Opposite normal signs are allowed; distinct nearby thin layers are
kept separate by the thickness test. Support must exceed three weighted neighbors.

The small-neighborhood bilateral mean suppresses color noise. An optional
unsharp term uses the difference between small and larger bilateral means,
rather than sharpening raw noisy colors. Changes are limited to 0.025 per RGB
channel. Centers, scales, orientations, opacity and all non-DC SH coefficients
remain unchanged. The resulting checkpoint exports as a standard SH PLY.

Reference: [Open3D point-cloud bilateral smoothing](https://www.open3d.org/docs/latest/python_api/open3d.geometry.PointCloud.html)
and its [implementation interface](https://github.com/isl-org/Open3D/blob/main/cpp/open3d/geometry/PointCloud.h).
Open3D describes geometric point-cloud smoothing. Our independent NumPy/SciPy
implementation filters Gaussian colors and adds thickness/appearance gates;
it does not directly call Open3D or copy its implementation.

## Reproduction

```powershell
python scene_fusion/bilateral_appearance.py --reference path/to/reference --out path/to/fresh-candidate
# Denoising ablation, without the unsharp term:
python scene_fusion/bilateral_appearance.py --reference path/to/reference --out path/to/fresh-denoise --sharpen 0
```

The source contains `optimized.npz` and the tour teacher dataset. Output is kept
separate. Existing rendered comparison/export tools can evaluate the checkpoint.
Tests cover constant color, noise suppression, color boundaries, preservation of
geometry/opacity/SH-rest, neighboring thin layers and unreliable isotropic normals.

The six-camera experiment leaves roughly 94% of Gaussians untouched because
they lack sufficient compatible surface neighbors. Relaxing these gates merely
to produce a stronger visible effect risks the color leakage already seen in
this project. Broad haze can instead result from overlapping translucent layers,
large splat footprints or details absent from the supervision. This operator
cannot be claimed to solve those problems. See the experiment summary for
rendered results; do not promote a trial just because it looks more contrasted.

## Six-camera result, 2026-11-13

The thickness-gated version affects 80,683 of 1,367,420 Gaussians (5.9%). At 640
pixels, each variant was compared on 36 source views and 264 held-out views.
Anchor SSIM: baseline 0.983973743, denoise-only 0.983972477, denoise+sharpen
0.983968040. Denoise-only MAE slightly decreases, while SSIM is essentially
unchanged; sharpening does not improve these metrics. Screenshots show no
convincing haze reduction. These scores use synthetic original-PLY references,
so they are not proof of photographic correctness or perceptual preference.

The combined variant is exported solely as a labeled comparison experiment,
not promoted to the preferred scene. The earlier v1 lacked the additional
thickness gate and is rejected after the adjacent-layer unit test exposed mixing.
