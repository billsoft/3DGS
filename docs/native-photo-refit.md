# One panorama per native PLY

`scene_fusion/native_photo_refit.py` fits each native model exclusively to its
own full-resolution panorama. It consumes the calibrated scene manifest and
native package, validates station order and source hashes, and supports the
station count in those manifests without a fixed four/six-camera assumption.

The current reader supports degree-zero Gaussian PLYs; nonzero higher-order SH
is rejected rather than silently discarded. Exact field preservation requires
binary PLY. Source panoramas in this experiment are 7680×3840.

```powershell
.\run-refit-gpu.ps1 scene_fusion/native_photo_refit.py `
  --scene four-room/scenes/geometry-quads-v1 `
  --native four-room/scenes/native-source-six-v1 `
  --out four-room/scenes/native-photo-six-v3 `
  --size 1536 --steps 1200
```

Use a fresh output directory. Each station is processed sequentially on the GPU.
The original panorama is resampled into perspective views. Twenty-four training
orientations overlap at yaw offsets 0/22.5/45/67.5 degrees. Only per-Gaussian RGB
is optimized, through alpha-composited rendering with fixed geometry/opacity.
The loss combines L1, structural similarity, image-gradient agreement, and a
small original-color prior. RGB is bounded to the range supported by the viewer.
There is no sharpening applied to the displayed image and no texture overlay.

Export edits only the three DC color fields in a copy of the original binary
PLY. The header, all non-color vertex fields (including the original quaternion
bytes), and ancillary data are preserved. The existing native coordinate
transforms and camera centers are reused. Each resulting PLY is independently
usable in a degree-zero Gaussian viewer; the continuous tour still uses multiple
models, not a newly fused single PLY.

## Checks and limitations

Each station is checked at its capture center in six cube directions plus six
directions with a withheld 11.25-degree yaw. Checks use 1024-square views and
the original model's alpha > 0.95 support. Acceptance requires every checked
view's MAE to increase by no more than 0.001 and SSIM to decrease by no more
than 0.002, plus an improvement in mean MAE. A failed station retains its
original file in the playback package; its candidate and comparisons remain
available for diagnosis. The UI explicitly identifies fallback stations.

These are checks against the **same photograph**, not unseen-position ground
truth. They do not guarantee every local region or every nearby viewpoint will
improve. A calibrated one-to-one photo/PLY pair can still have residual image
alignment, depth, occlusion, and footprint errors. Fixed coarse Gaussian
footprints cannot reproduce arbitrary photographic detail. Color-only fitting
cannot reconstruct unseen surfaces or remove geometry-induced layering.

The first experiment trained only six fixed cube faces. All six stations failed
the per-direction acceptance gate despite lower average photo error: rotated
views exposed rasterization-specific overfitting. This motivates the overlapping
orientations and higher training resolution in the second experiment. Neither
experiment replaces the accepted original tour assets.

The v3 package uses this second training configuration with byte-preserving
export. The first v2 station's fitted colors were reused after full revalidation;
its exporter had triggered an exact-geometry check because normalizing an
already normalized quaternion changed floating-point bytes. The revised export
copies the source quaternion bytes instead of normalizing them again.

`tests/test_native_photo_io.py` checks color-only binary export, both byte orders,
unaltered non-unit quaternions and ancillary data, and rejection of nonfinite
colors. `tests/check_native_tour.mjs <scene>` checks all station endpoints in the
actual browser, upward/downward/horizontal views, normalized transition weights,
capture-center invariance on rotation, and runtime exceptions.

`tests/check_native_photo_exports.py <calibrated-scene> <output> <native-input>`
checks real output files against every source non-color field, source hashes,
normalized calibration caches, and native playback camera transforms.
