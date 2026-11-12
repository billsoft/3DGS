# Accepted single Gaussian model: photo detail and mesh extraction

This experiment starts from the accepted `tour-distilled-six-v1` model. It does
not rebuild geometry from the independent input PLYs. The accepted model remains
unchanged. The camera registration must already exist; these tools do not solve
unknown panorama poses.

## Two independent outputs

1. **Fixed-geometry photo detail:** `view_color_fit.py` optimizes SH colors against
   original-photo cube views. Means, scales, rotations and opacity remain exact.
   `export_photo_detail.py` checks those invariants, the evaluation hash, and the
   full SH PLY round trip before publishing an independent standard PLY.
2. **Textured mesh:** `single_gs_mesh.py` renders expected projective depth from
   the accepted Gaussian model at all source cameras and pairwise midpoint
   cameras. Open3D integrates these synthetic depths into a TSDF and extracts a
   triangle mesh. Existing `textured_mesh.py` projects native-resolution photos
   onto visible faces with coherent ownership and narrow seam feathering.

The depth is **synthetic**, not measured. A visually continuous Gaussian model
can still have inconsistent surface depths. Mesh conversion can introduce
holes, layers, and thin-object deformation. Neither extraction nor photo
projection is a lossless conversion. Photo-supervised fitting scores reuse the
training photographs and do not prove accuracy between cameras.

## Windows execution

Use the existing CUDA/gsplat environment through `run-refit-gpu.ps1`. Install
`requirements-mesh.txt` in a **separate** environment for Open3D. The Windows
Open3D 0.20.0 wheel returned empty meshes even for a synthetic plane on the test
machine; 0.19.0 passed. Extraction runs a plane smoke test before reading scene
depth. CPU photo baking uses the project's existing NumPy/OpenCV/Numba/SciPy
environment.

Example paths below are generated, private scene folders; they are not bundled.

```powershell
./run-refit-gpu.ps1 scene_fusion/view_color_fit.py --data four-room/scenes/geometry-quads-v1/data-1024 --model four-room/scenes/tour-distilled-six-v1/optimized.npz --reference four-room/scenes/tour-distilled-six-v1/optimized.npz --out four-room/scenes/tour-photo-detail-six-v1 --steps 2200 --dc-lr .002 --sh-lr .001 --regularization .00005 --graph-weight .01 --blotch-weight .1
./run-refit-gpu.ps1 fusion_next/refit_evaluate.py --data four-room/scenes/geometry-quads-v1/data-1024 --candidate four-room/scenes/tour-photo-detail-six-v1/optimized.npz --baseline four-room/scenes/tour-distilled-six-v1/optimized.npz --baseline-label accepted-single-ply --out four-room/scenes/tour-photo-detail-six-v1/evaluation
python scene_fusion/export_photo_detail.py --reference four-room/scenes/tour-distilled-six-v1 --out four-room/scenes/tour-photo-detail-six-v1

./run-refit-gpu.ps1 scene_fusion/single_gs_mesh.py capture --source four-room/scenes/geometry-quads-v1 --model four-room/scenes/tour-distilled-six-v1/optimized.npz --out four-room/scenes/single-gs-depth-six-v1 --size 1024
# Run this command in the isolated Open3D environment:
python scene_fusion/single_gs_mesh.py integrate --source four-room/scenes/single-gs-depth-six-v1 --out four-room/scenes/single-gs-mesh-six-v1 --resolution 640 --max-faces 1600000
# Return to the CPU photo-baking environment:
python scene_fusion/textured_mesh.py --source four-room/scenes/single-gs-depth-six-v1 --geometry four-room/scenes/single-gs-mesh-six-v1/surface.npz --warped --footprint --texture-width 7680 --smoothing .2 --out four-room/scenes/single-gs-photo-mesh-six-v1

# Optional conservative small-hole repair experiment; no object masks or planes:
python scene_fusion/mesh_regularize.py --mesh four-room/scenes/single-gs-photo-mesh-six-v1/surface-owned.npz --source four-room/scenes/single-gs-depth-six-v1 --out four-room/scenes/single-gs-mesh-repaired-six-v2
python scene_fusion/textured_mesh.py --source four-room/scenes/single-gs-depth-six-v1 --geometry four-room/scenes/single-gs-mesh-repaired-six-v2/surface.npz --warped --footprint --texture-width 7680 --smoothing 0 --out four-room/scenes/single-gs-photo-mesh-six-v2
```

`--warped` here selects the newly captured single-model depth folder. It must
not point at the old independent-PLY warped depths. Model and camera centers
are checked for agreement at capture. Scene metadata retains original photo
checksums, verified again before texture baking. The checkpoint contains the
same Gaussian parameters exported in the accepted PLY.

Camera counts are read from metadata. Midpoint edges come from the calibrated
scene graph; no six-camera or object-coordinate special case is used. Current
checkpoint capture expects the distillation package's `data/dataset.json` next
to `optimized.npz` to verify the reference frame.

## Validation and preview

```powershell
# Open3D environment:
python tests/test_single_gs_mesh.py
# CPU environment:
python -m unittest discover -s tests -p test_textured_mesh.py -v
python -m unittest discover -s tests -p test_sh_io.py -v
# Task-owned Edge debugging session on port 19346, static server on 7870:
node tests/check_tour_distill_browser.mjs tour-photo-detail-six-v1
node tests/check_photo_mesh.mjs single-gs-photo-mesh-six-v2
```

The GS browser test confirms that only the candidate's one PLY is requested and
that degree-two SH is active. The mesh test renders 18 source directions, six
midpoints, geometry, and ownership. Automated checks are supplemented by image
inspection. Models, original photos, and screenshots remain excluded from Git.

- GS: `/scene-room.html?scene=tour-photo-detail-six-v1&inspect=3`
- Photo comparison: `/scene-report.html?scene=tour-photo-detail-six-v1`
- Mesh: `/mesh-photo-room.html?scene=single-gs-photo-mesh-six-v2`
- Portable mesh: `/four-room/scenes/single-gs-photo-mesh-six-v2/textured-room.zip`

## References

- [Open3D RGB-D integration](https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html): TSDF integration and extraction API used here.
- [2D Gaussian Splatting](https://surfsplatting.github.io/): surface-oriented representation, depth distortion and normal consistency address limitations of ordinary 3D Gaussian surface reconstruction. This experiment does **not** implement or claim to run a 2DGS retraining pipeline.
