# Multi-view render-constrained refit

## Problem and decision

The accepted single PLY preserves room coverage but loses source-camera detail.
Directly rendering its textured mesh already reproduces jagged silhouettes and
incorrect gray/color fragments. Therefore Gaussian footprint filtering alone
cannot recover the missing constraints. This observation does not establish
whether geometry, texture ownership, or both cause every individual artifact.

The new experimental stage optimizes the **final Gaussian model against all four
calibrated input panoramas**. It does not change the player into a panorama overlay,
does not identify lamps or curtains, and does not concatenate four transparent
models. Output remains the existing 14-field, degree-zero Gaussian PLY format.

## Implementations reviewed

- [GaussReg fusion code](https://github.com/GAP-LAB-CUHK-SZ/GaussReg/blob/main/gs_fusion.py):
  after registration, uses distances to cloud centroids to choose retained points.
  Useful for ownership, but does not resolve inconsistent independent reconstructions.
- [MVS-Texturing view selection](https://github.com/nmoehrle/mvs-texturing/blob/master/libs/tex/view_selection.cpp):
  surface-adjacency graph with unary costs and Potts pairwise label costs. This
  assumes an adequate common surface; optimizing labels cannot make wrong geometry correct.
- [2DGS training](https://github.com/hbb1/2d-gaussian-splatting/blob/main/train.py):
  rendered photo losses plus distortion and depth/normal consistency. Its surface
  representation and training renderer are not a drop-in PLY merger.
- [PGSR training](https://github.com/zju3dv/PGSR/blob/main/train.py): photo loss,
  image-weighted depth/normal loss, and visibility-gated cross-camera reprojection
  and patch photometric losses. Sparse observations remain a limitation.
- [gsplat](https://github.com/nerfstudio-project/gsplat), Apache-2.0: selected as the
  differentiable rasterization dependency, with native Windows CUDA support.

The experiment uses independently written orchestration and losses. It does not
copy the noncommercial 2DGS/PGSR training code or claim to implement those methods.
In particular, explicit PGSR normal/patch consistency is not implemented here.

## Stages

1. `prepare_refit.py` freezes the accepted model and camera transforms. It samples
   the original panoramas into 24 cube faces at four physical camera centers.
   Two additional source directions and twelve midpoint directions are diagnostics.
2. `refit_probe.py` checks CUDA rendering and nonzero finite color gradients on an
   original Gaussian scene before any fitting.
3. `refit_train.py` runs a color-only ablation and a bounded position/scale/color
   optimization. Point count, opacity and rotations remain fixed. No point deletion,
   densification, semantic patches or view-dependent colors are used.
4. RGB, local structural and image-gradient losses constrain source renders.
   Coverage loss penalizes lost baseline opacity. Midpoint baseline renders provide
   a weak trust region, **not ground truth**. Position changes are bounded in model
   units; scale changes are limited to 0.8–1.25 of the starting scale.
   A geometry/color-aware nearest-neighbor graph regularizes local colors and
   relative displacement. Neighboring surface points adjust coherently instead of
   individually encoding conflicting projections as noise. Color differences and
   differing normals reduce smoothing across real boundaries.
5. `refit_evaluate.py` compares identical cameras, fixed baseline masks and all
   covered pixels, including newly missing pixels. Inspect images as well as scores.
6. Export a candidate only after comparing the actual exported PLY and player.
   Never automatically replace an accepted baseline based on training loss.

## Acceptance

- All outputs finite; PLY and browser load correctly.
- No new substantial holes or broken silhouettes at source or midpoint views.
- Mean photo SSIM improves, at least 90% of the source directions improve, and no
  source direction loses more than 0.01 SSIM.
- Lamps, curtains, table edges, TV and globe must be visually checked. Aggregate
  gains cannot override a visible regression on these structures.
- Photo scores are fitting diagnostics. The four photographs do not provide
  independent novel-view truth or observations behind occluders. Any remaining
  artifacts must be disclosed, and a failed candidate remains experimental.

## Iteration results

The first structural-loss trial failed because the installed Windows PyTorch
runtime's channels-last average-pooling backward did not agree with finite
differences. Explicit contiguous NCHW layout corrected the discrepancy. The
gradient check is retained as a regression test; this was an implementation fault,
not evidence against SSIM as an objective.

Color-only and independent geometry/color refits improved average photo error but
introduced fine surface noise and regressed several source directions. They were
rejected. A graph-regularized 512-pixel refit improved all 26 source checks, but
1024-pixel evaluation exposed coverage and detail regressions that were hidden at
low resolution. A second optimization stage at 1024 corrected these issues.

Final CUDA checks at 1024 pixels: mean source SSIM 0.868247 → 0.941436, RGB MAE
0.031542 → 0.017115, all 26 source directions improved (minimum SSIM gain 0.017146).
Across 38 directions, three pixels fell from baseline alpha >0.95 to candidate
alpha <0.5. At 512, SSIM 0.842331 → 0.916642, all 26 directions improved and no
such new holes were counted. These are fitting and coverage diagnostics, not
independent evidence of unseen geometry. Remaining table edges, chair/rod gaps,
globe texture irregularity and fine lamp-outline roughness must remain disclosed.

The high-detail candidate above is retained as a comparison. A final generic
displacement-field diffusion trial reduced inconsistent local motion without
averaging the original surface itself. Its graph uses normal, tangential distance
and cross-layer separation, **not RGB**, so textured curved surfaces also move
coherently. After freezing positions and refitting appearance/scales for 800 steps,
this surface-guard candidate was selected for the default preview. It trades some
texture fitting for more coherent motion: at 1024, SSIM 0.868247 → 0.918154,
MAE 0.031542 → 0.019672; all 26 directions improve, minimum SSIM gain 0.004962.
There are 249 newly low-alpha pixels across 38 views (not 249 physical holes).
At 512, SSIM 0.842331 → 0.901338, all 26 directions improve, five newly low-alpha
pixels. Both pass the numerical gate; neither is a perfect room reconstruction.
The default preview and its downloadable PLY refer to `refit-surface-guard`;
`?version=detail` selects `refit-coherent-hd`.

## Native Windows execution

Data preparation uses the existing CPU environment and `GS_INPUT_DIR`/
`GS_OUTPUT_DIR`. CUDA fitting uses a separate environment and `run-refit-gpu.ps1`.
The local test runtime reuses an already-installed PyTorch 2.7.0+cu128 without
modifying it, compiles gsplat 1.5.3 for the RTX 4090, and uses no WSL. Runtime paths
are parameters; private photos/models and local environments are not committed.

For a fresh isolated environment, install PyTorch 2.7.0 from the official cu128
index and then `requirements-refit.txt`. Run `fusion_next/fix_gsplat_windows.py` to
replace unsupported GCC flags in gsplat 1.5.3's Windows JIT setup. This retains an
original backup and changes compile flags only. Configure the installed Visual
Studio and CUDA paths through `run-refit-gpu.ps1` parameters. Use named
`-PythonArguments @('script.py', '--option', 'value')`, not positional arguments.

Final training recipe:

1. CPU environment: `prepare_refit.py --size 512` with `GS_OUTPUT_DIR` pointing at
   `four-room/refit-data`, then `refit_neighbors.py --data four-room/refit-data`.
2. GPU environment: `refit_gradient_check.py`, then `refit_train.py --data
   four-room/refit-data --out four-room/refit-coherent --mode joint --steps 2400
   --warmup 240 --graph-weight 1.5 --edge-weight 0.3 --color-lr 0.002`.
3. Prepare a separate `four-room/refit-data-hd` at size 1024. Reuse its identical
   initial model's neighbor graph only after confirming model identity.
4. GPU environment: `refit_train.py --data four-room/refit-data-hd --out
   four-room/refit-coherent-hd --mode joint --steps 1600 --warmup 0 --graph-weight
   1.5 --edge-weight 0.3 --color-lr 0.001 --position-lr 0.00005 --scale-lr 0.0003
   --resume four-room/refit-coherent/optimized.npy`.
5. Evaluate with `refit_evaluate.py` at **both resolutions**, export with
   `export_refit.py` in the CPU environment, and inspect the PLY in the real player.
6. Optional final surface guard: run `refit_smooth_motion.py --initial
   four-room/refit-data/initial.npy --candidate four-room/refit-coherent-hd/optimized.npy
   --out four-room/refit-surface-guard`. Resume training from its `motion-smoothed.npy`
   for 800 HD steps, warmup 0, graph weight 1.5, edge weight 0.3, color LR 0.0008,
   position LR 0 and scale LR 0.0002. Re-evaluate both resolutions before export.

The final single PLY is post-refit Gaussian geometry; the old mesh preview is not
updated to match it and must not be presented as the new result.
