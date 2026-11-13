# Static PLY clarity experiment (2026-11-13)

The scene viewer previously forced device pixel ratio 1. On a DPR 2 display,
an 800x600 CSS viewport was rendered at 800x600, then displayed over 1600x1200
physical pixels. The default now honors device pixel ratio. `quality=standard`
restores the lower-cost mode; `quality=hd` uses native resolution. Both links
preserve the scene and inspection query. This improves sampling on high-DPI
screens, not missing detail inside the model. DPR 1 screens gain no extra pixels.

`tour_refine.py --appearance-only` freezes positions, Gaussian scales and
rotations and disables splitting. Color/SH and opacity remain trainable. The
six-camera candidate `tour-appearance-six-v1` resumes `tour-distilled-six-v3`
for 1500 steps at 640, 1024 and 1536 with the same 49% spatial teacher. Its
1,367,420 geometry records were compared with the reference and are exactly
equal. This is synthetic original-PLY supervision, not high-resolution photo
retexturing and not an image sharpening filter.

At 1024, average anchor SSIM improved 0.978611 -> 0.982455, and held-out SSIM
0.954653 -> 0.957049. At 640 both average scores also improved. This is a modest
appearance improvement, not proof of photo-level fidelity. The strict all-anchor
gate still fails; some ceiling stains and fine-structure blur remain. Keep v3
available and publish the candidate as an explicitly labeled experiment.

Validation: geometry equality, export round-trip, 300 rendered comparisons at
each resolution, five surface-refinement unit tests, browser DPR 2 canvas-size
assertions, and seven single-PLY browser viewpoints. Numerical summaries are in
`experiments/2026-11-13-clarity`. The browser screenshots are local scene artifacts.
