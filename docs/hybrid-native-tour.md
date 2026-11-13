# Native source fidelity with a narrow fused bridge

Open `native-tour-room.html?scene=native-source-six-v1&fused=tour-appearance-six-v1`.
Without `fused`, the existing native-only tour is preserved. The hybrid uses
the appearance-refined static PLY, not the rejected bilateral-color trial.

Source geometry remains byte-identical to the packaged original PLYs. The
fused viewer uses its own SH degree and the common world coordinate frame;
camera centers must match before playback is allowed. It renders into a
separate image layer rather than changing individual Gaussian opacity.

The default native domain is 49% along a two-camera route, leaving a narrow
49–51% transition. Existing 30/38/44% choices and a new 49% choice are available.
For original pair weights a,b, the fused bridge weight is 4ab; original weights
are multiplied by 1-4ab. At the exact midpoint the fused model has weight 1.
Near either capture position it has weight 0. Outside the transition or once
settled with the default clarity option, the fused layer is disabled immediately
instead of retaining a long temporal fade. Source model transitions still ease.

The spatial band is defined by relative distances to the nearest cameras, not
a constant time duration. Free movement far from the capture route can expose
source-model defects. Loading failures retain available source rendering. The
fused asset is preloaded before click-to-travel movement. The additional model
increases startup time and GPU memory.

Validation on 2026-11-13: unit checks with 2/4/6/9 cameras, normalized weights,
49% domain and unavailable-model fallback. Actual browser checks cover source 1
horizontal/up/down with 100% original weight, route positions 48%/50%/52%, rest,
and arriving at source 2. No runtime exceptions. See
`experiments/2026-11-13-hybrid-tour` and local scene `hybrid-browser` screenshots.

The current source-1 raw PLY also shows some floor blotches when rendered alone.
The corresponding source photo contains bright floor reflections. This indicates
at least some discoloration precedes fusion; it is not evidence that every
reported artifact has the same cause. The hybrid restores source-model playback,
not photographic fidelity or a repair of those existing artifacts.
