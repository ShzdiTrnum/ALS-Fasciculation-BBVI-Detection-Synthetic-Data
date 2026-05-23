# Algorithm Overview — BBVI Synthetic Detection

## Purpose

This document explains the algorithm in `bbvi_detection_synthetic.py` and the rationale behind parameter choices specific to the synthetic dataset.

---

## Key Differences from the ALS Pipeline

### 1. Frame Step (`FRAME_STEP = 3`)

The synthetic video rotates rings by ±4° over 40 frames — roughly 0.2°/frame. Pairing frame `i` with frame `i+3` amplifies the apparent rotation to ~0.6°/frame, making it easier for the optimizer to find a meaningful gradient. For real ultrasound (ALS repo), consecutive frames (`FRAME_STEP=1`) suffice because motion is less predictable.

### 2. Ring Configuration

- **Synthetic**: 10 rings, radii 10–190 px (step 20). Matches the ring sizes used in the `optical_flow_synthetic.py` script.
- **ALS**: 20 rings, radii 5–100 px (step 5). Finer resolution for subtle real motion.

### 3. ROI Size

- **Synthetic**: 500×500 px (`240:740, 124:624`). Covers the full simulated disc region.
- **ALS**: 250×250 px (`250:500, 200:450`). Tighter crop around the muscle fascicle.

### 4. Initialization CSV Source

For synthetic data, `ring_initial_rotation_values.csv` comes **directly from the simulation** — these are the exact angles used to generate each frame. This means:
- The BBVI starts from the correct answer
- Convergence is faster and more stable
- Residual error after convergence measures the pipeline's precision

---

## BBVI + EM Algorithm

### Probabilistic Model

For a frame pair `(f1, f2)`:

```
f1(x) ≈ warp(f2, θ, c) + ε,    ε ~ N(0, σ_n²)
```

Where `warp(f2, θ, c)` rotates `f2` by angle `θ_r` around center `c` for each ring `r`.

### Variational Posterior

```
q(θ) = ∏_r N(θ_r | μ_r, σ²_r)
```

Parameters `{μ_r, log σ²_r}` for `r = 1..10` are optimized per frame pair.

### E-Step (Variational Update)

Maximize ELBO w.r.t. `{μ_r, log σ²_r}` using reparameterization:

```
θ^(s) = μ + σ · ε^(s),    ε^(s) ~ N(0, I)

ELBO ≈ -(1/S) Σ_s ||f1 - warp(f2, θ^(s), c)||² / (2σ_n²)
```

With `S_mc = 10` samples. The mean `μ` is frozen on the first frame pair (initialisation frame).

### M-Step (Center + Noise Update)

Minimize expected squared warping error w.r.t. center `c`:

```
min_c  E_q[||f1 - warp(f2, θ, c)||²]
```

`σ_n` is updated analytically:

```
σ_n² = E_q[||f1 - warp(f2, θ, c)||²] / (H × W)
```

Fixed after the first frame pair.

### Bayesian Prior Fusion (Coarse → Fine)

At each pyramid level transition, the prior for the next finer level is:

```
p(θ) ∝ N(θ | μ_coarse, σ²_coarse) · N(θ | μ_prev_frame, σ²_prev_frame)
     = N(θ | μ_fused, σ²_fused)
```

Precision addition:

```
1/σ²_fused = 1/σ²_coarse + 1/σ²_prev_frame
μ_fused    = σ²_fused · (μ_coarse/σ²_coarse + μ_prev_frame/σ²_prev_frame)
```

### Temporal Propagation

The finest-level posterior `(μ_fin, log σ²_fin)` from pair `t` seeds all pyramid levels of pair `t+1`.

---

## Warping Implementation

Rotation warping uses `torch.nn.functional.grid_sample` with bilinear interpolation:

```
x_src = cos(θ)·(x - cx) - sin(θ)·(y - cy) + cx
y_src = sin(θ)·(x - cx) + cos(θ)·(y - cy) + cy
```

Grid normalized to `[-1, 1]` as required by `grid_sample`.

---

## Zoomed Inset COR Plot

The centre-of-rotation (COR) trajectory uses `mpl_toolkits.axes_grid1.inset_locator`:

- `zoomed_inset_axes`: creates a magnified inset panel
- `mark_inset`: draws dashed connecting lines from the zoomed region to the inset
- Each point is annotated with its source frame index

This is particularly useful for synthetic data where the simulated COR drift is small (≈30 px) and hard to see on a full-ROI plot.

---

## Output Diagnostics

| Plot | What to check |
|------|--------------|
| Rotation angle vs frame | Should follow the known ±4° sweep pattern from the simulation |
| ELBO | Should increase within each level; plateau = convergence |
| MSE | Should decrease; final value reflects residual noise floor |
| Noise variance | Should stabilize after first pair |
| Posterior variance | Should decrease over pairs as the posterior sharpens |
| COR trajectory | Should stay near the simulated center drift path |
