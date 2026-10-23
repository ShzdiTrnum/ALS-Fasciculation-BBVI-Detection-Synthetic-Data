# Algorithm overview: `src/bbvi_synthetic.py`

What the code computes, in the order it runs. Section numbers are referenced from the README.

## 1. Frame pairs and rings

* Frames 1–24 are loaded as grayscale in `[0, 1]` and cropped to the ROI (x 240–740, y 124–624).
* Pairs are `(i, i + FRAME_STEP)`: 23 pairs at step 1, 21 pairs at step 3.
* Ring *k* holds the pixels whose distance from the **ROI centre** is in `(r_k − Δ/2, r_k + Δ/2]` with
  `r_k = 10, 30, …, 190` and `Δ = radii[0] = 10`. The rings are therefore 10 px wide with 10 px gaps between
  them (e.g. 15–25 px); pixels in the gaps, and beyond 195 px, are warped with θ = 0.

## 2. Observation model

```
f1(x) = f2( R(θ_k) (x − c) + c ) + n,   n ~ N(0, σ_n²),   x in ring k,   R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]]
```

`f1` = frame *i*, `f2` = frame *i + step*, warped with `grid_sample` (bilinear, border padding).

## 3. Posterior

* **Unstructured** (`STRUCTURED = False`): `q(θ) = Π_k N(μ_k, σ_k²)`, parameters `μ_k, log σ_k²`.
* **Structured** (`STRUCTURED = True`): `μ = Φ α`, `log σ² = Φ β`, where Φ (10 × U) is an orthonormal DCT-II
  cosine basis over the ring index (`build_cosine_basis`) and `U = U_BASIS = 5`. Only `α, β` are optimised,
  so `μ(r)` is smooth in r by construction.

Samples: `θ^(s) = μ + σ ⊙ ε^(s)`, `ε ~ N(0, I)`, then clipped to `±THETA_CLAMP`.

## 4. E-step

Adam on `(μ, log σ²)` or `(α, β)` minimising

```
loss_E = (1 / 2σ_n²) · (1/S) Σ_s || f1 − warp(f2, θ^(s), c) ||²
```

The KL term is set to zero (`kl = 0`), and there is no entropy term, so this is the expected log-likelihood
part of the ELBO only. The "ELBO" convergence plot shows this quantity.

## 5. M-step

* Adam on the centre `c` with the same expected squared error; after each step `c` is clipped to the box
  `ROI centre ± CENTER_BOX_PX` (scaled per level).
* `σ_n² = (1/S) Σ_s ||f1 − warp(f2, θ^(s), c)||² / (H·W)` after every outer iteration.

## 6. Pyramid, prior fusion, temporal propagation

* Levels 0.25 → 0.5 → 1.0. After each level the final mean is clipped to `±THETA_CLAMP`.
* The level's posterior and the previous-pair posterior are combined by a product of Gaussians and used to
  **initialise** the next level (for the structured arm, `α, β` are re-fitted to the fused curve by least squares).
* The finest-level posterior of pair *t* initialises every level of pair *t + 1*; the centre is reset to the
  ROI centre at the start of each pair.

## 7. Evaluation inside the script

* Ground truth per pair: `gt_rel = deg2rad(α(j) − α(i))` from `ring_angles.csv`.
* Sign: with `GT_SIGN = None`, the sign of the estimate is chosen per run as the one with the lower error on
  the pairs with the largest true motion. `src/evaluate_runs.py` re-scores with a fixed sign of −1.
* MAE over all rings, inner five rings (r = 10–90) and outer five rings (r = 110–190); RMSE; centre error
  = distance between the estimated centre and the true centre of the **source** frame, averaged over pairs.
* One summary row is appended to `RESULTS_CSV`; the per-pair arrays are saved to `PERPAIR_DIR/<arm>_step<k>.npz`
  (overwritten by the next run of the same arm and step).

## 8. Why the inner rings are hard in this video

`generate_synthetic_video.py` renders each frame by rotating every ring about the fixed point (490, 374) and
then translating the whole pattern so it is centred on the drifting point in `ring_center_positions.csv`,
which moves from (490, 374) to (499, 380) over frames 3–9 and back over frames 16–22, i.e. during the same
frames as the rotation. The recorded centre is therefore where the pattern is, not a fixed pivot of the
frame-to-frame motion, so the centre error measures distance to the pattern centre. Between two frames the
content therefore rotates *and* translates by about 1.5 px. For an inner ring (r = 10 px) a 0.04 rad rotation
moves pixels by only 0.4 px, so the translation dominates; a pure rotation about one centre cannot represent
it, and a per-ring grid search over θ on the raw frames gives the opposite sign for ring 0 than for the outer
rings. The outer rings (r ≥ 90 px), where rotation dominates, are consistent with `θ = −Δα`.

## 9. Differences from the manuscript description

| Manuscript | Code |
|---|---|
| ELBO with KL to the fused prior | KL term omitted (§4) |
| Fused Gaussian used as the prior | Used as the initialisation (§6) |
| Basis functions of the warped intensities (Eq. 17) | Cosine basis over the ring index (§3) |
| Radius measured from the estimated centre | Rings fixed around the ROI centre (§1) |
| Single reported MAE | Best of four unstructured runs; see `evaluate_runs.py` for mean ± sd |
