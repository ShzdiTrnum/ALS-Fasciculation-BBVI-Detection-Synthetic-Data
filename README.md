# BBVI Rotation Estimation on Synthetic Fasciculation Video (Ground-Truth Validation)

Synthetic-data experiments for the structured variational inference extension of

> T. Shahzadi, J. Zhou, K. Bokuda, N. Tagawa. *Fasciculation Detection in Ultrasound Images for ALS
> Diagnosis Based on Black Box Variational Inference.* VISAPP 2026, vol. 3, pp. 688–695.
> doi:[10.5220/0014623000004084](https://doi.org/10.5220/0014623000004084)

A 24-frame synthetic video of ten textured rings that rotate to +8°, rest, and rotate back to 0°, with a
drifting rotation centre, is analysed with the black-box variational inference (BBVI) + variational EM
rotation estimator. Because the true ring angles and centre path are recorded, the estimates can be scored
directly. The real-ultrasound code is in
[ALS-Fasciculation-BBVI-Detection-Ultrasound](https://github.com/ShzdiTrnum/ALS-Fasciculation-BBVI-Detection-Ultrasound).

## Repository layout

```
src/
  generate_synthetic_video.py  makes the 24-frame video and its ground truth (angles + centre path)
  bbvi_synthetic.py          BBVI + EM estimator; unstructured or structured (cosine basis in r), step 1 or 3
  optical_flow_synthetic.py  optical-flow / Green's-theorem baseline on the same video
  evaluate_runs.py           summary over runs, zero-motion baseline, fixed-sign re-scoring
  runconfig.py               command-line overrides and figure saving (shared helper)
data/                        synthetic video and its ground truth (included)
results/                     the runs reported in the manuscript (CSV + per-pair .npz)
docs/algorithm_overview.md   what the code computes, step by step
```

## Quick start

```bash
pip install -r requirements.txt

# optional: regenerate the video (writes to outputs/regenerated, not data/)
python src/generate_synthetic_video.py --seed 0

# the four cells of the 2 x 2 experiment (each appends one row to outputs/structuring_results.csv)
python src/bbvi_synthetic.py --STRUCTURED False --FRAME_STEP 1
python src/bbvi_synthetic.py --STRUCTURED True  --FRAME_STEP 1
python src/bbvi_synthetic.py --STRUCTURED False --FRAME_STEP 3
python src/bbvi_synthetic.py --STRUCTURED True  --FRAME_STEP 3

python src/optical_flow_synthetic.py               # baseline
python src/evaluate_runs.py                        # summarise results/ (or pass --results-csv outputs/...)
```

Run `python src/bbvi_synthetic.py --help-config` to list every constant that can be set with `--NAME VALUE`.
`--seed N` fixes the random seed (the reported runs did not), `--show` opens the figures, and all
figures are saved to `outputs/`. A GPU is strongly recommended.

## Settings used for the reported runs

| Setting | Value |
|---|---|
| Video | `data/fasciculation_simulation_24frames.mp4`, 980 × 748, 24 frames |
| ROI | x 240–740, y 124–624 (500 × 500) |
| Rings | radii 10, 30, …, 190 px (10 rings) |
| Pyramid | 0.25, 0.5, 1.0 |
| MC samples / outer EM / E-steps / M-steps | 20 / 40 / 60 / 50 |
| E-step learning rate per level | 7e-4, 5e-4, 3e-4 |
| M-step learning rate per level | 2e-3, 1e-3, 5e-4 |
| Structured basis | DCT-II cosine in r, `U_BASIS = 5` |
| Initial angles | row 1 of `ring_initial_rotation_values.csv` (all zeros) |
| `THETA_CLAMP` | 0.0873 rad (±5°) for step 1; the step-3 run in `results/` reaches ±0.3 rad, so it was run with a wider bound |
| `CENTER_BOX_PX` | 30 px |
| `GT_SIGN` | `None` (chosen automatically per run, see below) |

### Settings that use ground-truth knowledge

Three settings were chosen with knowledge of the synthetic ground truth, and they should be read that way:

* **`THETA_CLAMP`** bounds every sample and the final estimate to ±0.0873 rad. The largest true inter-frame
  rotation at step 1 is 0.0654 rad, so the bound was set from the known motion range.
* **`CENTER_BOX_PX`** keeps the centre within ±30 px of the ROI centre; the true centre drifts by about 10 px.
* **`GT_SIGN = None`** picks, for each run, whichever sign gives the lower error on the high-motion pairs.
  The step-1 runs chose −1 and the step-3 run chose +1. `evaluate_runs.py` re-scores every run with one
  fixed sign (−1, which follows from the warp model and is what the outer rings show) so the runs are
  comparable.

## Results in `results/`

From `python src/evaluate_runs.py` (MAE of the per-pair relative rotation, radians):

| Arm | Step | Runs | MAE (mean ± sd, range) | Zero-motion baseline | Centre error (px) |
|---|---|---|---|---|---|
| unstructured | 1 | 4 | 0.0154 ± 0.0016 (0.0139–0.0172) | 0.0121 | 11.2 (6.1–16.9) |
| structured, U = 5 | 1 | 1 | 0.0171 | 0.0121 | 7.7 |
| unstructured | 3 | 1 | 0.0677 (0.0689 with fixed sign) | 0.0392 | 6.8 |

The **zero-motion baseline** is the error obtained by predicting no rotation for every ring and pair.
In these runs the BBVI error is higher than that baseline, so the synthetic experiment does not yet show
an accuracy gain over predicting no motion. Part of the reason is in the data: the generator moves the
rotation centre (about 1.5 px per frame) during exactly the frames where the rings rotate: each frame is
rendered by rotating the rings about the fixed point (490, 374) and then translating the whole pattern.
Between two frames the inner rings therefore mostly translate rather than rotate, which a pure rotation
about one centre cannot fit (see `docs/algorithm_overview.md`, §8).

## Plots

`bbvi_synthetic.py` smooths the plotted rotation curves with a 3-frame moving average (`SMOOTH_PLOT = 3`),
and smooths the estimated radial profile in the "high-motion pair" figure across radius, to match the
optical-flow plots. The saved numbers (`.npz`, CSV) and all error metrics use the unsmoothed estimates.

## Data

`data/` holds the synthetic video and its ground truth (synthetic, so it can be shared):

| File | Content |
|---|---|
| `fasciculation_simulation_24frames.mp4` | the video |
| `ring_angles.csv` | 24 × 10, absolute angle of each ring per frame, **degrees**, no header |
| `ring_initial_rotation_values.csv` | same values with a `Frame` column; row 1 (all zeros) is the initial value |
| `ring_center_positions.csv` | true rotation centre per frame, image pixels |

## Citation

```bibtex
@inproceedings{shahzadi2026bbvi,
  author    = {Shahzadi, Turrnum and Zhou, Junfeng and Bokuda, Kota and Tagawa, Norio},
  title     = {Fasciculation Detection in Ultrasound Images for {ALS} Diagnosis Based on Black Box Variational Inference},
  booktitle = {Proceedings of the 21st International Conference on Computer Vision Theory and Applications (VISAPP)},
  volume    = {3}, pages = {688--695}, year = {2026},
  doi       = {10.5220/0014623000004084}
}
```

## License

MIT (see `LICENSE`).
