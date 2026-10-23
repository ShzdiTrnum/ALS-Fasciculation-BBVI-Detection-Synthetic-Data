"""Summarise and re-score the synthetic BBVI runs against ground truth.

Inputs (defaults are the files shipped in this repository):
  results/structuring_results.csv   one row per run, written by bbvi_synthetic.py
  results/perpair_runs/*.npz        per-pair arrays of the last run of each {arm, step}
  data/ring_angles.csv              true ring angles per frame (degrees, 24 x 10)

What it reports
  1. For every {arm, U, frame_step}: number of runs and the mean, std, min and max of
     the angle MAE and centre error, so a single best run is not mistaken for the
     typical result.
  2. The zero-motion baseline: the MAE obtained by predicting no rotation for every
     ring and pair. An estimator is only useful if it beats this number.
  3. For each saved .npz: the MAE re-computed with the FIXED sign convention
     theta_gt = -(alpha(b) - alpha(a)), which follows from the warp model and was checked
     numerically, instead of the per-run automatic sign choice; the correlation with the
     ground truth; and the share of estimates sitting on the +/-THETA_CLAMP bound.

Usage
  python src/evaluate_runs.py
  python src/evaluate_runs.py --results-csv outputs/structuring_results.csv --perpair-dir outputs/perpair_runs
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def zero_baseline(gt_deg, step):
    d = np.deg2rad(gt_deg[step:] - gt_deg[:-step])
    half = gt_deg.shape[1] // 2
    return {"mae_all_rad": float(np.abs(d).mean()),
            "mae_inner_rad": float(np.abs(d[:, :half]).mean()),
            "mae_outer_rad": float(np.abs(d[:, half:]).mean()),
            "max_true_rotation_rad": float(np.abs(d).max())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-csv", default="results/structuring_results.csv")
    p.add_argument("--perpair-dir", default="results/perpair_runs")
    p.add_argument("--gt", default="data/ring_angles.csv")
    p.add_argument("--theta-clamp", type=float, default=0.0873)
    p.add_argument("--out", default="outputs/evaluation_summary.json")
    a = p.parse_args()

    gt_deg = np.loadtxt(a.gt, delimiter=",")
    report = {"zero_motion_baseline": {}, "runs_summary": [], "perpair_rescored": {}}

    df = pd.read_csv(a.results_csv)
    for step in sorted(df["frame_step"].unique()):
        report["zero_motion_baseline"][f"step{int(step)}"] = zero_baseline(gt_deg, int(step))

    print("=" * 78)
    print("Runs in", a.results_csv)
    print("=" * 78)
    for (arm, U, step), g in df.groupby(["arm", "U", "frame_step"]):
        zb = report["zero_motion_baseline"][f"step{int(step)}"]["mae_all_rad"]
        s = {"arm": arm, "U": int(U), "frame_step": int(step), "n_runs": int(len(g)),
             "mae_all_mean": float(g.mae_all_rad.mean()), "mae_all_std": float(g.mae_all_rad.std(ddof=1)) if len(g) > 1 else None,
             "mae_all_min": float(g.mae_all_rad.min()), "mae_all_max": float(g.mae_all_rad.max()),
             "center_err_mean": float(g.center_err_px.mean()),
             "center_err_min": float(g.center_err_px.min()), "center_err_max": float(g.center_err_px.max()),
             "gt_signs_used": sorted(set(float(x) for x in g.gt_sign)),
             "zero_motion_mae": zb}
        report["runs_summary"].append(s)
        std = f" ± {s['mae_all_std']:.4f}" if s["mae_all_std"] is not None else ""
        print(f"{arm:<12} U={int(U)} step={int(step)}  runs={len(g)}  "
              f"MAE {s['mae_all_mean']:.4f}{std} rad (min {s['mae_all_min']:.4f}, max {s['mae_all_max']:.4f})  "
              f"| zero-motion {zb:.4f}  | centre err {s['center_err_mean']:.2f} px "
              f"(range {s['center_err_min']:.2f}-{s['center_err_max']:.2f})  | signs {s['gt_signs_used']}")

    print("\n" + "=" * 78)
    print("Per-pair files re-scored with the fixed sign convention")
    print("=" * 78)
    for f in sorted(glob.glob(os.path.join(a.perpair_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        sign = float(z["sign"])
        est_raw = z["est_signed"] / sign            # undo the per-run automatic sign
        gt_rel = z["gt_rel"]                        # alpha(b) - alpha(a), radians
        est = -est_raw                              # fixed convention
        err = np.abs(est - gt_rel)
        on_bound = np.isclose(np.abs(est_raw), a.theta_clamp, atol=2e-4)
        half = int(z["half"])
        r = {"auto_sign_used": sign,
             "mae_auto_sign": float(np.abs(z["est_signed"] - gt_rel).mean()),
             "mae_fixed_sign": float(err.mean()),
             "mae_inner_fixed": float(err[:, :half].mean()),
             "mae_outer_fixed": float(err[:, half:].mean()),
             "zero_motion_mae": float(np.abs(gt_rel).mean()),
             "corr_with_gt_fixed_sign": float(np.corrcoef(est.ravel(), gt_rel.ravel())[0, 1]),
             "share_on_theta_clamp": float(on_bound.mean()),
             "per_ring_mae_fixed": err.mean(0).round(5).tolist(),
             "center_err_mean_px": float(z["center_err"].mean())}
        report["perpair_rescored"][os.path.basename(f)] = r
        print(f"{os.path.basename(f):<26} MAE fixed-sign {r['mae_fixed_sign']:.4f} "
              f"(auto-sign {r['mae_auto_sign']:.4f}, sign {sign:+.0f})  zero-motion {r['zero_motion_mae']:.4f}  "
              f"corr {r['corr_with_gt_fixed_sign']:+.2f}  on clamp {100*r['share_on_theta_clamp']:.1f}%")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
