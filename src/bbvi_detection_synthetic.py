import os
import math
import copy
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from tqdm import trange

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 14,
    'figure.titlesize': 16
})

import torch
import torch.nn.functional as F
from torch import nn

# -----------------------------
# User-specified paths & params
# -----------------------------
video_path      = "/US/Synthetic/fasciculation_simulation_accel_decel_test_new4.mp4"
csv_init_angles = "/US/Synthetic/ring_initial_rotation_values.csv"
FRAME_START = 1
FRAME_END   = 40
FRAME_STEP  = 3          # process pairs (frame i, frame i+FRAME_STEP)

# ROI
xmin, xmax = 240, 740
ymin, ymax = 124, 624

# Radii: 10-pixel intervals up to 190 (10 rings)
radii     = np.arange(10, 200, 20)
num_radii = len(radii)

# Pyramid scales (coarse -> fine)
pyramid_scales = [0.25, 0.5, 1.0]
num_levels     = len(pyramid_scales)

# BBVI / EM hyperparams
S_mc           = 10
OUTER_EM_ITERS = 50
E_UPDATES      = 40
M_UPDATES      = 30

# Learning rates per pyramid level
Elr_levels = [0.0001, 0.0001, 0.0001]
Mlr_levels = [0.0001, 0.0001, 0.0001]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)


# -----------------------------
# Utility functions
# -----------------------------

def load_video_frames(video_path, frame_start, frame_end):
    """
    Load frames from video_path using OpenCV.
    frame_start, frame_end are 1-based inclusive indices.
    Returns list of grayscale float32 images (H x W) normalized to [0,1].
    """
    assert os.path.exists(video_path), f"Video not found: {video_path}"
    cap    = cv2.VideoCapture(video_path)
    frames = []
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx in range(frame_start - 1, frame_end):
        if idx < 0 or idx >= total:
            raise IndexError(f"Requested frame {idx+1} out of bounds (total {total})")
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {idx+1} from {video_path}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        frames.append(gray)
    cap.release()
    return frames


def crop_roi(img, xmin, xmax, ymin, ymax):
    return img[ymin:ymax, xmin:xmax]


def build_image_pyramid(img, scales):
    """
    img: numpy array HxW (grayscale)
    scales: list of scale factors (e.g., [0.25, 0.5, 1.0])
    Returns list of torch tensors (1,1,H,W) on device.
    """
    pyram = []
    for s in scales:
        if s == 1.0:
            img_s = img
        else:
            h_new = max(1, int(round(img.shape[0] * s)))
            w_new = max(1, int(round(img.shape[1] * s)))
            img_s = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(img_s).unsqueeze(0).unsqueeze(0).to(device)
        pyram.append(t)
    return pyram


def make_radial_masks(roi_h, roi_w, center, radii):
    """
    Build annular boolean masks for each radius in radii.
    Each mask selects pixels with distance from center in (r - step/2, r + step/2].
    Returns list of boolean masks shaped (H, W).
    """
    yy = np.arange(roi_h) - center[1]
    xx = np.arange(roi_w) - center[0]
    X, Y = np.meshgrid(xx, yy)
    R    = np.sqrt(X**2 + Y**2)
    masks = []
    step  = radii[0]
    for r in radii:
        r_min = max(0.0, r - step / 2)
        r_max = r + step / 2
        masks.append((R > r_min) & (R <= r_max))
    return masks


def rotation_grid_from_theta(theta_map, center, H, W):
    """
    Build a sampling grid for F.grid_sample that rotates each pixel
    by the angle given in theta_map around center.
    Returns grid of shape (1, H, W, 2) normalized to [-1, 1].
    """
    ys = torch.arange(0, H, device=theta_map.device, dtype=theta_map.dtype)
    xs = torch.arange(0, W, device=theta_map.device, dtype=theta_map.dtype)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')
    Xr = X - center[0];  Yr = Y - center[1]
    ct = torch.cos(theta_map);  st = torch.sin(theta_map)
    Xw = ct * Xr - st * Yr + center[0]
    Yw = st * Xr + ct * Yr + center[1]
    xn = 2.0 * (Xw / (float(W) - 1) - 0.5)
    yn = 2.0 * (Yw / (float(H) - 1) - 0.5)
    return torch.stack((xn, yn), dim=-1).unsqueeze(0)


def mse_np(a, b):
    return np.mean((a - b) ** 2)


# -----------------------------
# Bayesian Prior Fusion
# -----------------------------

def fuse_gaussian_priors(mu1, logvar1, mu2, logvar2):
    """
    Fuse two independent Gaussian posteriors via product-of-Gaussians.

    Given:
        q1(theta) = N(mu1, sigma1^2)   <- posterior from previous frame
        q2(theta) = N(mu2, sigma2^2)   <- posterior from coarser pyramid level

    The fused distribution is:
        p(theta) ∝ q1 * q2 = N(mu_fused, sigma_fused^2)

    Precision addition:
        prec_fused = prec1 + prec2
        mu_fused   = (mu1*prec1 + mu2*prec2) / prec_fused

    Args:
        mu1, logvar1      : (num_radii,) mean and log-variance from source 1
        mu2, logvar2      : (num_radii,) mean and log-variance from source 2

    Returns:
        mu_fused, logvar_fused : (num_radii,) fused parameters
    """
    var1 = torch.exp(logvar1);  var2 = torch.exp(logvar2)
    prec1 = 1.0 / (var1 + 1e-8);  prec2 = 1.0 / (var2 + 1e-8)
    prec_fused   = prec1 + prec2
    var_fused    = 1.0 / (prec_fused + 1e-8)
    mu_fused     = (mu1 * prec1 + mu2 * prec2) / (prec_fused + 1e-8)
    logvar_fused = torch.log(var_fused + 1e-8)
    return mu_fused, logvar_fused


# -----------------------------
# Core Model
# -----------------------------

class IndependentRotEstimator:
    """
    Independent rotation estimator: each annular ring has its own
    rotation angle parameter — no shared basis functions.
    """
    def __init__(self, roi_h, roi_w, radii, device):
        self.roi_h     = roi_h
        self.roi_w     = roi_w
        self.radii     = radii
        self.num_radii = len(radii)
        self.device    = device

    def theta_from_params(self, theta_params):
        return theta_params

    def theta_map_from_theta_r(self, theta_r, radial_masks):
        """
        Assign each annular mask region its rotation value to build
        a full-resolution (H, W) theta map.
        """
        H, W      = radial_masks[0].shape
        theta_map = torch.zeros((H, W), dtype=torch.float32, device=self.device)
        for i, mask in enumerate(radial_masks):
            if np.any(mask):
                theta_map[torch.from_numpy(mask).to(self.device)] = theta_r[i]
        return theta_map


# -----------------------------
# Main processing
# -----------------------------

def main():
    print("\n" + "=" * 60)
    print("BBVI INDEPENDENT ROTATION — SYNTHETIC DATA")
    print("Each radius has its own independent rotation parameter")
    print("Bayesian prior fusion: product-of-Gaussians from")
    print("  (1) previous frame posterior")
    print("  (2) coarser pyramid level posterior")
    print(f"Frame step = {FRAME_STEP}  (pairs: i -> i+{FRAME_STEP})")
    print("=" * 60 + "\n")

    # Load frames
    frames      = load_video_frames(video_path, FRAME_START, FRAME_END)
    num_frames  = len(frames)
    frame_pairs = [(i, i + FRAME_STEP) for i in range(num_frames - FRAME_STEP)]
    num_pairs   = len(frame_pairs)
    print(f"Frames loaded : {num_frames}")
    print(f"Frame pairs   : {num_pairs}  (step={FRAME_STEP})\n")

    # ROI and center
    roi0         = crop_roi(frames[0], xmin, xmax, ymin, ymax)
    roi_h, roi_w = roi0.shape
    roi_center   = (int(round((xmax - xmin) / 2.0)),
                    int(round((ymax - ymin) / 2.0)))

    radial_masks = make_radial_masks(roi_h, roi_w, roi_center, radii)
    estimator    = IndependentRotEstimator(roi_h, roi_w, radii, device)

    # Load initial rotation CSV (from Optical-Flow or Fasciculation-Simulation)
    if not os.path.exists(csv_init_angles):
        raise FileNotFoundError(f"CSV not found: {csv_init_angles}")

    init_df = pd.read_csv(csv_init_angles)
    if init_df.shape[1] >= num_radii + 1:
        init_angles_arr = init_df.iloc[0, 1:num_radii + 1].astype(float).values.astype(np.float32)
    else:
        possible_vals = pd.to_numeric(init_df.iloc[:, 1], errors='coerce')
        if possible_vals.notna().sum() >= num_radii:
            init_angles_arr = possible_vals.dropna().values[:num_radii].astype(np.float32)
        else:
            raise ValueError("CSV must contain at least num_radii numeric angle columns.")

    mu_theta_init = torch.from_numpy(init_angles_arr).to(device)

    # Storage
    all_mu_theta_r_per_pair     = np.zeros((num_pairs, num_radii), dtype=np.float32)
    all_logvar_theta_r_per_pair = np.zeros((num_pairs, num_radii), dtype=np.float32)
    centers_traj                = np.zeros((num_pairs, 2),         dtype=np.float32)
    noise_variances_by_level    = [np.zeros(OUTER_EM_ITERS, dtype=np.float32) for _ in range(num_levels)]
    mse_by_level                = [np.zeros(OUTER_EM_ITERS, dtype=np.float32) for _ in range(num_levels)]
    elbo_by_level               = [np.zeros(OUTER_EM_ITERS, dtype=np.float32) for _ in range(num_levels)]
    sigma_n_fixed_levels        = [None] * num_levels

    # ----------------------------------------------------------------
    # Main loop over frame pairs
    # ----------------------------------------------------------------
    for pidx, (i_f, j_f) in enumerate(frame_pairs):
        print(f"\nPair {pidx+1}/{num_pairs}: frame {FRAME_START+i_f} -> {FRAME_START+j_f}  (gap={j_f-i_f})")

        f1_full = crop_roi(frames[i_f], xmin, xmax, ymin, ymax)
        f2_full = crop_roi(frames[j_f], xmin, xmax, ymin, ymax)
        pyr_f1  = build_image_pyramid(f1_full, pyramid_scales)
        pyr_f2  = build_image_pyramid(f2_full, pyramid_scales)

        # Initialise per-level parameters
        per_level_params = []
        for level_idx, scale in enumerate(pyramid_scales):
            mu_theta     = mu_theta_init.clone().float()
            logvar_theta = torch.log(torch.ones_like(mu_theta) * (1.0 ** 2))
            c_x          = roi_center[0] * scale
            c_y          = roi_center[1] * scale
            per_level_params.append({
                "mu_theta":     mu_theta.detach().clone().requires_grad_(True),
                "logvar_theta": logvar_theta.detach().clone().requires_grad_(True),
                "sigma_n":      torch.tensor(0.02, dtype=torch.float32, device=device, requires_grad=True),
                "center":       torch.tensor([c_x, c_y], dtype=torch.float32, device=device, requires_grad=True)
            })

        # Seed all levels from previous frame posterior (pair > 0)
        if pidx > 0:
            prev_mu     = torch.from_numpy(all_mu_theta_r_per_pair[pidx-1, :]).to(device)
            prev_logvar = torch.from_numpy(all_logvar_theta_r_per_pair[pidx-1, :]).to(device)
            for lvl in range(num_levels):
                per_level_params[lvl]["mu_theta"]     = prev_mu.clone().detach().requires_grad_(True)
                per_level_params[lvl]["logvar_theta"]  = prev_logvar.clone().detach().requires_grad_(True)
                c_x = roi_center[0] * pyramid_scales[lvl]
                c_y = roi_center[1] * pyramid_scales[lvl]
                per_level_params[lvl]["center"]  = torch.tensor([c_x, c_y], dtype=torch.float32, device=device, requires_grad=True)
                per_level_params[lvl]["sigma_n"] = torch.tensor(0.02, dtype=torch.float32, device=device, requires_grad=True)

        # Coarse -> fine pyramid loop
        for level_idx, scale in enumerate(pyramid_scales):
            print(f"  Level {level_idx}  scale={scale}")

            f1_t = pyr_f1[level_idx].to(device)
            f2_t = pyr_f2[level_idx].to(device)
            Hlvl = f1_t.shape[2];  Wlvl = f1_t.shape[3]

            # Scale masks to current pyramid level
            if scale != 1.0:
                scaled_masks = []
                for mask in radial_masks:
                    m_rs = cv2.resize((mask.astype(np.uint8)*255), (Wlvl, Hlvl), interpolation=cv2.INTER_NEAREST)
                    scaled_masks.append(m_rs.astype(bool))
            else:
                scaled_masks = radial_masks

            params       = per_level_params[level_idx]
            mu_theta     = params["mu_theta"].requires_grad_(True)
            logvar_theta = params["logvar_theta"].requires_grad_(True)
            sigma_n      = params["sigma_n"].requires_grad_(True)
            center_param = params["center"].requires_grad_(True)
            params.update({"mu_theta": mu_theta, "logvar_theta": logvar_theta,
                            "sigma_n": sigma_n,  "center": center_param})

            freeze_mu = (pidx == 0)

            optimizerE = torch.optim.Adam([mu_theta, logvar_theta], lr=Elr_levels[level_idx])
            optimizerM = torch.optim.Adam([center_param, sigma_n],  lr=Mlr_levels[level_idx])

            if sigma_n_fixed_levels[level_idx] is not None and pidx > 0:
                sigma_n           = torch.tensor(float(sigma_n_fixed_levels[level_idx]),
                                                  dtype=torch.float32, device=device, requires_grad=False)
                params["sigma_n"] = sigma_n
                optimizerM        = torch.optim.Adam([center_param], lr=Mlr_levels[level_idx])

            # Outer EM loop
            for outer in range(OUTER_EM_ITERS):

                # E-step
                for _ in range(E_UPDATES):
                    optimizerE.zero_grad()
                    std_theta     = torch.exp(0.5 * logvar_theta)
                    eps           = torch.randn((S_mc, mu_theta.shape[0]), device=device, dtype=mu_theta.dtype)
                    theta_samples = (mu_theta.detach() if freeze_mu else mu_theta).unsqueeze(0) + std_theta.unsqueeze(0) * eps
                    losses = []
                    for s in range(S_mc):
                        theta_map = estimator.theta_map_from_theta_r(theta_samples[s], scaled_masks)
                        grid      = rotation_grid_from_theta(theta_map, (center_param[0], center_param[1]), Hlvl, Wlvl)
                        f2_warp   = F.grid_sample(f2_t, grid, mode='bilinear', padding_mode='border', align_corners=True)
                        losses.append(((f1_t - f2_warp) ** 2).sum())
                    mc_se  = torch.stack(losses).mean()
                    loss   = 0.5 * mc_se / (params["sigma_n"] ** 2)
                    loss.backward()
                    if freeze_mu and mu_theta.grad is not None:
                        mu_theta.grad.zero_()
                    optimizerE.step()

                # M-step
                for _ in range(M_UPDATES):
                    optimizerM.zero_grad()
                    std_theta     = torch.exp(0.5 * logvar_theta)
                    eps           = torch.randn((S_mc, mu_theta.shape[0]), device=device)
                    theta_samples = mu_theta.unsqueeze(0) + std_theta.unsqueeze(0) * eps
                    se_list = []
                    for s in range(S_mc):
                        theta_map = estimator.theta_map_from_theta_r(theta_samples[s], scaled_masks)
                        grid      = rotation_grid_from_theta(theta_map, (center_param[0], center_param[1]), Hlvl, Wlvl)
                        f2_warp   = F.grid_sample(f2_t, grid, mode='bilinear', padding_mode='border', align_corners=True)
                        se_list.append(((f1_t - f2_warp) ** 2).sum())
                    mc_se = torch.stack(se_list).mean()
                    mc_se.backward()
                    optimizerM.step()

                # Analytical sigma_n update
                with torch.no_grad():
                    params["sigma_n"] = torch.sqrt(mc_se / (Hlvl * Wlvl))

                # Logging
                with torch.no_grad():
                    std_theta     = torch.exp(0.5 * logvar_theta)
                    eps           = torch.randn((S_mc, mu_theta.shape[0]), device=device, dtype=mu_theta.dtype)
                    theta_samples = mu_theta.unsqueeze(0) + std_theta.unsqueeze(0) * eps
                    losses_log = []
                    for s in range(S_mc):
                        theta_map = estimator.theta_map_from_theta_r(theta_samples[s], scaled_masks)
                        grid      = rotation_grid_from_theta(theta_map, (center_param[0], center_param[1]), Hlvl, Wlvl)
                        f2_warp   = F.grid_sample(f2_t, grid, mode='bilinear', padding_mode='border', align_corners=True)
                        losses_log.append(((f1_t - f2_warp) ** 2).sum())
                    mc_se_log   = float(torch.stack(losses_log).mean().cpu())
                    mse_val     = mc_se_log / (Hlvl * Wlvl)
                    sigma_n_val = params["sigma_n"].item() if isinstance(params["sigma_n"], torch.Tensor) else float(params["sigma_n"])
                    elbo_val    = -0.5 * mc_se_log / (sigma_n_val ** 2)
                    noise_variances_by_level[level_idx][outer] = sigma_n_val
                    mse_by_level[level_idx][outer]             = mse_val
                    elbo_by_level[level_idx][outer]            = elbo_val

                if (outer + 1) % 5 == 0:
                    print(f"    outer {outer+1}/{OUTER_EM_ITERS}: sigma_n={sigma_n_val:.6f}, mse={mse_val:.6e}, elbo={elbo_val:.4f}")

            # Store level results
            per_level_params[level_idx].update({
                "mu_theta":     mu_theta.detach(),
                "logvar_theta": logvar_theta.detach(),
                "sigma_n":      params["sigma_n"].detach() if isinstance(params["sigma_n"], torch.Tensor) else params["sigma_n"],
                "center":       center_param.detach()
            })
            if pidx == 0:
                sigma_n_fixed_levels[level_idx] = float(per_level_params[level_idx]["sigma_n"].cpu())

            # Coarse -> fine Bayesian fusion
            if level_idx < num_levels - 1:
                mu_fused, logvar_fused = fuse_gaussian_priors(
                    per_level_params[level_idx]["mu_theta"],
                    per_level_params[level_idx]["logvar_theta"],
                    per_level_params[level_idx+1]["mu_theta"].detach(),
                    per_level_params[level_idx+1]["logvar_theta"].detach()
                )
                scale_ratio = pyramid_scales[level_idx+1] / pyramid_scales[level_idx]
                cx = per_level_params[level_idx]["center"][0].item() * scale_ratio
                cy = per_level_params[level_idx]["center"][1].item() * scale_ratio
                per_level_params[level_idx+1].update({
                    "mu_theta":     mu_fused.detach().requires_grad_(True),
                    "logvar_theta": logvar_fused.detach().requires_grad_(True),
                    "center":       torch.tensor([cx, cy], dtype=torch.float32, device=device, requires_grad=True),
                    "sigma_n":      per_level_params[level_idx]["sigma_n"]
                })

        # Record finest-level results
        all_mu_theta_r_per_pair[pidx, :]     = per_level_params[-1]["mu_theta"].cpu().numpy()
        all_logvar_theta_r_per_pair[pidx, :] = per_level_params[-1]["logvar_theta"].cpu().numpy()
        centers_traj[pidx, :]                = per_level_params[-1]["center"].cpu().numpy()
        print(f"  Done pair {pidx+1}: center={centers_traj[pidx,:]}, first 5 theta={all_mu_theta_r_per_pair[pidx,:5]}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    frame_pair_indices = np.array([FRAME_START + i_f for (i_f, _) in frame_pairs])

    # 1. Rotation angle per radius
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    for ridx in range(num_radii):
        ax1.plot(frame_pair_indices, all_mu_theta_r_per_pair[:, ridx],
                 marker='o', linewidth=2, markersize=6, label=f"r={int(radii[ridx])}")
    ax1.set_xlabel("Source frame index")
    ax1.set_ylabel("Rotation angle (radians)")
    ax1.set_title(f"Independent Rotation per Radius — Synthetic Data\n(Bayesian Prior Fusion, step={FRAME_STEP})")
    ax1.axhline(0, linestyle='--', linewidth=1)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    plt.tight_layout()
    plt.show()

    # 2. Convergence metrics
    for metric_name, metric_data in [
        ("Noise Variance", noise_variances_by_level),
        ("MSE",            mse_by_level),
        ("ELBO",           elbo_by_level)
    ]:
        fig = plt.figure(figsize=(14, 7))
        ax_main = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
        for lvl, color in enumerate(['C0', 'C1', 'C2']):
            ax_main.plot(np.arange(1, OUTER_EM_ITERS+1), metric_data[lvl], color=color, label=f"Level {lvl}")
        ax_main.set_xlabel("Outer EM iteration");  ax_main.set_ylabel(metric_name)
        ax_main.set_title(f"{metric_name} across pyramid levels");  ax_main.legend()
        for lvl, color in enumerate(['C0', 'C1', 'C2']):
            ax = plt.subplot2grid((3, 2), (lvl, 1))
            ax.plot(np.arange(1, OUTER_EM_ITERS+1), metric_data[lvl], color=color)
            ax.set_title(f"Level {lvl}");  ax.set_xlabel("Iteration")
        plt.tight_layout();  plt.show()

    # 3. Center trajectory with zoom inset
    centers = centers_traj.copy()
    fig_traj, ax_traj = plt.subplots(figsize=(9, 9))
    sc = ax_traj.scatter(centers[:, 0], centers[:, 1], c=frame_pair_indices, cmap='viridis', s=60, zorder=3)
    ax_traj.plot(centers[:, 0], centers[:, 1], color='black', linewidth=1.5, zorder=2)
    for i, (xi, yi) in enumerate(zip(centers[:, 0], centers[:, 1])):
        ax_traj.annotate(str(frame_pair_indices[i]), xy=(xi, yi), fontsize=9,
                         ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
    ax_traj.add_patch(plt.Rectangle((0, 0), roi_w, roi_h,
                                     fill=False, edgecolor='red', linestyle='--', linewidth=2))
    ax_traj.set_title(f"Adaptive COR Trajectory — Synthetic Data\n(step={FRAME_STEP})")
    ax_traj.set_xlabel("X Coordinate");  ax_traj.set_ylabel("Y Coordinate")
    ax_traj.invert_yaxis();  ax_traj.set_aspect('equal', adjustable='box')
    plt.colorbar(sc, ax=ax_traj, label="Frame Pair")

    from matplotlib.lines import Line2D
    ax_traj.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Estimated Centers'),
        Line2D([0], [0], color='black', linewidth=1.5, label='Trajectory'),
    ], loc='upper left', fontsize=11)

    margin  = 10
    x_min_z = centers[:, 0].min() - margin;  x_max_z = centers[:, 0].max() + margin
    y_min_z = centers[:, 1].min() - margin;  y_max_z = centers[:, 1].max() + margin
    axins   = zoomed_inset_axes(ax_traj, zoom=2.5, loc='lower left',
                                 bbox_to_anchor=(0.02, 0.02), bbox_transform=ax_traj.transAxes, borderpad=1)
    axins.scatter(centers[:, 0], centers[:, 1], c=frame_pair_indices, cmap='viridis', s=40, zorder=3)
    axins.plot(centers[:, 0], centers[:, 1], color='black', linewidth=1.2, zorder=2)
    for i, (xi, yi) in enumerate(zip(centers[:, 0], centers[:, 1])):
        axins.annotate(str(frame_pair_indices[i]), xy=(xi, yi), fontsize=7,
                       ha='center', va='bottom', xytext=(0, 4), textcoords='offset points')
    axins.set_xlim(x_min_z, x_max_z);  axins.set_ylim(y_max_z, y_min_z)
    axins.set_aspect('equal', adjustable='box');  axins.tick_params(labelsize=8)
    mark_inset(ax_traj, axins, loc1=2, loc2=4, fc="none", ec="gray", lw=1.2, linestyle='--')
    plt.tight_layout();  plt.show()

    # 4. Posterior variance over time
    fig_var, ax_var = plt.subplots(figsize=(14, 6))
    for ridx in range(0, num_radii, 2):
        ax_var.plot(frame_pair_indices, np.exp(all_logvar_theta_r_per_pair[:, ridx]),
                    marker='s', linewidth=1.5, markersize=4, label=f"r={int(radii[ridx])}")
    ax_var.set_xlabel("Frame index");  ax_var.set_ylabel("Posterior variance (rad²)")
    ax_var.set_title("Posterior Variance per Radius — Synthetic Data\n(should decrease with Bayesian fusion)")
    ax_var.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax_var.tick_params(axis='x', labelsize=18);  ax_var.tick_params(axis='y', labelsize=18)
    plt.tight_layout();  plt.show()


if __name__ == "__main__":
    main()
