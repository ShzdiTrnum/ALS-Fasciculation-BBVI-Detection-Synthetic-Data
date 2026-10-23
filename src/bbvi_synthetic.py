import os
import math
import copy
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
#from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from tqdm import trange
from collections import OrderedDict


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

# =====================================================================
# EXPERIMENT SWITCHES  (comment 1a + comment 2)
# ---------------------------------------------------------------------
#   STRUCTURED = False -> independent / unstructured arm (each radius a
#                         free parameter; this is your original model)
#   STRUCTURED = True  -> structured-in-r arm (per-radius mean/logvar are
#                         built from a low-order cosine basis in r)
#
#   FRAME_STEP = 1     -> consecutive frames
#   FRAME_STEP = 3     -> skipped / distant frames (large-motion stress)
#
# Run all four combinations:  {False, True} x {1, 3}
# Each run appends one row of ground-truth error numbers to RESULTS_CSV,
# so the 2x2 table assembles itself.
# =====================================================================
STRUCTURED = False        # unstructured: best-looking earlier result (9/10 rings clean)
U_BASIS    = 5            # number of cosine basis terms (structured arm). U << num_radii

# -----------------------------
# User-specified paths & params
# -----------------------------
video_path      = "data/fasciculation_simulation_24frames.mp4"
csv_init_angles = "data/ring_initial_rotation_values.csv"
# >>> FILL THIS IN: ground-truth per-frame angles for the 24-frame video
#     (shape 24 x 10, degrees, inner ring = column 0), produced by your generator.
csv_gt_angles   = "data/ring_angles.csv"

# Where per-run error summaries are appended (one row per run)
RESULTS_CSV     = "outputs/structuring_results.csv"

# Directory where each run dumps its FULL per-pair arrays (for the separate
# per-frame comparison script). One .npz per {arm, frame_step}.
PERPAIR_DIR     = "outputs/perpair_runs"

FRAME_START = 1
FRAME_END = 24
FRAME_STEP = 1          # skip frames between pairs (i -> i+FRAME_STEP)

# ---- Ground-truth generator geometry (must match the generator script) ----
# The true drifting rotation centre is NOT a formula: the generator records it
# per frame in ring_center_positions.csv (header: Frame,center_x,center_y ;
# 1-indexed ; ORIGINAL image coordinates, e.g. starts at 490,374). It is a
# STEPWISE drift (holds ~6 frames, then jumps a few px). We READ it rather than
# recompute it -- exact and assumption-free.
GEN_IMG_W, GEN_IMG_H     = 980, 748
# >>> FILL THIS IN: recorded true centre path for THIS 24-frame video.
csv_gt_centers   = "data/ring_center_positions.csv"

# ROI
xmin, xmax = 240, 740
ymin, ymax = 124, 624

# Radii: 20-pixel intervals -> 10 rings at [10,30,...,190]
radii = np.arange(10, 200, 20)
num_radii = len(radii)

# Pyramid scales (coarse -> fine)
pyramid_scales = [0.25, 0.5, 1.0]  # Level 0: coarsest, Level 2: finest
num_levels = len(pyramid_scales)

# BBVI / EM hyperparams
S_mc = 20           # MC samples
OUTER_EM_ITERS = 40
E_UPDATES = 60
M_UPDATES = 50


#S_mc = 5           # MC samples
#OUTER_EM_ITERS = 5
#E_UPDATES = 3
#M_UPDATES = 10

# Learning rates per level (coarser -> finer)
Elr_levels = [0.0007, 0.0005, 0.0003]
Mlr_levels = [0.002, 0.001, 0.0005]

# Sign convention for comparing estimated angle vs ground-truth angle.
# OpenCV getRotationMatrix2D (generator) and this estimator's grid may differ
# in sign. Set to None to AUTO-DETECT from the first pair (recommended);
# or force to +1.0 / -1.0 once you know it.
GT_SIGN = None

# Bounds used in the reported runs (see README, 'Settings that use ground-truth knowledge').
THETA_CLAMP   = 0.0873   # rad; samples and final mean are clipped to +/- this value
CENTER_BOX_PX = 30.0     # px; centre is kept within +/- this box around the ROI centre

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

# --- added for the public release: command-line overrides + figure saving ---
from runconfig import configure
configure(globals())


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
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
    pyram = []
    for s in scales:
        if s == 1.0:
            img_s = img
        else:
            h_new = max(1, int(round(img.shape[0] * s)))
            w_new = max(1, int(round(img.shape[1] * s)))
            img_s = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(img_s).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
        pyram.append(t)
    return pyram


def make_radial_masks(roi_h, roi_w, center, radii):
    yy = np.arange(roi_h) - center[1]
    xx = np.arange(roi_w) - center[0]
    X, Y = np.meshgrid(xx, yy)
    R = np.sqrt(X**2 + Y**2)
    masks = []
    step = radii[0]
    for i, r in enumerate(radii):
        r_min = r - step / 2
        r_max = r + step / 2
        if r_min < 0:
            r_min = 0.0
        mask = (R > r_min) & (R <= r_max)
        masks.append(mask)
    return masks


def rotation_grid_from_theta(theta_map, center, H, W):
    ys = torch.arange(0, H, device=theta_map.device, dtype=theta_map.dtype)
    xs = torch.arange(0, W, device=theta_map.device, dtype=theta_map.dtype)
    Y, X = torch.meshgrid(ys, xs, indexing='ij')
    Xr = X - center[0]
    Yr = Y - center[1]
    ct = torch.cos(theta_map)
    st = torch.sin(theta_map)
    Xw = ct * Xr - st * Yr + center[0]
    Yw = st * Xr + ct * Yr + center[1]
    Hf = float(H)
    Wf = float(W)
    xn = 2.0 * (Xw / (Wf - 1) - 0.5)
    yn = 2.0 * (Yw / (Hf - 1) - 0.5)
    grid = torch.stack((xn, yn), dim=-1).unsqueeze(0)  # (1,H,W,2)
    return grid


def mse_np(a, b):
    return np.mean((a - b) ** 2)


# -----------------------------
# Structured basis in r  (comment 1a, option A)
# -----------------------------

def build_cosine_basis(radii, U):
    """
    Build an orthonormal-ish low-order cosine basis Phi of shape (num_radii, U),
    evaluated at the ring radii. Column 0 is the constant (flat) term; each
    successive column adds one more half-cosine 'wiggle' along r.

        phi_0(r) = 1/sqrt(R)                      (constant)
        phi_u(r) = sqrt(2/R) * cos(pi*u*(n+0.5)/R),  u = 1..U-1

    where n indexes the ring (0..R-1). This is the standard DCT-II basis, so
    the columns are mutually orthonormal -> mu(r)=Phi@alpha is smooth in r by
    construction, and the coefficients are well-conditioned.

    Returns Phi as a torch tensor on `device`, shape (num_radii, U).
    """
    R = len(radii)
    n = np.arange(R)
    Phi = np.zeros((R, U), dtype=np.float32)
    Phi[:, 0] = 1.0 / np.sqrt(R)
    for u in range(1, U):
        Phi[:, u] = np.sqrt(2.0 / R) * np.cos(np.pi * u * (n + 0.5) / R)
    return torch.from_numpy(Phi).to(device)


def fit_coeffs_to_angles(Phi, angles):
    """
    Least-squares fit of basis coefficients to a target per-radius vector.
    Solves  Phi @ c ~= angles  ->  c = pinv(Phi) @ angles.
    Phi: (R, U) tensor; angles: (R,) tensor. Returns (U,) tensor.
    """
    Phi_np    = Phi.detach().cpu().numpy()
    ang_np    = angles.detach().cpu().numpy()
    c, *_     = np.linalg.lstsq(Phi_np, ang_np, rcond=None)
    return torch.from_numpy(c.astype(np.float32)).to(device)


# -----------------------------
# Bayesian Prior Fusion
# -----------------------------

def fuse_gaussian_priors(mu1, logvar1, mu2, logvar2):
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)

    prec1 = 1.0 / (var1 + 1e-8)
    prec2 = 1.0 / (var2 + 1e-8)

    prec_fused = prec1 + prec2

    var_fused    = 1.0 / (prec_fused + 1e-8)
    mu_fused     = (mu1 * prec1 + mu2 * prec2) / (prec_fused + 1e-8)
    logvar_fused = torch.log(var_fused + 1e-8)

    return mu_fused, logvar_fused


# -----------------------------
# Ground-truth helpers
# -----------------------------

def load_gt_angles(path, n_frames_needed, n_rings):
    """
    Load ground-truth per-frame ring angles (generator output ring_angles.csv).
    Expected shape (>= n_frames_needed, n_rings), in DEGREES, inner ring = col 0.
    Returns np.float32 array (n_frames_needed, n_rings) in RADIANS.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Ground-truth angles CSV not found at {path}. "
            f"Fill in `csv_gt_angles` with the ring_angles.csv for THIS 24-frame video."
        )
    gt = np.loadtxt(path, delimiter=',')
    if gt.ndim == 1:
        gt = gt.reshape(-1, 1)
    if gt.shape[1] != n_rings:
        raise ValueError(
            f"ring_angles.csv has {gt.shape[1]} columns but estimator uses {n_rings} rings. "
            f"They must match one-to-one (inner->outer)."
        )
    if gt.shape[0] < n_frames_needed:
        raise ValueError(
            f"ring_angles.csv has only {gt.shape[0]} rows but {n_frames_needed} frames "
            f"are analysed."
        )
    gt = gt[:n_frames_needed, :].astype(np.float32)
    # Sanity: generator stores DEGREES (values commonly up to ~90). If it looks
    # like radians (tiny magnitudes), warn instead of silently mis-scaling.
    if np.nanmax(np.abs(gt)) < 6.5:
        print("  [GT WARNING] ring_angles.csv max |value| < 6.5 -- looks like it "
              "may already be in radians. Expected DEGREES. Check units.")
    return np.deg2rad(gt)  # -> radians


def load_gt_centers(path, n_frames_needed):
    """
    Load the generator's RECORDED true centre path (ring_center_positions.csv).
    Format: header 'Frame,center_x,center_y', 1-indexed, ORIGINAL image coords.
    Returns np.float32 array (n_frames_needed, 2) in original-image pixels,
    aligned so row k == frame k (0-based).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"True-centre CSV not found at {path}. Fill in `csv_gt_centers` with "
            f"the ring_center_positions.csv for THIS 24-frame video."
        )
    df = pd.read_csv(path)   # has a header row
    cols = [c.lower() for c in df.columns]
    if "center_x" in cols and "center_y" in cols:
        cx = df[df.columns[cols.index("center_x")]].values.astype(np.float32)
        cy = df[df.columns[cols.index("center_y")]].values.astype(np.float32)
    else:
        # fallback: assume columns [Frame, x, y]
        cx = df.iloc[:, 1].values.astype(np.float32)
        cy = df.iloc[:, 2].values.astype(np.float32)
    centers = np.stack([cx, cy], axis=1)  # (T, 2), original-image coords
    if centers.shape[0] < n_frames_needed:
        raise ValueError(
            f"ring_center_positions.csv has {centers.shape[0]} rows but "
            f"{n_frames_needed} frames are analysed."
        )
    return centers[:n_frames_needed, :]


def centers_to_roi(centers_full):
    """
    Convert recorded centres (original-image coords) to the estimator's cropped
    ROI coordinates by subtracting the crop origin (xmin, ymin). Directly
    comparable to the estimator's `center` parameter at full resolution.
    """
    out = centers_full.copy()
    out[:, 0] -= xmin
    out[:, 1] -= ymin
    return out


# -----------------------------
# Core Models
# -----------------------------

class IndependentRotEstimator:
    """
    Unstructured: each radius has its own rotation parameter. No basis.
    (theta_map builder is shared by both arms.)
    """
    def __init__(self, roi_h, roi_w, radii, device):
        self.roi_h     = roi_h
        self.roi_w     = roi_w
        self.radii     = radii
        self.num_radii = len(radii)
        self.device    = device

    def theta_map_from_theta_r(self, theta_r, radial_masks):
        H, W = radial_masks[0].shape
        theta_map = torch.zeros((H, W), dtype=torch.float32, device=self.device)
        for i, mask in enumerate(radial_masks):
            if np.any(mask):
                mask_t = torch.from_numpy(mask).to(self.device)
                theta_map[mask_t] = theta_r[i]
        return theta_map


# -----------------------------
# Main processing
# -----------------------------

def main():
    arm_name = "STRUCTURED (cosine basis in r)" if STRUCTURED else "UNSTRUCTURED (independent per radius)"
    print("\n" + "=" * 66)
    print(f"RUNNING {arm_name}")
    if STRUCTURED:
        print(f"  Basis: DCT-II cosine in r,  U = {U_BASIS}  ({U_BASIS} coeffs << {num_radii} rings)")
    print("Bayesian prior fusion: product-of-Gaussians from")
    print("  (1) previous frame posterior")
    print("  (2) coarser pyramid level posterior")
    print(f"Frame sampling: FRAME_STEP={FRAME_STEP} (pairs: i -> i+{FRAME_STEP})")
    print("=" * 66 + "\n")

    # ------------------------------------------------------------------
    # Load frames
    # ------------------------------------------------------------------
    frames     = load_video_frames(video_path, FRAME_START, FRAME_END)
    num_frames = len(frames)

    frame_pairs = [(i, i + FRAME_STEP)
                   for i in range(num_frames - FRAME_STEP)]
    num_pairs   = len(frame_pairs)

    print(f"Total frames loaded : {num_frames}")
    print(f"Total frame pairs   : {num_pairs} (step={FRAME_STEP})\n")

    # ------------------------------------------------------------------
    # ROI dimensions and center
    # ------------------------------------------------------------------
    frame0 = frames[0]
    roi0   = crop_roi(frame0, xmin, xmax, ymin, ymax)
    roi_h, roi_w = roi0.shape
    roi_center_x = int(round((xmax - xmin) / 2.0))
    roi_center_y = int(round((ymax - ymin) / 2.0))
    roi_center   = (roi_center_x, roi_center_y)

    radial_masks = make_radial_masks(roi_h, roi_w, roi_center, radii)
    estimator = IndependentRotEstimator(roi_h, roi_w, radii, device)

    # ------------------------------------------------------------------
    # Structured basis (built once)
    # ------------------------------------------------------------------
    Phi = build_cosine_basis(radii, U_BASIS) if STRUCTURED else None  # (num_radii, U)

    # ------------------------------------------------------------------
    # Load initial rotation angles CSV (SEED ONLY)
    # ------------------------------------------------------------------
    if not os.path.exists(csv_init_angles):
        raise FileNotFoundError(f"Initial angles CSV not found at {csv_init_angles}")

    init_df = pd.read_csv(csv_init_angles)
    if init_df.shape[1] >= num_radii + 1:
        init_angles_arr = init_df.iloc[0, 1:num_radii + 1].astype(float).values.astype(np.float32)
    else:
        possible_vals = pd.to_numeric(init_df.iloc[:, 1], errors='coerce')
        if possible_vals.notna().sum() >= num_radii:
            init_angles_arr = possible_vals.dropna().values[:num_radii].astype(np.float32)
        else:
            raise ValueError("Initial angles CSV must contain at least num_radii numeric values.")

    mu_theta_init = torch.from_numpy(init_angles_arr).to(device)  # (num_radii,) radians

    # Structured: initial coefficients by least-squares fit to the seed angles.
    if STRUCTURED:
        alpha_init = fit_coeffs_to_angles(Phi, mu_theta_init)                 # (U,)
        # logvar seed: constant log(1)=0 across radii -> only the flat coeff is nonzero.
        logvar_seed_vec = torch.zeros_like(mu_theta_init)                     # log(1)=0
        beta_init  = fit_coeffs_to_angles(Phi, logvar_seed_vec)               # (U,)

    # ------------------------------------------------------------------
    # Ground truth (radians), aligned to loaded frames (rows 0..num_frames-1)
    # ------------------------------------------------------------------
    gt_angles_rad = load_gt_angles(csv_gt_angles, num_frames, num_radii)  # (num_frames, num_radii)

    # ------------------------------------------------------------------
    # Storage for results
    # ------------------------------------------------------------------
    all_mu_theta_r_per_pair      = np.zeros((num_pairs, num_radii), dtype=np.float32)
    all_logvar_theta_r_per_pair  = np.zeros((num_pairs, num_radii), dtype=np.float32)
    centers_traj                 = np.zeros((num_pairs, 2),         dtype=np.float32)
    noise_variances_by_level     = [np.zeros(OUTER_EM_ITERS, dtype=np.float32) for _ in range(num_levels)]
    mse_by_level                 = [np.zeros(OUTER_EM_ITERS, dtype=np.float32) for _ in range(num_levels)]
    elbo_by_level                = [np.zeros(OUTER_EM_ITERS, dtype=np.float32) for _ in range(num_levels)]

    sigma_n_fixed_levels = [None] * num_levels

    # ------------------------------------------------------------------
    # Iterate over frame pairs
    # ------------------------------------------------------------------
    for pidx, (i_f, j_f) in enumerate(frame_pairs):
        print(f"\nProcessing pair {pidx + 1}/{num_pairs}: "
              f"frame {FRAME_START + i_f} -> {FRAME_START + j_f} (gap={j_f - i_f})")

        f1_full = crop_roi(frames[i_f], xmin, xmax, ymin, ymax)
        f2_full = crop_roi(frames[j_f], xmin, xmax, ymin, ymax)

        pyr_f1 = build_image_pyramid(f1_full, pyramid_scales)
        pyr_f2 = build_image_pyramid(f2_full, pyramid_scales)

        # ----------------------------------------------------------------
        # Initialise per-level parameters
        #  - unstructured: leaf tensors mu_theta / logvar_theta
        #  - structured:   leaf tensors alpha / beta; mu_theta/logvar derived
        # ----------------------------------------------------------------
        per_level_params = []

        for level_idx, scale in enumerate(pyramid_scales):
            sigma_n      = torch.tensor(0.02, dtype=torch.float32, device=device, requires_grad=True)
            c_x          = roi_center[0] * scale
            c_y          = roi_center[1] * scale
            center_param = torch.tensor([c_x, c_y], dtype=torch.float32, device=device, requires_grad=True)

            entry = {"sigma_n": sigma_n, "center": center_param}

            if STRUCTURED:
                entry["alpha"] = alpha_init.detach().clone().requires_grad_(True)
                entry["beta"]  = beta_init.detach().clone().requires_grad_(True)
            else:
                mu_theta     = mu_theta_init.clone().float()
                logvar_theta = torch.log(torch.ones_like(mu_theta) * (1.0 ** 2))
                entry["mu_theta"]     = mu_theta.detach().clone().requires_grad_(True)
                entry["logvar_theta"] = logvar_theta.detach().clone().requires_grad_(True)

            per_level_params.append(entry)

        # ----------------------------------------------------------------
        # For pair > 0, seed ALL levels with previous frame posterior
        #  - unstructured: seed mu/logvar arrays directly
        #  - structured:   re-fit coefficients to the previous posterior curve
        # ----------------------------------------------------------------
        if pidx > 0:
            prev_mu_theta     = torch.from_numpy(all_mu_theta_r_per_pair[pidx - 1, :]).to(device)
            prev_logvar_theta = torch.from_numpy(all_logvar_theta_r_per_pair[pidx - 1, :]).to(device)

            if STRUCTURED:
                prev_alpha = fit_coeffs_to_angles(Phi, prev_mu_theta)
                prev_beta  = fit_coeffs_to_angles(Phi, prev_logvar_theta)

            for level_idx in range(num_levels):
                if STRUCTURED:
                    per_level_params[level_idx]["alpha"] = prev_alpha.clone().detach().requires_grad_(True)
                    per_level_params[level_idx]["beta"]  = prev_beta.clone().detach().requires_grad_(True)
                else:
                    per_level_params[level_idx]["mu_theta"]     = prev_mu_theta.clone().detach().requires_grad_(True)
                    per_level_params[level_idx]["logvar_theta"] = prev_logvar_theta.clone().detach().requires_grad_(True)

                c_x = roi_center[0] * pyramid_scales[level_idx]
                c_y = roi_center[1] * pyramid_scales[level_idx]
                per_level_params[level_idx]["center"] = torch.tensor(
                    [c_x, c_y], dtype=torch.float32, device=device, requires_grad=True
                )
                per_level_params[level_idx]["sigma_n"] = torch.tensor(
                    0.02, dtype=torch.float32, device=device, requires_grad=True
                )

        # ----------------------------------------------------------------
        # Iterate over pyramid levels coarse -> fine
        # ----------------------------------------------------------------
        for level_idx, scale in enumerate(pyramid_scales):
            print(f"  Level {level_idx} scale {scale}")

            f1_t = pyr_f1[level_idx].to(device)
            f2_t = pyr_f2[level_idx].to(device)
            Hlvl = f1_t.shape[2]
            Wlvl = f1_t.shape[3]

            if scale != 1.0:
                scaled_masks = []
                for mask in radial_masks:
                    mask_uint8 = (mask.astype(np.uint8) * 255)
                    mask_rs    = cv2.resize(mask_uint8, (Wlvl, Hlvl), interpolation=cv2.INTER_NEAREST)
                    scaled_masks.append(mask_rs.astype(bool))
            else:
                scaled_masks = radial_masks

            params = per_level_params[level_idx]
            freeze_mu_for_first_pair = False

            # --- Set up leaf tensors + E-optimiser depending on arm ---
            if STRUCTURED:
                alpha = params["alpha"].requires_grad_(True)
                beta  = params["beta"].requires_grad_(True)
                params["alpha"] = alpha
                params["beta"]  = beta
                e_param_list    = [alpha, beta]
            else:
                mu_theta     = params["mu_theta"].requires_grad_(True)
                logvar_theta = params["logvar_theta"].requires_grad_(True)
                params["mu_theta"]     = mu_theta
                params["logvar_theta"] = logvar_theta
                e_param_list           = [mu_theta, logvar_theta]

            sigma_n      = params["sigma_n"].requires_grad_(True)
            center_param = params["center"].requires_grad_(True)
            params["sigma_n"] = sigma_n
            params["center"]  = center_param

            optimizerE = torch.optim.Adam(e_param_list, lr=Elr_levels[level_idx])
            optimizerM = torch.optim.Adam([center_param, sigma_n], lr=Mlr_levels[level_idx])

            if (sigma_n_fixed_levels[level_idx] is not None) and (pidx > 0):
                fixed_sigma_val    = float(sigma_n_fixed_levels[level_idx])
                sigma_n            = torch.tensor(fixed_sigma_val, dtype=torch.float32,
                                                  device=device, requires_grad=False)
                params["sigma_n"]  = sigma_n
                optimizerM         = torch.optim.Adam([center_param], lr=Mlr_levels[level_idx])

            # helper: build (mu_theta, logvar_theta) for the current arm
            def current_mu_logvar():
                if STRUCTURED:
                    return (Phi @ params["alpha"]), (Phi @ params["beta"])
                else:
                    return params["mu_theta"], params["logvar_theta"]

            # ------------------------------------------------------------
            # Outer EM loop
            # ------------------------------------------------------------
            for outer in range(OUTER_EM_ITERS):

                # --------------------------- E-step ---------------------------
                for e_it in range(E_UPDATES):
                    optimizerE.zero_grad()

                    mu_theta_cur, logvar_theta_cur = current_mu_logvar()
                    std_theta     = torch.exp(0.5 * logvar_theta_cur)
                    eps           = torch.randn((S_mc, num_radii), device=device, dtype=torch.float32)
                    theta_samples = mu_theta_cur.unsqueeze(0) + std_theta.unsqueeze(0) * eps
                    theta_samples = theta_samples.clamp(-THETA_CLAMP, THETA_CLAMP)  # clamp to +/-5 deg (=0.0873 rad): matches true max motion, kills inner-ring overshoot

                    if freeze_mu_for_first_pair:
                        # keep mean fixed at seed on first pair; only variance/coeffs of variance learn
                        mu_theta_det  = mu_theta_cur.detach()
                        theta_samples = mu_theta_det.unsqueeze(0) + std_theta.unsqueeze(0) * eps

                    losses = []
                    for s in range(S_mc):
                        theta_r   = theta_samples[s]
                        theta_map = estimator.theta_map_from_theta_r(theta_r, scaled_masks)
                        grid      = rotation_grid_from_theta(
                            theta_map, (center_param[0], center_param[1]), Hlvl, Wlvl)
                        f2_warp   = F.grid_sample(f2_t, grid, mode='bilinear',
                                                   padding_mode='border', align_corners=True)
                        res       = f1_t - f2_warp
                        losses.append((res ** 2).sum())

                    mc_se         = torch.stack(losses).mean()
                    sigma_n_param = params["sigma_n"]
                    loglik        = -0.5 * mc_se / (sigma_n_param ** 2)
                    kl            = torch.tensor(0.0, device=device)
                    elbo          = loglik - kl
                    loss          = -elbo
                    loss.backward()

                    if freeze_mu_for_first_pair:
                        # zero the gradient of whatever drives the MEAN
                        if STRUCTURED:
                            if params["alpha"].grad is not None:
                                params["alpha"].grad.zero_()
                        else:
                            if params["mu_theta"].grad is not None:
                                params["mu_theta"].grad.zero_()

                    optimizerE.step()

                # --------------------------- M-step ---------------------------
                for m_it in range(M_UPDATES):
                    optimizerM.zero_grad()

                    mu_theta_cur, logvar_theta_cur = current_mu_logvar()
                    std_theta     = torch.exp(0.5 * logvar_theta_cur)
                    eps           = torch.randn((S_mc, num_radii), device=device)
                    theta_samples = mu_theta_cur.unsqueeze(0) + std_theta.unsqueeze(0) * eps
                    theta_samples = theta_samples.clamp(-THETA_CLAMP, THETA_CLAMP)  # clamp to +/-5 deg (=0.0873 rad): matches true max motion, kills inner-ring overshoot

                    se_list = []
                    for s in range(S_mc):
                        theta_r   = theta_samples[s]
                        theta_map = estimator.theta_map_from_theta_r(theta_r, scaled_masks)
                        grid      = rotation_grid_from_theta(
                            theta_map, (center_param[0], center_param[1]), Hlvl, Wlvl)
                        f2_warp   = F.grid_sample(f2_t, grid, mode='bilinear',
                                                   padding_mode='border', align_corners=True)
                        res       = f1_t - f2_warp
                        se_list.append((res ** 2).sum())

                    mc_se       = torch.stack(se_list).mean()
                    loss_center = mc_se
                    loss_center.backward()
                    optimizerM.step()

                    # Safeguard: keep the estimated centre within a sane region.
                    # The true centre drifts only ~9px around the ROI centre, so a
                    # +/-30px window (scaled per level) cannot exclude the truth but
                    # prevents rare runaway pairs (e.g. flying to (328,152)).
                    with torch.no_grad():
                        cx0 = roi_center[0] * scale
                        cy0 = roi_center[1] * scale
                        margin = CENTER_BOX_PX * scale
                        center_param[0].clamp_(cx0 - margin, cx0 + margin)
                        center_param[1].clamp_(cy0 - margin, cy0 + margin)
                with torch.no_grad():
                    num_pixels        = Hlvl * Wlvl
                    sigma_n2          = mc_se / num_pixels
                    params["sigma_n"] = torch.sqrt(sigma_n2)

                # --------------------------- Logging ---------------------------
                with torch.no_grad():
                    mu_theta_cur, logvar_theta_cur = current_mu_logvar()
                    std_theta     = torch.exp(0.5 * logvar_theta_cur)
                    eps           = torch.randn((S_mc, num_radii), device=device, dtype=torch.float32)
                    theta_samples = mu_theta_cur.unsqueeze(0) + std_theta.unsqueeze(0) * eps
                    theta_samples = theta_samples.clamp(-THETA_CLAMP, THETA_CLAMP)  # clamp to +/-5 deg (=0.0873 rad): matches true max motion, kills inner-ring overshoot

                    losses_log = []
                    for s in range(S_mc):
                        theta_r   = theta_samples[s]
                        theta_map = estimator.theta_map_from_theta_r(theta_r, scaled_masks)
                        grid      = rotation_grid_from_theta(
                            theta_map, (center_param[0], center_param[1]), Hlvl, Wlvl)
                        f2_warp   = F.grid_sample(f2_t, grid, mode='bilinear',
                                                   padding_mode='border', align_corners=True)
                        res       = f1_t - f2_warp
                        losses_log.append((res ** 2).sum())

                    mc_se_log  = float(torch.stack(losses_log).mean().cpu().numpy())
                    num_pixels = float(Hlvl * Wlvl)
                    mse_val    = mc_se_log / num_pixels

                    if isinstance(params["sigma_n"], torch.Tensor) and not params["sigma_n"].requires_grad:
                        sigma_n_val = params["sigma_n"].item()
                    else:
                        sigma_n_val = float(sigma_n.detach().cpu().numpy())

                    loglik_val = -0.5 * mc_se_log / (sigma_n_val ** 2)
                    elbo_val   = loglik_val

                    noise_variances_by_level[level_idx][outer] = sigma_n_val
                    mse_by_level[level_idx][outer]             = mse_val
                    elbo_by_level[level_idx][outer]            = elbo_val

                if (outer + 1) % 5 == 0:
                    print(f"    outer {outer+1}/{OUTER_EM_ITERS}: "
                          f"sigma_n={sigma_n_val:.6f}, mse={mse_val:.6e}, elbo={elbo_val:.4f}")

            # End outer EM loop

            # Materialise final mu/logvar for this level (both arms)
            with torch.no_grad():
                if STRUCTURED:
                    mu_final     = (Phi @ params["alpha"]).detach()
                    logvar_final = (Phi @ params["beta"]).detach()
                else:
                    mu_final     = params["mu_theta"].detach()
                    logvar_final = params["logvar_theta"].detach()
                # Apply the +/-5 deg (0.0873 rad) bound CONSISTENTLY to the final
                # per-radius mean as well (not just the MC samples), so the stored
                # and plotted estimate never exceeds the physical limit.
                mu_final = mu_final.clamp(-THETA_CLAMP, THETA_CLAMP)

            per_level_params[level_idx]["mu_theta"]     = mu_final
            per_level_params[level_idx]["logvar_theta"] = logvar_final
            per_level_params[level_idx]["sigma_n"]      = (sigma_n.detach()
                                                            if isinstance(sigma_n, torch.Tensor)
                                                            else sigma_n)
            per_level_params[level_idx]["center"]       = center_param.detach()

            if pidx == 0:
                sigma_n_fixed_levels[level_idx] = float(
                    per_level_params[level_idx]["sigma_n"].cpu().numpy())

            # ----------------------------------------------------------
            # Coarse -> fine propagation using Bayesian prior fusion
            # (operates on materialised mu/logvar; for structured arm the
            #  next level re-fits coefficients to the fused curve above.)
            # ----------------------------------------------------------
            if level_idx < num_levels - 1:
                mu_coarse     = per_level_params[level_idx]["mu_theta"]
                logvar_coarse = per_level_params[level_idx]["logvar_theta"]

                mu_prev_frame     = per_level_params[level_idx + 1]["mu_theta"].detach() \
                                    if "mu_theta" in per_level_params[level_idx + 1] else mu_coarse
                logvar_prev_frame = per_level_params[level_idx + 1]["logvar_theta"].detach() \
                                    if "logvar_theta" in per_level_params[level_idx + 1] else logvar_coarse

                mu_fused, logvar_fused = fuse_gaussian_priors(
                    mu_coarse, logvar_coarse, mu_prev_frame, logvar_prev_frame)

                # Seed next level.
                if STRUCTURED:
                    nxt_alpha = fit_coeffs_to_angles(Phi, mu_fused)
                    nxt_beta  = fit_coeffs_to_angles(Phi, logvar_fused)
                    per_level_params[level_idx + 1]["alpha"] = nxt_alpha.detach().requires_grad_(True)
                    per_level_params[level_idx + 1]["beta"]  = nxt_beta.detach().requires_grad_(True)
                else:
                    per_level_params[level_idx + 1]["mu_theta"]     = mu_fused.detach().requires_grad_(True)
                    per_level_params[level_idx + 1]["logvar_theta"] = logvar_fused.detach().requires_grad_(True)

                scale_ratio = pyramid_scales[level_idx + 1] / pyramid_scales[level_idx]
                cx = per_level_params[level_idx]["center"][0].cpu().item() * scale_ratio
                cy = per_level_params[level_idx]["center"][1].cpu().item() * scale_ratio
                per_level_params[level_idx + 1]["center"] = torch.tensor(
                    [cx, cy], dtype=torch.float32, device=device, requires_grad=True)
                per_level_params[level_idx + 1]["sigma_n"] = per_level_params[level_idx]["sigma_n"]

        # ----------------------------------------------------------------
        # After all levels: record finest-level results
        # ----------------------------------------------------------------
        mu_theta_fin     = per_level_params[-1]["mu_theta"].cpu().numpy()
        logvar_theta_fin = per_level_params[-1]["logvar_theta"].cpu().numpy()

        all_mu_theta_r_per_pair[pidx, :]     = mu_theta_fin.copy()
        all_logvar_theta_r_per_pair[pidx, :] = logvar_theta_fin.copy()
        centers_traj[pidx, :]                = per_level_params[-1]["center"].cpu().numpy()

        print(f"  Finished pair {pidx + 1}: center={centers_traj[pidx, :]}, "
              f"first 5 theta={mu_theta_fin[:5]}")

    # ==================================================================
    # GROUND-TRUTH EVALUATION  (comment 1a / comment 2)
    # ==================================================================
    # True relative angle for a pair (i_f -> j_f), per ring:
    #     gt_rel[r] = gt_angles_rad[j_f, r] - gt_angles_rad[i_f, r]
    # Estimated relative angle is all_mu_theta_r_per_pair[pidx].
    # A global sign convention (GT_SIGN) may be needed.
    gt_rel = np.zeros((num_pairs, num_radii), dtype=np.float32)
    for pidx, (i_f, j_f) in enumerate(frame_pairs):
        gt_rel[pidx, :] = gt_angles_rad[j_f, :] - gt_angles_rad[i_f, :]

    est = all_mu_theta_r_per_pair.copy()

    # --- auto sign detection, ROBUST to the up->plateau->down motion pattern ---
    # The true motion is +ve (CCW phase), ~0 (plateau), then -ve (CW phase).
    # Averaging |error| over ALL pairs can wash the sign out, so we decide the
    # sign using only the pairs with the LARGEST true motion (where sign is
    # unambiguous), then apply it globally.
    if GT_SIGN is None:
        motion_mag = np.abs(gt_rel).mean(axis=1)              # per-pair true motion
        thr        = max(motion_mag.max() * 0.5, 1e-9)        # top-half-motion pairs
        strong     = motion_mag >= thr
        if strong.sum() == 0:
            strong = np.ones(num_pairs, dtype=bool)
        mae_pos = np.mean(np.abs( est[strong] - gt_rel[strong]))
        mae_neg = np.mean(np.abs(-est[strong] - gt_rel[strong]))
        sign = 1.0 if mae_pos <= mae_neg else -1.0
        print(f"\n[GT SIGN] auto-detected sign = {sign:+.0f}  "
              f"(decided on {int(strong.sum())} high-motion pairs; "
              f"MAE +:{mae_pos:.4e} vs -:{mae_neg:.4e})")
        # Warn if the two signs are nearly tied: in that case auto-detect is
        # unreliable and DIFFERENT runs may pick DIFFERENT signs, making their
        # error numbers non-comparable. Lock GT_SIGN at the top once known.
        rel_gap = abs(mae_pos - mae_neg) / (min(mae_pos, mae_neg) + 1e-12)
        if rel_gap < 0.05:
            print(f"  [GT SIGN WARNING] +/- are within {rel_gap*100:.2f}% -- "
                  f"auto-detect is UNRELIABLE here. Set GT_SIGN explicitly at the "
                  f"top (e.g. GT_SIGN = {sign:+.0f}) and re-run all four cells so "
                  f"every run uses the SAME sign convention.")
    else:
        sign = float(GT_SIGN)
        print(f"\n[GT SIGN] forced sign = {sign:+.0f}")

    est_signed = sign * est
    abs_err    = np.abs(est_signed - gt_rel)   # (num_pairs, num_radii) radians

    # Region split: inner (center) vs outer (periphery) rings
    half        = num_radii // 2
    inner_idx   = np.arange(0, half)
    outer_idx   = np.arange(half, num_radii)

    mae_all     = float(abs_err.mean())
    mae_inner   = float(abs_err[:, inner_idx].mean())
    mae_outer   = float(abs_err[:, outer_idx].mean())
    rmse_all    = float(np.sqrt((abs_err ** 2).mean()))
    mae_deg     = np.rad2deg(mae_all)

    # Per-ring MAE (useful for the figure)
    per_ring_mae = abs_err.mean(axis=0)  # (num_radii,)

    # --- center-localisation error vs RECORDED true centre (ROI coords) ---
    # Load the generator's recorded true centre path and convert to ROI coords.
    # estimator center at finest level is in ROI pixel coords (scale=1.0).
    gt_centers_full = load_gt_centers(csv_gt_centers, num_frames)   # (num_frames, 2) original coords
    gt_centers_roi  = centers_to_roi(gt_centers_full)               # (num_frames, 2) ROI coords
    center_err = np.zeros(num_pairs, dtype=np.float32)
    for pidx, (i_f, j_f) in enumerate(frame_pairs):
        true_c = gt_centers_roi[i_f]      # reference at source frame of the pair
        est_c  = centers_traj[pidx, :]
        center_err[pidx] = float(np.linalg.norm(est_c - true_c))
    mean_center_err = float(center_err.mean())

    print("\n" + "-" * 66)
    print(f"GROUND-TRUTH ERROR  |  arm={'structured' if STRUCTURED else 'unstructured'}  "
          f"U={U_BASIS if STRUCTURED else '-'}  FRAME_STEP={FRAME_STEP}")
    print("-" * 66)
    print(f"  Angle MAE  (all rings)     : {mae_all:.6f} rad   ({mae_deg:.4f} deg)")
    print(f"  Angle MAE  (inner/center)  : {mae_inner:.6f} rad")
    print(f"  Angle MAE  (outer/periph.) : {mae_outer:.6f} rad")
    print(f"  Angle RMSE (all rings)     : {rmse_all:.6f} rad")
    print(f"  Center-localisation error  : {mean_center_err:.4f} px (mean over pairs)")
    print("-" * 66)

    # --- append one row to the results CSV (assembles the 2x2 table) ---
    row = {
        "arm":          "structured" if STRUCTURED else "unstructured",
        "U":            U_BASIS if STRUCTURED else 0,
        "frame_step":   FRAME_STEP,
        "num_pairs":    num_pairs,
        "mae_all_rad":  mae_all,
        "mae_inner_rad":mae_inner,
        "mae_outer_rad":mae_outer,
        "rmse_all_rad": rmse_all,
        "mae_all_deg":  mae_deg,
        "center_err_px":mean_center_err,
        "gt_sign":      sign,
    }
    try:
        if os.path.exists(RESULTS_CSV):
            df_prev = pd.read_csv(RESULTS_CSV)
            df_out  = pd.concat([df_prev, pd.DataFrame([row])], ignore_index=True)
        else:
            df_out  = pd.DataFrame([row])
        df_out.to_csv(RESULTS_CSV, index=False)
        print(f"[results] appended run to {RESULTS_CSV}")
    except Exception as e:
        print(f"[results] could not write {RESULTS_CSV}: {e}")

    # --- dump FULL per-pair arrays for the separate comparison script ---
    # One .npz per {arm, frame_step}. The comparison script loads these and
    # overlays independent vs structured vs ground truth per frame pair.
    try:
        os.makedirs(PERPAIR_DIR, exist_ok=True)
        arm_tag = "structured" if STRUCTURED else "unstructured"
        fname   = os.path.join(PERPAIR_DIR, f"{arm_tag}_step{FRAME_STEP}.npz")
        np.savez(
            fname,
            arm=arm_tag,
            U=(U_BASIS if STRUCTURED else 0),
            frame_step=FRAME_STEP,
            radii=radii,
            half=half,
            sign=sign,
            source_frames=np.array([FRAME_START + i_f for (i_f, _) in frame_pairs]),
            est_signed=est_signed,                 # (num_pairs, num_radii) rad, sign-corrected
            gt_rel=gt_rel,                         # (num_pairs, num_radii) rad
            abs_err=abs_err,                       # (num_pairs, num_radii) rad
            per_ring_mae=per_ring_mae,             # (num_radii,)
            center_err=center_err,                 # (num_pairs,)
            mse_final=mse_by_level[-1],            # finest-level MSE curve (per outer iter)
        )
        print(f"[perpair] saved full arrays to {fname}")
    except Exception as e:
        print(f"[perpair] could not write per-pair arrays: {e}")

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    frame_pair_indices = np.array([FRAME_START + i_f for (i_f, _) in frame_pairs])

    # 1. Rotation angle vs frame for each radius.
    # Use the sign-corrected estimate (est_signed = sign * est) so the plotted
    # convention matches the ground truth (CCW positive, CW negative).
    #
    # SMOOTH_PLOT: the optical-flow reference plot applies a moving-average to its
    # curves (apply_moving_average) before plotting. For a fair, apples-to-apples
    # visual comparison we apply the SAME smoothing to the BBVI curves here.
    # This is COSMETIC (plotting only) -- it does not change the estimates or MAE.
    # Set SMOOTH_PLOT = 1 to disable and show raw values.
    SMOOTH_PLOT = 3   # moving-average window over frames (odd; 1 = off)

    def _smooth(y, w):
        if w <= 1:
            return y
        return np.convolve(y, np.ones(w) / w, mode='same')

    fig1, ax1 = plt.subplots(figsize=(14, 8))
    for ridx in range(0, num_radii, 1):
        r     = radii[ridx]
        yvals = _smooth(est_signed[:, ridx], SMOOTH_PLOT)
        ax1.plot(frame_pair_indices, yvals, marker='o', linewidth=2, markersize=6,
                 label=f"r={int(r)}")
    ax1.set_xlabel("Source frame index")
    ax1.set_ylabel("Rotation angle (radians)")
    xtick_vals = np.arange(frame_pair_indices.min(), frame_pair_indices.max() + 1, 5)
    ax1.set_xticks(xtick_vals)
    ax1.axhline(0, linestyle='--', linewidth=1)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    ax1.tick_params(axis='x', labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)
    plt.tight_layout()
    plt.show()

    # 1b. estimated vs ground-truth relative angle + per-ring MAE.
    # Pick the HIGHEST-MOTION pair (not the last one, which is often a rest pair
    # with zero true motion and therefore uninformative).
    motion_per_pair = np.abs(gt_rel).mean(axis=1)
    show_pair = int(np.argmax(motion_per_pair))
    show_src  = FRAME_START + frame_pairs[show_pair][0]
    fig_gt, (axA, axB) = plt.subplots(1, 2, figsize=(16, 6))
    # Smooth the estimated curve across radius (same idea as the OF moving-average)
    # so it reads cleanly against the ground truth. GT is left raw as the reference.
    axA.plot(radii, _smooth(est_signed[show_pair, :], SMOOTH_PLOT),
             marker='o', label="Estimated")
    axA.plot(radii, gt_rel[show_pair, :], marker='s', linestyle='--', label="Ground truth")
    axA.set_xlabel("Radius r (px)")
    axA.set_ylabel("Relative rotation (rad)")
    axA.set_xticks(radii)          # show actual ring radii: 10, 30, ..., 190
    axA.set_title(f"High-motion pair (src frame {show_src}): est vs GT "
                  f"({'structured' if STRUCTURED else 'unstructured'}, step={FRAME_STEP})")
    axA.legend()
    axB.plot(radii, per_ring_mae, marker='o', color='C3')
    axB.set_xlabel("Radius r (px)")
    axB.set_ylabel("Angle MAE (rad)")
    axB.set_xticks(radii)          # show actual ring radii: 10, 30, ..., 190
    axB.set_title("Per-ring MAE (inner rings = center)")
    axB.axvline(radii[half], linestyle=':', color='gray')
    plt.tight_layout()
    plt.show()

    # 2. Metrics plots
    metrics = [
        ("Noise Variance",        noise_variances_by_level),
        ("MSE",                   mse_by_level),
        ("ELBO (Log-Likelihood)", elbo_by_level)
    ]
    for metric_name, metric_data in metrics:
        fig = plt.figure(figsize=(14, 7))
        ax_main = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
        colors  = ['C0', 'C1', 'C2']
        for lvl in range(num_levels):
            ax_main.plot(np.arange(1, OUTER_EM_ITERS + 1), metric_data[lvl],
                         color=colors[lvl], label=f"Level {lvl}")
        ax_main.set_xlabel("Outer EM iteration")
        ax_main.set_ylabel(metric_name)
        ax_main.set_title(f"{metric_name} across pyramid levels")
        ax_main.legend()
        for lvl in range(num_levels):
            ax = plt.subplot2grid((3, 2), (lvl, 1))
            ax.plot(np.arange(1, OUTER_EM_ITERS + 1), metric_data[lvl], color=colors[lvl])
            ax.set_title(f"Level {lvl}")
            ax.set_xlabel("Iteration")
        plt.tight_layout()
        plt.show()

    # 3. Center trajectory
    centers = centers_traj.copy()
    fig_traj, ax_traj = plt.subplots(figsize=(9, 9))
    sc = ax_traj.scatter(centers[:, 0], centers[:, 1],
                         c=frame_pair_indices, cmap='viridis', s=60, zorder=3)
    ax_traj.plot(centers[:, 0], centers[:, 1], color='black', linewidth=1.5,
                 label='Connecting Center Trajectory', zorder=2)
    for i, (xi, yi) in enumerate(zip(centers[:, 0], centers[:, 1])):
        ax_traj.annotate(str(frame_pair_indices[i]), xy=(xi, yi), fontsize=9,
                         ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')
    ax_traj.add_patch(plt.Rectangle((0, 0), roi_w, roi_h, fill=False, edgecolor='red',
                                    linestyle='--', linewidth=2, label='ROI'))
    ax_traj.set_title("Adaptive COR Trajectory within Same ROI")
    ax_traj.set_xlabel("X Coordinate")
    ax_traj.set_ylabel("Y Coordinate")
    ax_traj.invert_yaxis()
    ax_traj.set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(sc, ax=ax_traj); cbar.set_label("Frame Pair")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple',
               markersize=8, label='Estimated Centers'),
        Line2D([0], [0], color='black', linewidth=1.5, label='Connecting Trajectory'),
    ]
    ax_traj.legend(handles=legend_elements, loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.show()

    # 3b. Zoomed center trajectory (grouped labels, no overlap)
    # Pass the PAIR-ALIGNED true centres (one per pair, indexed by each pair's
    # source frame i_f) -- the SAME indexing the full trajectory plot uses --
    # so the red true-centre curve lines up with the estimated centres instead
    # of plotting all 24 raw frames in the wrong order.
    # True-center overlay intentionally omitted (no red drift line).
    plot_cor_zoom(centers_traj, frame_pair_indices)


def _frames_to_label(fr_list):
    """Compact a sorted frame list into ranges, e.g. [10,11,12,15] -> '10-12,15'."""
    fr_list = sorted(int(f) for f in fr_list)
    parts, start, prev = [], fr_list[0], fr_list[0]
    for f in fr_list[1:]:
        if f == prev + 1:
            prev = f
        else:
            parts.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = f
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def _group_by_proximity(frs, xs, ys, tol=0.75):
    """Group frames whose estimated centers fall within tol px of each other,
    so their labels don't pile up (COR estimates are continuous, not integer)."""
    groups = OrderedDict()
    for fr, x, y in zip(frs, xs, ys):
        placed = False
        for (kx, ky) in groups:
            if abs(x - kx) <= tol and abs(y - ky) <= tol:
                groups[(kx, ky)].append(fr)
                placed = True
                break
        if not placed:
            groups[(x, y)] = [fr]
    return groups


def plot_cor_zoom(centers_traj, frame_pair_indices, gt_centers_roi=None, tol=0.75):
    cols = np.asarray(centers_traj[:, 0], dtype=float)
    rows = np.asarray(centers_traj[:, 1], dtype=float)
    frs  = list(frame_pair_indices)

    margin = 1.5
    xmn, xmx = cols.min() - margin, cols.max() + margin
    ymn, ymx = rows.min() - margin, rows.max() + margin

    plt.figure(figsize=(12, 12))
    sc = plt.scatter(cols, rows, c=frs, cmap='viridis', s=140, zorder=3,
                     label='Estimated centers')
    plt.plot(cols, rows, 'k-', lw=1.6, zorder=2)

    if gt_centers_roi is not None:
        gt = np.asarray(gt_centers_roi, dtype=float)
        plt.plot(gt[:, 0], gt[:, 1], 'r--', lw=1.6, zorder=1,
                 label='True center (drift)')

    groups = _group_by_proximity(frs, cols, rows, tol=tol)
    for k, ((xi, yi), fr_list) in enumerate(groups.items()):
        dy = 12 if (k % 2 == 0) else -16
        va = 'bottom' if dy > 0 else 'top'
        plt.annotate(_frames_to_label(fr_list), xy=(xi, yi),
                     fontsize=12, ha='center', va=va, fontweight='bold',
                     xytext=(0, dy), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray',
                               alpha=0.85, lw=0.6))

    plt.xlim(xmn, xmx)
    plt.ylim(ymx, ymn)          # inverted y to match image convention
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("X Coordinate", fontsize=18)
    plt.ylabel("Y Coordinate", fontsize=18)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.title("Zoomed COR Trajectory", fontsize=16)
    cbar = plt.colorbar(sc); cbar.set_label("Frame Pair")
    plt.legend(loc='upper left', fontsize=11)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()