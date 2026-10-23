import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
import os


def compute_optical_flow(previous_frame, current_frame):
    """
    Dense Farneback optical flow between two consecutive frames.
    Returns horizontal (u) and vertical (v) flow components.
    """
    flow = cv2.calcOpticalFlowFarneback(
        previous_frame, current_frame, None, 0.5, 3, 15, 20, 5, 1.0, 0
    )
    return flow[:, :, 0], flow[:, :, 1]


def calculate_local_rotation(flow_x, flow_y, center_row, center_col):
    """
    Rotation component from optical-flow derivatives (finite-difference curl):
        curl = dv/dx - du/dy
    """
    dv_dx = flow_y[center_row, center_col + 1] - flow_y[center_row, center_col]
    du_dy = flow_x[center_row + 1, center_col] - flow_x[center_row, center_col]
    return dv_dx - du_dy


def apply_moving_average(signal, window_length=3):
    """Smooth a 1-D signal with a moving-average filter."""
    return np.convolve(signal, np.ones(window_length) / window_length, mode='valid')


# =====================================================================
# Configuration
# =====================================================================
VIDEO_PATH = 'data/fasciculation_simulation_24frames.mp4'

# Recorded TRUE rotation-centre path (per frame), original-image coordinates.
# Columns: Frame, center_x, center_y  (1-indexed frames).
CENTER_CSV = 'data/ring_center_positions.csv'

REGION_OF_INTEREST = {
    'x_min': 240, 'x_max': 624,
    'y_min': 124, 'y_max': 740
}
ANALYSIS_RANGE = {
    'start_frame': 1,
    'end_frame': 24
}
RECTANGLE_DIMENSIONS = list(range(10, 200, 20))

# --- added for the public release: command-line overrides + figure saving ---
from runconfig import configure
configure(globals())


# =====================================================================
# Load the per-frame rotation centre (original-image coords) and convert
# to ROI-LOCAL coordinates. The flow arrays live in the cropped ROI frame,
# so the centre used for rotation sampling must be expressed relative to the
# ROI origin (x_min, y_min). This replaces the old fixed geometric midpoint.
# =====================================================================
def load_center_path_roi_local(path, roi):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Center CSV not found: {path}")
    centers = {}
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        # tolerate different header capitalisations
        fmap = {k.lower(): k for k in reader.fieldnames}
        fcol = fmap.get('frame')
        xcol = fmap.get('center_x', reader.fieldnames[1])
        ycol = fmap.get('center_y', reader.fieldnames[2])
        for row in reader:
            fr = int(float(row[fcol]))
            cx = float(row[xcol]) - roi['x_min']   # -> ROI-local x (column)
            cy = float(row[ycol]) - roi['y_min']   # -> ROI-local y (row)
            centers[fr] = (cx, cy)
    return centers


center_path_roi = load_center_path_roi_local(CENTER_CSV, REGION_OF_INTEREST)


# =====================================================================
# Video capture
# =====================================================================
video_capture = cv2.VideoCapture(VIDEO_PATH)
if not video_capture.isOpened():
    raise IOError(f"Cannot open video file: {VIDEO_PATH}")

rotation_measurements = {size: [] for size in RECTANGLE_DIMENSIONS}
center_used_per_frame = []   # (frame, col, row) actually used, for reporting

success, initial_frame = video_capture.read()
if not success:
    raise ValueError("Unable to read initial video frame")

previous_grayscale = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
frame_counter = 1

roi_h = REGION_OF_INTEREST['y_max'] - REGION_OF_INTEREST['y_min']
roi_w = REGION_OF_INTEREST['x_max'] - REGION_OF_INTEREST['x_min']


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# =====================================================================
# Main processing loop
# =====================================================================
while True:
    success, current_frame = video_capture.read()
    if not success or frame_counter > ANALYSIS_RANGE['end_frame']:
        break

    if frame_counter >= ANALYSIS_RANGE['start_frame']:
        # ---- Extract ROI for both frames ----
        roi_previous = previous_grayscale[
            REGION_OF_INTEREST['y_min']:REGION_OF_INTEREST['y_max'],
            REGION_OF_INTEREST['x_min']:REGION_OF_INTEREST['x_max']
        ]
        roi_current = cv2.cvtColor(current_frame[
            REGION_OF_INTEREST['y_min']:REGION_OF_INTEREST['y_max'],
            REGION_OF_INTEREST['x_min']:REGION_OF_INTEREST['x_max']
        ], cv2.COLOR_BGR2GRAY)

        # ---- Optical flow within ROI ----
        flow_x, flow_y = compute_optical_flow(roi_previous, roi_current)
        Hf, Wf = flow_x.shape

        # ---- ESTIMATED ROTATION CENTRE for this frame (ROI-local) ----
        # Use the recorded centre for the CURRENT frame of the pair; fall back
        # to the ROI geometric midpoint only if this frame is missing from CSV.
        if frame_counter in center_path_roi:
            c_col, c_row = center_path_roi[frame_counter]   # (x=col, y=row)
        else:
            c_col, c_row = (Wf / 2.0, Hf / 2.0)

        # integer pixel indices for sampling, clamped to valid interior range
        center_col = int(round(c_col))
        center_row = int(round(c_row))
        # keep a 1-px margin because calculate_local_rotation reads +1 neighbours
        center_col = clamp(center_col, 1, Wf - 2)
        center_row = clamp(center_row, 1, Hf - 2)

        center_used_per_frame.append((frame_counter, center_col, center_row))

        # ---- Rotation for each rectangle size, sampled AROUND the estimated centre ----
        for rect_size in RECTANGLE_DIMENSIONS:
            rotation_sum = 0.0
            half = rect_size // 2
            # bound the sampling window so +1 neighbour access stays in-bounds
            r0 = clamp(center_row - half, 0, Hf - 2)
            r1 = clamp(center_row + half, 1, Hf - 1)
            c0 = clamp(center_col - half, 0, Wf - 2)
            c1 = clamp(center_col + half, 1, Wf - 1)
            for row in range(r0, r1):
                for col in range(c0, c1):
                    rotation_sum += calculate_local_rotation(flow_x, flow_y, row, col)
            rotation_measurements[rect_size].append(rotation_sum / (rect_size ** 2))

        previous_grayscale = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    frame_counter += 1

video_capture.release()
# cv2.destroyAllWindows()  # Removed this line as it causes an error in headless environments

# =====================================================================
# Post-processing (unchanged normalisation / radius scaling)
# =====================================================================
for size in RECTANGLE_DIMENSIONS:
    rotation_measurements[size] = np.array(rotation_measurements[size])

processed_results = {}
radius_scaled_results = {}

for size, raw_values in rotation_measurements.items():
    smoothed_values = apply_moving_average(raw_values, 3)
    normalization_factor = 8 * size  # perimeter approximation
    normalized_data = smoothed_values / normalization_factor
    processed_results[size] = np.concatenate((
        [raw_values[0] / normalization_factor],
        normalized_data,
        [raw_values[-1] / normalization_factor]
    ))
    radius_scaled_results[size] = processed_results[size] * size

# ---- report the estimated centre actually used ----
print("Rotation centre used per frame (ROI-local col,row):")
for fr, ccol, crow in center_used_per_frame:
    print(f"  frame {fr:2d}: (col={ccol}, row={crow})")

# =====================================================================
# Plot: radius-scaled rotation values  (styled to match the target figure)
#
# Fixed output size: the figure canvas is FIG_W x FIG_H inches and the axes
# occupy a FIXED rectangle (AX_RECT). We deliberately do NOT use
# bbox_inches='tight' or tight_layout, because those crop/rescale to the
# content and make the saved width/height vary between figures. With this
# setup every saved file is exactly FIG_W*dpi by FIG_H*dpi pixels, and the
# plot box is the same shape, regardless of data amplitude or legend labels.
# =====================================================================
FIG_W, FIG_H = 14, 8               # inches  -> identical for every angle-vs-time figure
AX_RECT = [0.09, 0.12, 0.70, 0.83] # [left, bottom, width, height] as figure fractions
SAVE_DPI = 300                     # 14*300 x 8*300 = 4200 x 2400 px

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes(AX_RECT)         # fixed axes box -> constant plot width/height
for size, scaled_data in radius_scaled_results.items():
    frame_numbers = range(ANALYSIS_RANGE['start_frame'],
                          ANALYSIS_RANGE['start_frame'] + len(scaled_data))
    ax.plot(frame_numbers, scaled_data, marker='o', label=f"r={size}")   # no space after 'r='
ax.set_xlabel("Source frame index", fontsize=20)                         # was "Frame Number"
ax.set_ylabel("Rotation angle (radians)", fontsize=20)                   # was "Rotation Value (Scaled by Radius)"
ax.set_xticks([1, 6, 11, 16, 21])                                        # fixed tick values to match target
ax.tick_params(labelsize=20)
# To share ONE vertical scale with the other angle-vs-time figures, uncomment:
# ax.set_ylim(-0.10, 0.10)
ax.axhline(0, linestyle='--', linewidth=1)                               # thin blue dashed zero line (default C0)
ax.legend(fontsize=14, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
fig.savefig(os.path.join(OUTPUT_DIR, "rotation_angle_vs_frame.pdf"), dpi=SAVE_DPI)                  # vector PDF, fixed canvas (no bbox crop)
fig.savefig(os.path.join(OUTPUT_DIR, "rotation_angle_vs_frame.png"), dpi=SAVE_DPI)                  # same size PNG for quick checks
plt.show()

# =====================================================================
# Plot: estimated centre trajectory (ROI-local) for verification
# =====================================================================
if center_used_per_frame:
    frs  = [c[0] for c in center_used_per_frame]
    cols = [c[1] for c in center_used_per_frame]
    rows = [c[2] for c in center_used_per_frame]
    plt.figure(figsize=(7, 7))
    sc = plt.scatter(cols, rows, c=frs, cmap='viridis', s=60, zorder=3)
    plt.plot(cols, rows, 'k-', lw=1.2, zorder=2)
    plt.gca().add_patch(plt.Rectangle((0, 0), roi_w, roi_h, fill=False,
                                      edgecolor='red', linestyle='--', linewidth=2))
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("ROI-local X (col)", fontsize=16)
    plt.ylabel("ROI-local Y (row)", fontsize=16)
    plt.title("Estimated rotation centre per frame (from CSV)")
    cbar = plt.colorbar(sc); cbar.set_label("Frame")
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------
    # ZOOMED-IN view of the centre trajectory.
    # The drift is only ~9 px, so on the full ROI (384 x 616) all points
    # collapse into one blob. Zooming to the trajectory's own bounds (plus a
    # small margin) spreads the frames out so the per-frame path is visible.
    #
    # Because the centre is FROZEN during rest/plateau, several frames map to the
    # SAME pixel -> their labels would stack and overlap. We therefore GROUP all
    # frames that share a pixel into ONE compact label (e.g. "10-15"), and offset
    # it clear of the marker, so every frame is still shown but nothing overlaps.
    # -----------------------------------------------------------------
    cols_arr = np.array(cols, dtype=float)
    rows_arr = np.array(rows, dtype=float)
    margin = 1.5  # px padding around the trajectory (tighter -> more zoom)

    x_min_z = cols_arr.min() - margin
    x_max_z = cols_arr.max() + margin
    y_min_z = rows_arr.min() - margin
    y_max_z = rows_arr.max() + margin

    # group frames by identical (col,row) pixel
    from collections import OrderedDict
    groups = OrderedDict()
    for fr, xi, yi in zip(frs, cols_arr, rows_arr):
        groups.setdefault((xi, yi), []).append(fr)

    def frames_to_label(fr_list):
        """Compact a sorted frame list into ranges, e.g. [10,11,12,15] -> '10-12,15'."""
        fr_list = sorted(fr_list)
        parts, start, prev = [], fr_list[0], fr_list[0]
        for f in fr_list[1:]:
            if f == prev + 1:
                prev = f
            else:
                parts.append(f"{start}" if start == prev else f"{start}-{prev}")
                start = prev = f
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        return ",".join(parts)

    plt.figure(figsize=(12, 12))   # enlarged figure
    sc_z = plt.scatter(cols_arr, rows_arr, c=frs, cmap='viridis', s=140, zorder=3)
    plt.plot(cols_arr, rows_arr, 'k-', lw=1.6, zorder=2)
    # one grouped, offset label per unique pixel (alternate side to avoid clashes)
    for k, ((xi, yi), fr_list) in enumerate(groups.items()):
        dy = 12 if (k % 2 == 0) else -16   # alternate above/below
        va = 'bottom' if dy > 0 else 'top'
        plt.annotate(frames_to_label(fr_list), xy=(xi, yi),
                     fontsize=12, ha='center', va=va, fontweight='bold',
                     xytext=(0, dy), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray',
                               alpha=0.85, lw=0.6))
    plt.xlim(x_min_z, x_max_z)
    plt.ylim(y_max_z, y_min_z)   # inverted y to match the full-ROI view
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("ROI-local X (col)", fontsize=18)
    plt.ylabel("ROI-local Y (row)", fontsize=18)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.title("Zoomed: estimated rotation centre per frame", fontsize=16)
    cbar_z = plt.colorbar(sc_z); cbar_z.set_label("Frame")
    plt.tight_layout()
    plt.show()

# =====================================================================
# Export results to CSV
# =====================================================================
output_filename = (f'Normalized Rotation Values (Scaled by Radius) '
                   f'Frames {ANALYSIS_RANGE["start_frame"]}-{ANALYSIS_RANGE["end_frame"]}.csv')

output_filename = os.path.join(OUTPUT_DIR, output_filename)
with open(output_filename, 'w', newline='') as output_file:
    csv_writer = csv.writer(output_file)
    header_row = ['Frame'] + [f'Size {size}px' for size in RECTANGLE_DIMENSIONS]
    csv_writer.writerow(header_row)
    num_processed_frames = len(radius_scaled_results[RECTANGLE_DIMENSIONS[0]])
    for i in range(num_processed_frames):
        frame_num_to_write = ANALYSIS_RANGE['start_frame'] + i
        data_row = [frame_num_to_write] + [radius_scaled_results[size][i]
                                           for size in RECTANGLE_DIMENSIONS]
        csv_writer.writerow(data_row)

print(f"\nSaved rotation values to: {output_filename}")