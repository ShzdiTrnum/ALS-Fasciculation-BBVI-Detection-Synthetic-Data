import numpy as np
import cv2
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Video parameters
# ============================================================
width, height = 980, 748
center = (width // 2, height // 2)
radius_max = 200
total_frames = 24
fps = 5

# ============================================================
# Output folder -- all files (video + CSVs) are saved here.
# Change SYN_DIR to wherever you want the outputs (e.g. your Google Drive
# 'Syn' folder). The folder is created automatically if it doesn't exist.
# ============================================================
import os
# Default writes to outputs/regenerated so the shipped data/ files are not overwritten.
# The texture is random: use --seed N for a reproducible video. The video in data/ was
# made without a seed, so a regenerated video has the same motion but a different texture.
SYN_DIR = "outputs/regenerated"

# --- added for the public release: command-line overrides + figure saving ---
from runconfig import configure
configure(globals())
os.makedirs(SYN_DIR, exist_ok=True)

output_path = os.path.join(SYN_DIR, 'fasciculation_simulation_24frames.mp4')

num_rings = 10
ring_thickness = radius_max // num_rings

# ============================================================
# Timeline (24 frames), order = CCW event -> rest -> CW event.
# Rest frames at BOTH ends guarantee the first and last optical-flow
# measurements are exactly 0 (two identical frames -> zero flow).
#
#   lead-in rest : frames  0 ..  1   (2 frames, both at angle 0)
#   CCW event    : frames  2 ..  8   (positive rotation, all rings reach +TARGET)
#   middle rest  : frames  9 .. 14   (held at +TARGET)
#   CW event     : frames 15 .. 21   (negative rotation, all rings return to 0)
#   lead-out rest: frames 22 .. 23   (2 frames, both at angle 0)
# ============================================================
lead_in      = 2
event_frames = 7
mid_rest     = 6

ccw_start = lead_in                      # 2
ccw_end   = ccw_start + event_frames     # 9   (event = 2..8)
rest_end  = ccw_end + mid_rest           # 15  (rest  = 9..14)
cw_end    = rest_end + event_frames      # 22  (event = 15..21)
# frames 22..23 -> lead-out rest, brings us to total_frames = 24

ramp_frames = 4
# TARGET_ANGLE = 8 deg keeps the fast rings reaching full speed (~5 deg/frame),
# so the plot's y-magnitude (~0.01) matches your reference. The slow-ring floor
# is raised from 0.5 to 2.0 deg/frame so that EVERY ring still completes the
# rotation within event_frames and settles back to 0 (clean zero endpoints).
TARGET_ANGLE = 8.0

max_speeds = np.linspace(5, 2.0, num_rings)     # deg/frame (fast->slow), all settle in time
accelerations = max_speeds / ramp_frames

print(f"CCW event = {ccw_start}-{ccw_end-1}, rest = {ccw_end}-{rest_end-1}, "
      f"CW event = {rest_end}-{cw_end-1}, lead-out = {cw_end}-{total_frames-1}")

# ============================================================
# Center drift  (PHASE-TIED to the rotation events)
# The rotation centre moves ONLY while the rings are actually rotating, and it
# mirrors the there-and-back nature of the rotation:
#   * CCW event  (frames ccw_start..ccw_end-1): centre drifts OUT by +total_drift
#   * rest/plateau frames                      : centre is FROZEN (no motion)
#   * CW event   (frames rest_end..cw_end-1)   : centre drifts BACK by -total_drift
# Because the rings wind up CCW to +TARGET and unwind CW back to 0 (a there-and-
# back motion), the centre likewise leaves its start during CCW and RETURNS to
# exactly its start during CW. The pivot is thus coupled to the contraction/
# relaxation, not drifting arbitrarily. The whole concentric pattern is
# translated to stay centred on this drifting point each frame.
#   -> total_drift sets how far the centre wanders at peak (end of CCW).
# ============================================================
orig_center = np.array([width / 2.0, height / 2.0])   # (490, 374) fixed reference
total_drift = np.array([9.0, 6.0])                    # peak offset reached at end of CCW (px)

# per-frame increments spread evenly across each event's frames
ccw_len = ccw_end - ccw_start        # number of CCW frames (7)
cw_len  = cw_end  - rest_end         # number of CW  frames (7)
ccw_step = total_drift / ccw_len     # applied on each CCW frame (out)
cw_step  = -total_drift / cw_len     # applied on each CW  frame (back)

center_path = np.zeros((total_frames, 2))
cur = orig_center.copy()
for f in range(total_frames):
    if ccw_start <= f < ccw_end:        # CCW event -> drift out
        cur = cur + ccw_step
    elif rest_end <= f < cw_end:        # CW event  -> drift back
        cur = cur + cw_step
    # else: rest or plateau -> centre frozen (no change)
    center_path[f] = cur

# snap the final resting centre back to the exact start (kill tiny float drift)
center_path[cw_end - 1:] = orig_center
print(f"Center drift (phase-tied): start {tuple(orig_center)}, "
      f"peak at frame {ccw_end} = {tuple(np.round(center_path[ccw_end-1], 2))}, "
      f"returns to {tuple(np.round(center_path[-1], 2))}")

# ============================================================
# Video writer + base texture
# ============================================================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)

ring_masks = []
for i in range(num_rings):
    inner_radius = i * ring_thickness
    outer_radius = (i + 1) * ring_thickness
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, outer_radius, 255, -1)
    cv2.circle(mask, center, inner_radius, 0, -1)
    ring_masks.append(mask)
ring_textures = [cv2.bitwise_and(noise, mask) for mask in ring_masks]

# ============================================================
# State
# ============================================================
angles = np.zeros(num_rings)
angular_speeds = np.zeros(num_rings)
speed_history = np.zeros((total_frames, num_rings))
angle_history = np.zeros((total_frames, num_rings))

for frame_idx in range(total_frames):
    composed_frame = np.zeros((height, width), dtype=np.uint8)

    # ---- phase selection by frame index ----
    if ccw_start <= frame_idx < ccw_end:
        phase, direction, target = 'CCW event', +1, +TARGET_ANGLE
    elif rest_end <= frame_idx < cw_end:
        phase, direction, target = 'CW event', -1, 0.0
    else:
        phase, direction, target = 'rest', 0, None

    # Current (drifting) center for this frame, and the translation needed
    # to move the pattern (built about orig_center) onto that drifting point.
    frame_center = center_path[frame_idx]
    delta = frame_center - orig_center
    cx, cy = int(round(frame_center[0])), int(round(frame_center[1]))

    info_lines = [f"Frame: {frame_idx + 1}/{total_frames}  [{phase}]",
                  f"Center: ({cx},{cy})"]

    for i in range(num_rings):
        if phase == 'rest':
            angular_speeds[i] = 0.0
        else:
            if not np.isclose(angles[i], target, atol=1e-2):
                if direction == 1:
                    angular_speeds[i] = min(angular_speeds[i] + accelerations[i], max_speeds[i])
                else:
                    angular_speeds[i] = max(angular_speeds[i] - accelerations[i], -max_speeds[i])
                angles[i] += angular_speeds[i]
                if (direction == 1 and angles[i] > target) or \
                   (direction == -1 and angles[i] < target):
                    angles[i] = target
                    angular_speeds[i] = 0.0
            else:
                angular_speeds[i] = 0.0

        # Rotate about the fixed reference center, then translate by `delta`
        # so the whole concentric pattern is re-centred on the drifting point.
        rot_mat = cv2.getRotationMatrix2D(center, angles[i], 1.0)
        rot_mat[0, 2] += delta[0]
        rot_mat[1, 2] += delta[1]
        rotated_ring = cv2.warpAffine(ring_textures[i], rot_mat, (width, height),
                                      flags=cv2.INTER_LINEAR, borderValue=0)
        composed_frame = cv2.bitwise_or(composed_frame, rotated_ring)
        if i < 5:
            info_lines.append(f"Ring {i+1}: angle={angles[i]:.2f}, speed={abs(angular_speeds[i]):.2f}")

    # ---- build colour frame so we can draw a red centre marker ----
    color_frame = cv2.cvtColor(composed_frame, cv2.COLOR_GRAY2BGR)
    for idx, line in enumerate(info_lines):
        y = 20 + idx * 18
        cv2.putText(color_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # red dot (BGR = (0,0,255)) at the drifting centre
    cv2.circle(color_frame, (cx, cy), 6, (0, 0, 255), -1)

    video_writer.write(color_frame)
    speed_history[frame_idx, :] = angular_speeds.copy()
    angle_history[frame_idx, :] = angles.copy()

video_writer.release()
print("Video written:", output_path)

np.savetxt(os.path.join(SYN_DIR, 'ring_speeds.csv'),     speed_history, delimiter=',')
np.savetxt(os.path.join(SYN_DIR, 'ring_angles.csv'),     angle_history, delimiter=',')
np.savetxt(os.path.join(SYN_DIR, 'ring_speed_diff.csv'), np.diff(speed_history, axis=0), delimiter=',')
np.savetxt(os.path.join(SYN_DIR, 'ring_angle_diff.csv'), np.diff(angle_history, axis=0), delimiter=',')

# ------------------------------------------------------------
# Ground-truth cumulative rotation angle per ring per frame,
# written in the SAME format as ring_initial_rotation_values.csv:
#   header  : Frame, Rectangle_0, ..., Rectangle_9
#   column  : Rectangle_i corresponds to ring i (0 = fastest inner ring)
#   Frame   : 1-indexed
# These are the cumulative angles the generator actually applied,
# i.e. the ground truth (not the optical-flow-measured values).
# ------------------------------------------------------------
with open(os.path.join(SYN_DIR, 'ring_initial_rotation_values.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Frame'] + [f'Rectangle_{i}' for i in range(num_rings)])
    for frame_idx in range(total_frames):
        writer.writerow([frame_idx + 1] + list(angle_history[frame_idx, :]))

# ------------------------------------------------------------
# Ground-truth drifting-center path (1-indexed frames).
# ------------------------------------------------------------
with open(os.path.join(SYN_DIR, 'ring_center_positions.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Frame', 'center_x', 'center_y'])
    for frame_idx in range(total_frames):
        writer.writerow([frame_idx + 1,
                         center_path[frame_idx, 0], center_path[frame_idx, 1]])

print("final angles:", np.round(angle_history[-1], 3))
print(f"All files saved in folder: {os.path.abspath(SYN_DIR)}")
print("  - fasciculation_simulation_24frames.mp4")
print("  - ring_angles.csv, ring_speeds.csv, ring_angle_diff.csv, ring_speed_diff.csv")
print("  - ring_initial_rotation_values.csv (ground-truth angles)")
print("  - ring_center_positions.csv (center path)")