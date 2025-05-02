import os
import glob
import numpy as np

import argparse

# === ARGUMENTS ===
parser = argparse.ArgumentParser(description="View segmentation masks overlayed on video frames")
parser.add_argument("npy_seg_folder", type=str, help="Path to segmentations folder")
args = parser.parse_args()

SEGMENTATIONS_DIR = args.npy_seg_folder

# === FIND ALL PER-VIDEO NPY FILES ===
pattern = os.path.join(SEGMENTATIONS_DIR, f"Group_1_Subtask3_*.npy")
video_segmentation_files = sorted(glob.glob(pattern))

if not video_segmentation_files:
    raise RuntimeError("❌ No per-video segmentation files found.")

print(f"✅ Found {len(video_segmentation_files)} per-video files:")
for f in video_segmentation_files:
    print("  -", os.path.basename(f))

# === LOAD AND COMBINE ===
all_segmentations = []

for file in video_segmentation_files:
    segs = np.load(file)
    print(f"Loaded {file}, shape: {segs.shape}")
    all_segmentations.append(segs)

# Concatenate along frame axis
final_array = np.concatenate(all_segmentations, axis=0)  # Shape: (total_frames, height, width)
print("✅ Combined array shape:", final_array.shape)

# === SAVE FINAL SUBMISSION FILE ===
submission_path = os.path.join(SEGMENTATIONS_DIR, f"Group_1_Subtask3_Results.npy")
np.save(submission_path, final_array)

print(f"✅ Final submission file saved to {submission_path}")
