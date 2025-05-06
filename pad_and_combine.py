import os
import numpy as np

# === CONFIG ===
VIDEO_FRAME_DIR = r"C:\Users\Gaurav\Desktop\CreateChallenge\Data\Test_Data"
MASK_INPUT_DIR = r"C:\Users\Gaurav\Desktop\CreateChallenge\sam2-main"
FINAL_OUTPUT_DIR = r"C:\Users\Gaurav\Desktop\CreateChallenge\sam2-main\Subtask3_Output2"

os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

# === STEP 1: Get frame counts per video ===
frame_counts = {}
for folder in os.listdir(VIDEO_FRAME_DIR):
    folder_path = os.path.join(VIDEO_FRAME_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
    frame_counts[folder] = len(images)

# === STEP 2: Process each mask ===
for fname in os.listdir(MASK_INPUT_DIR):
    if not fname.endswith(".npy"):
        continue

    # Extract folder name (e.g., "Test_01") from "Group_1_Subtask3_Test_01.npy"
    video_id = fname.replace("Group_1_Subtask3_", "").replace(".npy", "")
    if video_id not in frame_counts:
        print(f"❌ Skipping {fname}: no folder named {video_id} in Test_Data.")
        continue

    # Load mask and count frames
    npy_path = os.path.join(MASK_INPUT_DIR, fname)
    masks = np.load(npy_path)
    expected_frames = frame_counts[video_id]
    mask_frames = masks.shape[0]

    if mask_frames < expected_frames:
        # Assume masks are from the end: pad at the front
        missing = expected_frames - mask_frames
        pad_shape = (missing, masks.shape[1], masks.shape[2])
        padding = np.zeros(pad_shape, dtype=np.uint8)
        masks = np.concatenate([padding, masks], axis=0)
        print(f"🩹 Padded {video_id} with {missing} empty frames at the beginning.")

    elif mask_frames > expected_frames:
        masks = masks[-expected_frames:]
        print(f"✂️ Trimmed {video_id} to last {expected_frames} frames.")

    # Save new .npy file
    save_name = f"{video_id}_segmentations.npy"
    save_path = os.path.join(FINAL_OUTPUT_DIR, save_name)
    np.save(save_path, masks)
    print(f"✅ Saved {save_name} with shape {masks.shape}")

print(f"\n🎉 All padded/trimmed masks saved in: {FINAL_OUTPUT_DIR}")
