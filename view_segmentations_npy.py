import os
import numpy as np
import cv2
import argparse

def visualize_mask_overlay(video_folder, npy_file):
    masks = np.load(npy_file)  # (T, H, W)
    frame_files = sorted([
        f for f in os.listdir(video_folder)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    total_frames = len(frame_files)
    mask_count = len(masks)

    if mask_count > total_frames:
        raise ValueError(f"More masks ({mask_count}) than frames ({total_frames})! Check data.")
    
    # Compute auto offset (e.g., SAM2 started at frame 154, so offset = 154)
    offset = total_frames - mask_count
    print(f"Detected frame/mask mismatch: applying offset of {offset} frames")

    #output_dir = os.path.join(video_folder, "overlay_output")
    #os.makedirs(output_dir, exist_ok=True)

    for i in range(mask_count):
        frame_name = frame_files[i + offset]
        frame_path = os.path.join(video_folder, frame_name)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"⚠️ Skipping unreadable frame: {frame_name}")
            continue

        mask = masks[i]

        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay = frame.copy()
        overlay[mask > 0] = (0, 255, 0)
        combined = np.hstack((frame, overlay))

        # === Show in window
        cv2.imshow("Original | Mask Overlay", combined)

        # === Save to output folder
        #out_path = os.path.join(output_dir, f"{i + offset:04d}.jpg")
        #cv2.imwrite(out_path, combined)

        key = cv2.waitKey(30)
        if key == 27:
            break


    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View segmentation overlay on video frames")
    parser.add_argument("video_folder", type=str, help="Folder containing video frames")
    parser.add_argument("npy_file", type=str, help=".npy file containing (T, H, W) mask array")
    args = parser.parse_args()

    visualize_mask_overlay(args.video_folder, args.npy_file)
