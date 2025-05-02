import os
import cv2
import torch
import argparse
import numpy as np
import yaml
from tqdm import tqdm
from segment_anything import sam_model_registry
from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO

# === YOLOv8 Wrapper ===
class YOLOv8:
    def __init__(self):
        self.model = YOLO("./bbox_checkpoints/flora/best.pt")
        self.model_config = "./bbox_checkpoints/flora/config.yaml"
        with open(self.model_config, "r") as f:
            config = yaml.safe_load(f)
            self.class_mapping = config['class_mapping']

    def predict(self, image):
        results = self.model.predict(image, cfg=self.model_config)
        bboxes = results[0].boxes.xyxy.gpu().numpy()
        class_nums = results[0].boxes.cls.gpu().numpy()
        confs = results[0].boxes.conf.gpu().numpy()
        resultList = []
        for i in range(bboxes.shape[0]):
            class_value = class_nums[i]
            class_name = self.class_mapping[class_value]
            xmin, ymin, xmax, ymax = bboxes[i]
            confidence = confs[i]
            bbox = {"class": class_name,
                    "xmin": round(xmin),
                    "ymin": round(ymin),
                    "xmax": round(xmax),
                    "ymax": round(ymax),
                    "conf": confidence}
            resultList.append(bbox)
        return resultList

# === CONFIGURATION ===
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CHECKPOINT = "./checkpoints/sam2.1_hiera_base_plus.pt"
DEVICE = "cuda"
MAX_FRAMES = 100  # How many frames to search through for a good bbox
CONFIDENCE_THRESHOLD = 0.90  # How confident YOLO must be before accepting

# === PARSE ARGUMENTS ===
parser = argparse.ArgumentParser(description="Segment ultrasound tool through video using SAM2 and YOLO.")
parser.add_argument("frame_folder", type=str, help="Path to folder containing frame images")
args = parser.parse_args()

FRAME_FOLDER = args.frame_folder

# Auto-generate output folder
input_folder_name = os.path.basename(os.path.normpath(FRAME_FOLDER))
OUTPUT_FOLDER = os.path.join("/Users/emma/Desktop/QUEENS/CREATE_CHALLENGE/Ultrasound_Segmentations", f"{input_folder_name}_video_segmentation")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === SETUP ===
predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device=DEVICE)
model = YOLOv8()

# Collect frames
frame_files = sorted([
    f for f in os.listdir(FRAME_FOLDER)
    if f.lower().endswith(('.jpg', '.png'))
])

if not frame_files:
    raise ValueError(f"No image files found in {FRAME_FOLDER}")

# === FIND A GOOD START FRAME ===
start_frame_idx = None
start_bbox = None
best_confidence = 0

for idx, frame_name in enumerate(tqdm(frame_files, desc="Searching for high confidence bbox")):
    frame_path = os.path.join(FRAME_FOLDER, frame_name)
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Skipping unreadable frame: {frame_name}")
        continue

    allboxes = model.predict(frame)
    ultrasound_obj = next((obj for obj in allboxes if obj['class'] == 'ultrasound'), None)

    if ultrasound_obj is not None and ultrasound_obj['conf'] > best_confidence:
        xmin = int(float(ultrasound_obj["xmin"]))
        ymin = int(float(ultrasound_obj["ymin"]))
        xmax = int(float(ultrasound_obj["xmax"]))
        ymax = int(float(ultrasound_obj["ymax"]))
        start_bbox = np.array([xmin, ymin, xmax, ymax])
        start_frame_idx = idx
        best_confidence = ultrasound_obj['conf']

        if ultrasound_obj['conf'] >= CONFIDENCE_THRESHOLD:
            print(f"✅ Found high confidence bbox at frame {frame_name} (index {idx}) with confidence {ultrasound_obj['conf']:.2f}")
            break

print('best confidence', best_confidence, 'index', start_frame_idx)
if start_frame_idx is None:
    raise RuntimeError("❌ Could not find any frame with high ultrasound detection.")

# === LOAD ALL FRAMES ===
all_frames = []
for frame_name in tqdm(frame_files, desc="Loading all frames"):
    frame_path = os.path.join(FRAME_FOLDER, frame_name)
    frame = cv2.imread(frame_path)
    if frame is None:
        raise RuntimeError(f"Failed to read frame: {frame_name}")
    all_frames.append(frame)

# === RUN VIDEO PREDICTION ===
# Initialize inference state
inference_state = predictor.init_state(video_path=FRAME_FOLDER)

# Add the starting bounding box
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=start_frame_idx,
    obj_id=1,
    box=start_bbox.astype(np.float32)
)

# === SAVE MASKS WITH OVERLAY ===
print(f"✅ Saving overlay masks to {OUTPUT_FOLDER}")

for frame_idx, object_ids, masks in predictor.propagate_in_video(inference_state):
    # Load original frame
    frame_path = os.path.join(FRAME_FOLDER, frame_files[frame_idx])
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"⚠️ Skipping unreadable frame {frame_idx}")
        continue

    # Get mask for object 1
    mask_logit = masks[0]  # (1, H, W)
    mask = (mask_logit > 0.0).gpu().numpy().astype(np.uint8)  # Still shape (1, H, W)

    # Squeeze channel dimension if necessary
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=0)  # Now shape (H, W)

    # Resize mask to match frame (just in case)
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Overlay
    overlay = frame.copy()
    overlay[mask > 0] = (0, 255, 0)

    # Save overlay
    out_path = os.path.join(OUTPUT_FOLDER, f"{frame_idx:04d}.jpg")
    cv2.imwrite(out_path, overlay)

print(f"✅ Done! Saved overlay frames.")
