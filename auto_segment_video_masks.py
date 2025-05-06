import os
import cv2
import torch
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
        bboxes = results[0].boxes.xyxy.cpu().numpy()
        class_nums = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        resultList = []
        for i in range(bboxes.shape[0]):
            class_value = class_nums[i]
            class_name = self.class_mapping[class_value]
            xmin, ymin, xmax, ymax = bboxes[i]
            confidence = confs[i]
            bbox = {
                "class": class_name,
                "xmin": round(xmin),
                "ymin": round(ymin),
                "xmax": round(xmax),
                "ymax": round(ymax),
                "conf": confidence
            }
            resultList.append(bbox)
        return resultList

# === CONFIGURATION ===
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CHECKPOINT = "./checkpoints/sam2.1_hiera_base_plus.pt"
DEVICE = "cuda"
CONFIDENCE_THRESHOLD = 0.90
ROOT_INPUT_DIR = r"C:\Users\Gaurav\Desktop\CreateChallenge\Data\Test_Data"
OUTPUT_FOLDER = r"C:\Users\Gaurav\Desktop\CreateChallenge\sam2-main"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === SETUP ===
predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT, device=DEVICE)
model = YOLOv8()

# === LOOP OVER FOLDERS ===
for folder_name in sorted(os.listdir(ROOT_INPUT_DIR)):
    FRAME_FOLDER = os.path.join(ROOT_INPUT_DIR, folder_name)
    if not os.path.isdir(FRAME_FOLDER):
        continue
    print(f"\n🔁 Processing folder: {folder_name}")
    # Skip if output .npy already exists
    submission_filename = f"Group_1_Subtask3_{folder_name}.npy"
    submission_path = os.path.join(OUTPUT_FOLDER, submission_filename)
    if os.path.exists(submission_path):
        print(f"✅ Already processed: {submission_filename}, skipping.")
        continue

    frame_files = sorted([
        f for f in os.listdir(FRAME_FOLDER)
        if f.lower().endswith(('.jpg', '.png'))
    ])

    if not frame_files:
        print(f"⚠️ No image files in {FRAME_FOLDER}")
        continue

    # === FIND A GOOD START FRAME ===
    start_frame_idx = None
    start_bbox = None
    best_confidence = 0

    for idx, frame_name in enumerate(tqdm(frame_files, desc="Searching for high confidence bbox")):
        frame_path = os.path.join(FRAME_FOLDER, frame_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        allboxes = model.predict(frame)
        ultrasound_obj = next((obj for obj in allboxes if obj['class'] == 'ultrasound'), None)
        if ultrasound_obj is not None and ultrasound_obj['conf'] > best_confidence:
            xmin = int(ultrasound_obj["xmin"])
            ymin = int(ultrasound_obj["ymin"])
            xmax = int(ultrasound_obj["xmax"])
            ymax = int(ultrasound_obj["ymax"])
            start_bbox = np.array([xmin, ymin, xmax, ymax])
            start_frame_idx = idx
            best_confidence = ultrasound_obj['conf']
            if best_confidence >= CONFIDENCE_THRESHOLD:
                print(f"✅ High confidence at frame {frame_name} index {idx}")
                break

    if start_frame_idx is None:
        print(f"❌ No valid ultrasound detection in {folder_name}, skipping.")
        continue

    # === LOAD ALL FRAMES ===
    all_frames = []
    for frame_name in tqdm(frame_files, desc="Loading all frames"):
        frame_path = os.path.join(FRAME_FOLDER, frame_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            raise RuntimeError(f"Failed to read frame: {frame_name}")
        all_frames.append(frame)

    # === RUN VIDEO PREDICTION ===
    inference_state = predictor.init_state(video_path=FRAME_FOLDER)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=start_frame_idx,
        obj_id=1,
        box=start_bbox.astype(np.float32)
    )

    print(f"✅ Running segmentation and saving outputs")
    all_masks = []
    for frame_idx, object_ids, masks in predictor.propagate_in_video(inference_state):
        frame_path = os.path.join(FRAME_FOLDER, frame_files[frame_idx])
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        mask_logit = masks[0]
        mask = (mask_logit > 0.0).cpu().numpy().astype(np.uint8)
        if mask.ndim == 3:
            mask = np.squeeze(mask, axis=0)
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        all_masks.append(mask_resized)

    if not all_masks:
        print(f"⚠️ No masks produced for {folder_name}")
        continue

    # === SAVE OUTPUT ===
    segmentation_array = np.stack(all_masks, axis=0)
    submission_filename = f"Group_1_Subtask3_{folder_name}.npy"
    submission_path = os.path.join(OUTPUT_FOLDER, submission_filename)
    np.save(submission_path, segmentation_array)
    print(f"✅ Saved output: {submission_path} with shape {segmentation_array.shape}")
