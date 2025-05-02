import os
import json
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
from segment_anything import sam_model_registry
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy
import os
import cv2
import yaml
from ultralytics import YOLO

class YOLOv8:
    def __init__(self):
        self.model = YOLO("../Central_Line_Challenge/Tool_Detection/best.pt")
        self.model_config = "../Central_Line_Challenge/Tool_Detection/CREATE_2/config.yaml"
        with open(self.model_config,"r") as f:
            config = yaml.safe_load(f)
            self.class_mapping = config['class_mapping']

    def predict(self,image):
        results = self.model.predict(image,cfg=self.model_config)
        bboxes = results[0].boxes.xyxy.gpu().numpy()
        class_nums = results[0].boxes.cls.gpu().numpy()
        confs = results[0].boxes.conf.gpu().numpy()
        resultList = []
        for i in range(bboxes.shape[0]):
            class_value = class_nums[i]
            class_name = self.class_mapping[class_value]
            xmin,ymin,xmax,ymax = bboxes[i]
            confidence = confs[i] #[class_value]
            bbox = {"class":class_name,
                    "xmin":round(xmin),
                    "ymin":round(ymin),
                    "xmax":round(xmax),
                    "ymax":round(ymax),
                    "conf":confidence}
            resultList.append(bbox)
        return resultList
    

# === CONFIGURATION ===
CSV_PATH = "../Training_Data/Training_Data.csv"
OUTPUT_DIR = "../Ultrasound_Segmentations/image_level"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
CHECKPOINT = "./checkpoints/sam2.1_hiera_base_plus.pt"
DEVICE = "cuda"

# === PARSE ARGUMENTS ===
parser = argparse.ArgumentParser(description="Segment one ultrasound frame using SAM2.")
parser.add_argument("frame", type=str, help="Name of the frame image (e.g. frame_00123.jpg)")
parser.add_argument("--save", action="store_true", help="If set, save the overlay instead of displaying")
args = parser.parse_args()
FRAME_NAME = args.frame
SAVE_IMG = args.save

# Load model
sam2_model = build_sam2(MODEL_CFG, CHECKPOINT, device=DEVICE)
predictor = SAM2ImagePredictor(sam2_model)

# Load frame
frame = cv2.imread(FRAME_NAME)
if frame is None:
    raise FileNotFoundError(f"Frame not found: {FRAME_NAME}")


# load bounding box
model = YOLOv8()
allboxes = model.predict(FRAME_NAME)
ultrasound_obj = next((obj for obj in allboxes if obj['class'] == 'ultrasound'), None)
if ultrasound_obj.get("conf", 1.0) < 0.9:
    print(f"WARN: Low confidence: {ultrasound_obj['conf']:.2f} for class '{ultrasound_obj['class']}'")

# Extract bounding box coordinates
xmin = int(float(ultrasound_obj["xmin"]))
ymin = int(float(ultrasound_obj["ymin"]))
xmax = int(float(ultrasound_obj["xmax"]))
ymax = int(float(ultrasound_obj["ymax"]))
bbox = np.array([xmin, ymin, xmax, ymax])

print("!! bbox", bbox, f"confidence: {ultrasound_obj['conf']:.2f}")

# predict from bounding box
predictor.set_image(frame)
masks, _, _ = predictor.predict(box=bbox[None, :], multimask_output=False)
mask = masks[0].astype(np.uint8) * 255

# Overlay
overlay = frame.copy()
overlay[mask > 0] = (0, 255, 0)  # Green overlay

if not SAVE_IMG:
    cv2.imshow("Segmentation Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    # Create output folders
    video_name = os.path.basename('.')
    mask_dir = os.path.join(OUTPUT_DIR, video_name, 'mask')
    overlay_dir = os.path.join(OUTPUT_DIR, video_name, 'overlay')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # Save mask
    mask_path = os.path.join(mask_dir, FRAME_NAME.replace(".jpg", "_mask.png"))
    cv2.imwrite(mask_path, mask)

    # Save overlay
    overlay_path = os.path.join(overlay_dir, FRAME_NAME.replace(".jpg", "_overlay.jpg"))
    cv2.imwrite(overlay_path, overlay)
