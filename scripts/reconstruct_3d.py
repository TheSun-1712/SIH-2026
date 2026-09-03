"""
AERO MESH 3D Reconstruction Script
Reconstructs a solid 3D scene from VisDrone images and YOLO detections.
Generates solid planes and CAD-like bounding boxes for an ultra-lightweight mesh representation.
"""

import os
import sys
import json
import random
import cv2
import torch
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    print("[Recon] ERROR: ultralytics is not installed. Please install it in venv_aeromesh.")
    sys.exit(1)

# Keep the original categories
CATEGORY_LABELS = {
    1: 'Pedestrian', 2: 'People', 3: 'Bicycle', 4: 'Car',
    5: 'Van', 6: 'Truck', 7: 'Tricycle', 8: 'Awning-Tricycle',
    9: 'Bus', 10: 'Motor', 11: 'Others'
}

CATEGORY_COLORS = {
    1: [56, 189, 248],   # Pedestrian - light blue
    2: [56, 189, 248],   # People
    3: [244, 63, 94],    # Bicycle - rose
    4: [16, 185, 129],   # Car - emerald
    5: [245, 158, 11],   # Van - amber
    6: [245, 158, 11],   # Truck
    7: [167, 139, 250],  # Tricycle - purple
    8: [167, 139, 250],  # Awning
    9: [245, 158, 11],   # Bus
    10: [244, 63, 94],   # Motor
    11: [100, 116, 139], # Others
}

def generate_mesh_data(weights_path: str, max_images: int = 5) -> tuple[list[dict], list[dict]]:
    """Generates 3D planes for ground map and 3D boxes for objects."""
    
    val_dir = os.path.join("VisDrone2019-DET-val", "VisDrone2019-DET-val", "images")
    if not os.path.exists(val_dir):
        print(f"[Recon] Could not find {val_dir}. Cannot generate map.")
        return [], []
        
    imgs = [os.path.join(val_dir, f) for f in sorted(os.listdir(val_dir)) if f.endswith('.jpg')]
    if max_images > 0:
        step = max(1, len(imgs) // max_images)
        imgs = imgs[::step][:max_images]
        
    print(f"[Recon] Loading YOLO weights: {weights_path} on cuda:0")
    model = YOLO(weights_path).to('cuda:0')
    
    planes: list[dict] = []
    boxes: list[dict] = []
    
    phys_scale = 0.05
    frame_spacing = 300.0 * phys_scale # spatial offset in meters
    
    for frame_idx, img_path in enumerate(imgs):
        filename = os.path.basename(img_path)
        
        # Load image for dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        offset_y = frame_idx * frame_spacing
        
        # 1. Generate the Ground Plane
        world_w = w * phys_scale
        world_h = h * phys_scale
        
        planes.append({
            "filename": filename,
            "x": 0.0,
            "y": offset_y,
            "z": -0.01, # slightly below zero to avoid z-fighting
            "w": world_w,
            "h": world_h,
            "frame": frame_idx
        })
        
        # 2. Run YOLO to get detection bounding boxes
        results = model(img_path, verbose=False)
        result = results[0]
        
        for box_data in result.boxes:
            b = box_data.xyxy[0].cpu().numpy()
            conf = float(box_data.conf[0].cpu().numpy())
            cls = int(box_data.cls[0].cpu().numpy())
            
            # Map YOLO class 0-9 to VisDrone category 1-10
            cls = cls + 1
            if cls not in CATEGORY_LABELS:
                cls = 11
                
            x1, y1, x2, y2 = b
            color = CATEGORY_COLORS.get(cls, CATEGORY_COLORS[11])
            label = CATEGORY_LABELS.get(cls, "Unknown")
            
            # Calculate physical dimensions of the bounding box
            area = (x2 - x1) * (y2 - y1)
            
            # Height depends roughly on area
            obj_height = np.clip(np.sqrt(area) * 0.8, 20.0, 150.0) * phys_scale
            
            world_w = (x2 - x1) * phys_scale
            world_l = (y2 - y1) * phys_scale
            
            # Center points relative to the plane
            px = ((x1 + x2) / 2.0 - w / 2.0) * phys_scale
            py = ((y1 + y2) / 2.0 - h / 2.0) * phys_scale + offset_y
            pz = obj_height / 2.0 # center of the box in Z
            
            boxes.append({
                "x": float(px),
                "y": float(py),
                "z": float(pz),
                "w": float(world_w),
                "h": float(obj_height),
                "d": float(world_l),
                "r": color[0], "g": color[1], "b": color[2],
                "conf": round(float(conf), 4),
                "cls": cls,
                "label": label,
                "frame": frame_idx
            })
            
        print(f"[Recon]   {frame_idx+1}/{len(imgs)} frames mapped -> {len(boxes)} boxes total")

    return planes, boxes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='aero_mesh/training/runs/visdrone_fast/weights/last.pt')
    parser.add_argument('--max-images', type=int, default=5, help='Number of frames to stitch')
    args = parser.parse_args()

    weights_path = os.path.abspath(args.weights)
    if not os.path.exists(weights_path):
        print(f"[Recon] YOLO weights not found at {weights_path}")
        sys.exit(1)

    planes, boxes = generate_mesh_data(weights_path, args.max_images)

    if not planes:
        print("[Recon] Failed to generate planes.")
        sys.exit(1)

    payload = {
        "generatedAt":  __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "weightsUsed":  weights_path,
        "epochAtExport": 39,
        "mAP50AtExport": 0.267,
        "modelName": "yolo11n",
        "classNames": list(CATEGORY_LABELS.values()),
        "nc": 10,
        "classColors": CATEGORY_COLORS,
        "planes": planes,
        "boxes": boxes
    }

    out_file = "web/public/pointcloud.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    with open(out_file, "w") as f:
        json.dump(payload, f)

    mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"\n[Recon] OK mesh map -> {os.path.abspath(out_file)}")
    print(f"        {len(planes)} planes, {len(boxes)} boxes | {mb:.2f} MB | epoch=39\n")


if __name__ == "__main__":
    main()
