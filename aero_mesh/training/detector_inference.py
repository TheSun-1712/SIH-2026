"""
AERO MESH — YOLOv11 Inference Detector
Uses the trained YOLOv11x VisDrone model for real-time dynamic object detection.
Integrates with Stage 7 (motion_filter.py) as the primary object detector,
replacing/augmenting RAFT optical flow with GT-quality bbox detection.

Usage in pipeline:
  from aero_mesh.training.detector_inference import VisDroneDetector
  detector = VisDroneDetector()
  masks = detector.detect_and_mask(frame_bgr)
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List, Tuple

# ── YOLO class names (matches visdrone_converter.py) ─────────────────────────
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

# ALL classes are dynamic for 3D reconstruction purposes
DYNAMIC_CLASS_INDICES = list(range(len(CLASS_NAMES)))  # 0-9 = all dynamic

# Default weights path (points to the active fast training run's latest checkpoint)
DEFAULT_WEIGHTS = r"E:\Projects\SIH\aero_mesh\training\runs\visdrone_fast\weights\last.pt"


class VisDroneDetector:
    """
    YOLOv11x detector for VisDrone dynamic objects.
    Produces binary masks for all detected dynamic objects in a frame.
    
    Integration with AERO MESH:
      - Replace/augment RAFT optical flow in motion_filter.py
      - Use bbox detections → SAM2 for precise instance masks
      - Zero out dynamic pixels in depth maps before TSDF fusion
    """

    def __init__(
        self,
        weights_path: str = DEFAULT_WEIGHTS,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        imgsz: int = 1024,
        device: str = "auto",
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self._model = None
        self._weights_path = weights_path
        self._device = device
        self._loaded = False

    def _load_model(self):
        if self._loaded:
            return self._model is not None
        self._loaded = True

        if not os.path.exists(self._weights_path):
            print(f"[Detector] ⚠ Weights not found: {self._weights_path}")
            print("[Detector]   Run training first: python aero_mesh/training/train_detector.py")
            print("[Detector]   Using COCO pretrained YOLOv11n as fallback...")
            weights = "yolo11n.pt"  # COCO pretrained fallback
        else:
            weights = self._weights_path
            print(f"[Detector] Loading YOLOv11x VisDrone weights: {weights}")

        try:
            from ultralytics import YOLO
            import torch

            if os.environ.get("AERO_CPU_ONLY") == "1":
                self._device = "cpu"
            elif self._device == "auto":
                self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

            self._model = YOLO(weights)
            print(f"[Detector] ✓ YOLOv11 loaded on {self._device}")
            return True
        except Exception as e:
            print(f"[Detector] ERROR: {e}")
            self._model = None
            return False

    def detect(self, image_bgr: np.ndarray) -> List[Tuple[int, float, int, int, int, int]]:
        """
        Run detection on a BGR image.
        Returns list of (class_id, confidence, x1, y1, x2, y2).
        """
        if not self._load_model() or self._model is None:
            return []

        try:
            results = self._model.predict(
                image_bgr,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self._device,
                verbose=False,
            )
            detections = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    detections.append((cls, conf, x1, y1, x2, y2))
            return detections
        except Exception as e:
            print(f"[Detector] Inference error: {e}")
            return []

    def detect_and_mask(self, image_bgr: np.ndarray,
                        expand_pixels: int = 4) -> np.ndarray:
        """
        Detect all dynamic objects and return a binary mask (H, W) bool.
        Mask is True at all dynamic object pixels.
        
        expand_pixels: morphological expansion to cover object borders.
        """
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        detections = self.detect(image_bgr)

        for (cls, conf, x1, y1, x2, y2) in detections:
            if cls not in DYNAMIC_CLASS_INDICES:
                continue
            x1 = max(0, x1 - expand_pixels)
            y1 = max(0, y1 - expand_pixels)
            x2 = min(w, x2 + expand_pixels)
            y2 = min(h, y2 + expand_pixels)
            mask[y1:y2, x1:x2] = 1

        # Light morphological dilation to cover vehicle edges
        if np.any(mask):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask.astype(bool)

    def annotate_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Draw detection bounding boxes on image for visualization.
        Returns annotated BGR image.
        """
        vis = image_bgr.copy()
        detections = self.detect(image_bgr)

        COLORS = [
            (0, 255, 0), (0, 200, 255), (255, 100, 0), (0, 0, 255),
            (255, 0, 255), (128, 0, 255), (0, 128, 255), (255, 128, 0),
            (0, 255, 128), (128, 255, 0),
        ]

        for (cls, conf, x1, y1, x2, y2) in detections:
            color = COLORS[cls % len(COLORS)]
            label = f"{CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else cls} {conf:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python detector_inference.py <image_path>")
        print("       python detector_inference.py <weights.pt> <image_path>")
        sys.exit(1)

    if len(sys.argv) >= 3:
        weights = sys.argv[1]
        img_path = sys.argv[2]
    else:
        weights = DEFAULT_WEIGHTS
        img_path = sys.argv[1]

    detector = VisDroneDetector(weights_path=weights)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        sys.exit(1)

    detections = detector.detect(img)
    print(f"Detections: {len(detections)}")
    for cls, conf, x1, y1, x2, y2 in detections:
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        print(f"  {name}: {conf:.2f} @ [{x1},{y1},{x2},{y2}]")

    vis = detector.annotate_image(img)
    out_path = img_path.replace(".jpg", "_detected.jpg")
    cv2.imwrite(out_path, vis)
    print(f"Annotated image saved: {out_path}")
