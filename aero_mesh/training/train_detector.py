"""
AERO MESH — YOLOv11x VisDrone Training Script
Trains YOLOv11x (largest, best accuracy) on VisDrone dataset for dynamic object detection.
Uses GPU (CUDA) for training. The trained model feeds Stage 7 of the AERO MESH pipeline
to produce object masks used in RAFT+SAM2 dynamic masking.

Usage:
  # First convert the dataset:
  python aero_mesh/dataset/visdrone_converter.py

  # Then train:
  python aero_mesh/training/train_detector.py

  # Or with custom options:
  python aero_mesh/training/train_detector.py \
    --model yolo11x.pt \
    --data E:/Projects/SIH/yolo_visdrone/dataset.yaml \
    --epochs 100 \
    --batch 8 \
    --imgsz 1024 \
    --device 0
"""

import os
import sys
import argparse
import json
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Ensure we can import from project root
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ──────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────
DEFAULT_YAML   = r"E:\Projects\SIH\yolo_visdrone\dataset.yaml"
DEFAULT_MODEL  = "yolo11x.pt"       # YOLOv11-XLarge — best accuracy
DEFAULT_EPOCHS = 100
DEFAULT_BATCH  = 8                  # Adjust to your VRAM; RTX 3080=8-16 @ imgsz=1024
DEFAULT_IMGSZ  = 1024               # VisDrone has high-res images; larger = better
DEFAULT_DEVICE = "0"                # GPU 0; use "cpu" if no CUDA
DEFAULT_OUTPUT = r"E:\Projects\SIH\aero_mesh\training\runs"
DEFAULT_WORKERS = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv11x on VisDrone for AERO MESH dynamic object masking"
    )
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--data",    default=DEFAULT_YAML)
    parser.add_argument("--epochs",  type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch",   type=int, default=DEFAULT_BATCH)
    parser.add_argument("--imgsz",   type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--device",  default=DEFAULT_DEVICE)
    parser.add_argument("--output",  default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--convert-only", action="store_true",
                        help="Only run dataset conversion, skip training")
    return parser.parse_args()


def run_conversion():
    """Run VisDrone → YOLO dataset conversion."""
    from aero_mesh.dataset.visdrone_converter import run_full_conversion
    result = run_full_conversion()
    return result


def run_training(args):
    """Run YOLOv11x training with Ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Train] ERROR: ultralytics not installed.")
        print("  Install: pip install ultralytics")
        sys.exit(1)

    import torch
    device_info = f"CUDA ({torch.cuda.get_device_name(0)})" \
                  if torch.cuda.is_available() else "CPU"
    print(f"\n{'=' * 60}")
    print(f"AERO MESH — YOLOv11x Training")
    print(f"{'=' * 60}")
    print(f"  Model     : {args.model}")
    print(f"  Data      : {args.data}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch     : {args.batch}")
    print(f"  Image size: {args.imgsz}px")
    print(f"  Device    : {device_info}")
    print(f"  Output    : {args.output}")
    print(f"{'=' * 60}\n")

    # Validate dataset YAML exists
    if not os.path.exists(args.data):
        print(f"[Train] ERROR: Dataset YAML not found: {args.data}")
        print("  Run dataset conversion first:")
        print("  python aero_mesh/dataset/visdrone_converter.py")
        sys.exit(1)

    # Load model
    model = YOLO(args.model)

    # Training hyperparameters optimised for VisDrone (small objects, high density)
    train_kwargs = {
        "data":           args.data,
        "epochs":         args.epochs,
        "batch":          args.batch,
        "imgsz":          args.imgsz,
        "device":         args.device,
        "workers":        args.workers,
        "project":        args.output,
        "name":           "visdrone_yolo11x",
        "resume":         args.resume,
        "patience":       25,           # Early stopping patience
        "save":           True,
        "save_period":    10,           # Save checkpoint every 10 epochs

        # Optimizer
        "optimizer":      "AdamW",
        "lr0":            0.001,
        "lrf":            0.01,
        "momentum":       0.937,
        "weight_decay":   0.0005,
        "warmup_epochs":  3.0,
        "warmup_momentum": 0.8,

        # Augmentation — strong for drone imagery
        "augment":        True,
        "flipud":         0.5,         # Aerial views: vertical flip is valid
        "fliplr":         0.5,
        "mosaic":         1.0,
        "mixup":          0.15,
        "copy_paste":     0.3,
        "scale":          0.5,
        "hsv_h":          0.015,
        "hsv_s":          0.7,
        "hsv_v":          0.4,
        "degrees":        5.0,         # Light rotation (drone pitch/yaw variation)
        "translate":      0.1,
        "shear":          2.0,
        "perspective":    0.0001,

        # Loss weights — tuned for small objects in aerial imagery
        "box":            7.5,
        "cls":            0.5,
        "dfl":            1.5,

        # Validation
        "val":            True,
        "plots":          True,
        "verbose":        True,
    }

    print("[Train] Starting YOLOv11x training...")
    results = model.train(**train_kwargs)

    # Save training summary
    summary = {
        "model": args.model,
        "dataset": args.data,
        "epochs_completed": args.epochs,
        "best_model_path": str(Path(args.output) / "visdrone_yolo11x" / "weights" / "best.pt"),
        "metrics": {
            "mAP50": getattr(results, "box", {}).get("map50", None),
            "mAP50_95": getattr(results, "box", {}).get("map", None),
        }
    }
    summary_path = Path(args.output) / "visdrone_yolo11x" / "training_summary.json"
    os.makedirs(summary_path.parent, exist_ok=True)
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2)

    best_weights = Path(args.output) / "visdrone_yolo11x" / "weights" / "best.pt"
    print(f"\n[Train] ✓ Training complete!")
    print(f"  Best weights : {best_weights}")
    print(f"  Summary      : {summary_path}")
    print(f"\n  To use in AERO MESH pipeline, set:")
    print(f"  DETECTOR_WEIGHTS = r'{best_weights}'")

    return str(best_weights)


def main():
    args = parse_args()

    # Step 1: Dataset conversion
    print("[Train] Step 1: Converting VisDrone dataset to YOLO format...")
    try:
        conv_result = run_conversion()
        print(f"[Train] Dataset ready: {conv_result['n_train']} train, "
              f"{conv_result['n_val']} val images")
    except Exception as e:
        print(f"[Train] Dataset conversion error: {e}")
        print("[Train] Attempting to use existing converted dataset...")

    if args.convert_only:
        print("[Train] --convert-only flag set. Skipping training.")
        return

    # Step 2: Train
    best_weights = run_training(args)
    return best_weights


if __name__ == "__main__":
    main()
