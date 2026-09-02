"""
AERO MESH — YOLOv11m Fast Training Script
Optimised to train on an RTX 5060 (8GB VRAM) within 8-9 hours.
Uses YOLOv11m (Medium), 640px resolution, mixed precision, and larger batches.
Supports pausing and resuming training seamlessly.

Usage:
  # Start fresh training:
  python aero_mesh/training/train_detector_fast.py

  # Resume training from a previous run (e.g. after a break):
  python aero_mesh/training/train_detector_fast.py --resume path/to/last.pt
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_YAML   = r"E:\Projects\SIH\yolo_visdrone\dataset.yaml"
DEFAULT_MODEL  = "yolo11m.pt"       # Medium model: Much faster, 8GB VRAM friendly
DEFAULT_EPOCHS = 100
DEFAULT_BATCH  = 16                 # 16 is optimal for 8GB VRAM @ 640px
DEFAULT_IMGSZ  = 640                # Faster training resolution
DEFAULT_DEVICE = "0"
DEFAULT_OUTPUT = r"E:\Projects\SIH\aero_mesh\training\runs"
DEFAULT_WORKERS = 8                 # Maximize CPU usage to feed the GPU faster


def parse_args():
    parser = argparse.ArgumentParser(description="Fast YOLOv11m Training (8GB VRAM)")
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--data",    default=DEFAULT_YAML)
    parser.add_argument("--epochs",  type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch",   type=int, default=DEFAULT_BATCH)
    parser.add_argument("--imgsz",   type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--output",  default=DEFAULT_OUTPUT)
    parser.add_argument("--resume",  type=str, default="",
                        help="Path to last.pt to resume training from")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Train] ERROR: ultralytics not installed.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"AERO MESH — Fast Training Mode (RTX 5060 / 8GB VRAM)")
    print(f"{'=' * 60}")
    
    # Check if we are resuming
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"[Error] Cannot find resume weights at {args.resume}")
            sys.exit(1)
            
        print(f"[*] RESUMING TRAINING FROM: {args.resume}")
        model = YOLO(args.resume)
        
        # When resuming, Ultralytics automatically loads the previous epochs and optimizer state
        results = model.train(resume=True)
        print("\n[Train] ✓ Resumed training complete!")
        return

    # Fresh Training
    print(f"[*] STARTING FRESH TRAINING")
    print(f"  Model     : {args.model}")
    print(f"  Image size: {args.imgsz}px (Optimised for speed)")
    print(f"  Batch     : {args.batch} (Fills 8GB VRAM efficiently)")
    print(f"{'=' * 60}\n")

    model = YOLO(args.model)

    train_kwargs = {
        "data":           args.data,
        "epochs":         args.epochs,
        "batch":          args.batch,
        "imgsz":          args.imgsz,
        "device":         DEFAULT_DEVICE,
        "workers":        DEFAULT_WORKERS,
        "project":        args.output,
        "name":           "visdrone_fast",
        "patience":       20,           # Stop early if no improvement for 20 epochs
        
        # Optimizations for RTX 5060 8GB
        "amp":            True,         # Automatic Mixed Precision (faster FP16 training)
        "cache":          False,        # Set to True if you have >32GB System RAM, otherwise False
        "optimizer":      "auto",
        
        # Slightly relaxed augmentation for faster convergence
        "augment":        True,
        "mosaic":         1.0,
        "mixup":          0.0,
        "copy_paste":     0.0,
        "degrees":        0.0,
        "scale":          0.5,
        "flipud":         0.5,
        "fliplr":         0.5,
    }

    results = model.train(**train_kwargs)
    
    best_weights = Path(args.output) / "visdrone_fast" / "weights" / "best.pt"
    last_weights = Path(args.output) / "visdrone_fast" / "weights" / "last.pt"
    
    print(f"\n[Train] ✓ Fresh training complete!")
    print(f"  Best weights : {best_weights}")
    print(f"  To resume later, run:")
    print(f"  python train_detector_fast.py --resume {last_weights}")


if __name__ == "__main__":
    main()
