"""
AERO MESH — YOLOv11m Fast Training Script (Windows-stable)
Optimised for RTX 5060 Laptop GPU (8GB VRAM / sm_120 Blackwell).
Target: complete within 8-9 hours.

Requires PyTorch Nightly cu128 for sm_120 (Blackwell) GPU support:
  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

Usage:
  # Verify GPU before training:
  python aero_mesh/training/train_detector_fast.py --check

  # Start fresh training:
  python aero_mesh/training/train_detector_fast.py

  # Resume from a checkpoint (safe after Ctrl+C or power loss):
  python aero_mesh/training/train_detector_fast.py --resume

  # Resume from a specific checkpoint file:
  python aero_mesh/training/train_detector_fast.py --resume --resume-path "path/to/last.pt"
"""

import os
import sys
import argparse
from pathlib import Path

# ── CRITICAL: Set BEFORE importing torch ─────────────────────────────────────
# Fixes CUDA OOM fragmentation on 8GB VRAM — must be set before any torch import
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_YAML    = r"E:\Projects\SIH\yolo_visdrone\dataset.yaml"
DEFAULT_MODEL   = "yolo11m.pt"
DEFAULT_EPOCHS  = 100
DEFAULT_BATCH   = 8           # 8 is the safe ceiling for 8GB VRAM + Nightly torch overhead
DEFAULT_IMGSZ   = 640
DEFAULT_OUTPUT  = r"E:\Projects\SIH\aero_mesh\training\runs"
DEFAULT_NAME    = "visdrone_fast"

# ── Windows CRITICAL: workers=0 prevents WinError 1455 (paging file too small)
# Windows multiprocessing spawns new Python processes for each worker.
# Each process loads the full dataset into virtual memory → paging file exhaustion.
# workers=0 uses the main process only — slower data loading but 100% stable on Windows.
DEFAULT_WORKERS = 0


def check_gpu():
    """Full GPU diagnostic — run with --check before training."""
    import torch
    print(f"\n{'=' * 60}")
    print(f"  GPU Diagnostics")
    print(f"{'=' * 60}")
    print(f"  torch version       : {torch.__version__}")
    print(f"  CUDA available      : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        idx  = 0
        name = torch.cuda.get_device_name(idx)
        cap  = torch.cuda.get_device_capability(idx)
        mem  = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        print(f"  GPU name            : {name}")
        print(f"  Compute capability  : sm_{cap[0]}{cap[1]}")
        print(f"  Total VRAM          : {mem:.1f} GB")
        print(f"  ALLOC_CONF          : {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}")
        if cap >= (12, 0):
            print(f"  Status              : ✓ Blackwell sm_120 — cu128 nightly required (installed)")
        else:
            print(f"  Status              : ✓ Compatible with stable torch")
        # Quick tensor test
        try:
            x = torch.ones(1, device="cuda")
            print(f"  Tensor test         : ✓ CUDA tensor creation works")
            del x
        except Exception as e:
            print(f"  Tensor test         : ✗ FAILED — {e}")
    else:
        print(f"  Status              : ✗ No CUDA GPU found")
    print(f"{'=' * 60}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast YOLOv11m Training — RTX 5060 / 8GB VRAM / Windows-stable"
    )
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--data",        default=DEFAULT_YAML)
    parser.add_argument("--epochs",      type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch",       type=int, default=DEFAULT_BATCH)
    parser.add_argument("--imgsz",       type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--output",      default=DEFAULT_OUTPUT)
    parser.add_argument("--check",       action="store_true",
                        help="Run GPU diagnostics only, do not train")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume training from last checkpoint in default run dir")
    parser.add_argument("--resume-path", type=str, default="",
                        help="Explicit path to last.pt to resume from")
    return parser.parse_args()


def get_last_checkpoint(output: str, name: str) -> Path:
    """Find the last.pt checkpoint in the default run directory."""
    return Path(output) / name / "weights" / "last.pt"


def main():
    args = parse_args()

    # Always show GPU info first
    try:
        check_gpu()
    except Exception as e:
        print(f"[GPU Check] Error: {e}")

    if args.check:
        return

    # ── Import after env var is set ────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Train] ERROR: ultralytics not installed.")
        print("  Run: pip install ultralytics")
        sys.exit(1)

    import torch
    if not torch.cuda.is_available():
        print("[Train] ✗ No CUDA GPU detected. Cannot start training.")
        print("  For RTX 5060 (sm_120/Blackwell), install:")
        print("  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128")
        sys.exit(1)

    # ── Resolve dataset YAML ───────────────────────────────────────────────
    if not os.path.exists(args.data):
        print(f"[Train] ✗ Dataset YAML not found: {args.data}")
        print("  Run the converter first:")
        print("  python aero_mesh/dataset/visdrone_converter.py")
        sys.exit(1)

    # ── Resolve resume checkpoint ──────────────────────────────────────────
    resume_path = None
    if args.resume_path:
        resume_path = Path(args.resume_path)
    elif args.resume:
        resume_path = get_last_checkpoint(args.output, DEFAULT_NAME)

    if resume_path is not None:
        if not resume_path.exists():
            print(f"[Train] ✗ Checkpoint not found: {resume_path}")
            print("  No previous training run found. Start fresh (remove --resume).")
            sys.exit(1)
        print(f"\n{'=' * 60}")
        print(f"AERO MESH — Resuming Training")
        print(f"{'=' * 60}")
        print(f"  Checkpoint   : {resume_path}")
        print(f"  Workers      : {DEFAULT_WORKERS} (Windows-safe)")
        print(f"  ALLOC_CONF   : {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
        print(f"{'=' * 60}\n")

        model = YOLO(str(resume_path))
        try:
            # resume=True restores epoch count, optimizer state, LR scheduler, best metrics
            results = model.train(resume=True)
            _print_done(args.output, DEFAULT_NAME)
        except KeyboardInterrupt:
            _print_interrupt(args.output, DEFAULT_NAME)
        return

    # ── Fresh training ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"AERO MESH — Fast Training Mode (RTX 5060 / 8GB VRAM)")
    print(f"{'=' * 60}")
    print(f"  Model        : {args.model}")
    print(f"  Image size   : {args.imgsz}px")
    print(f"  Batch size   : {args.batch}  (safe for 8GB VRAM)")
    print(f"  Epochs max   : {args.epochs} (early-stop patience=20)")
    print(f"  Workers      : {DEFAULT_WORKERS} (0 = Windows-safe single-process)")
    print(f"  AMP          : enabled (FP16 Tensor Cores)")
    print(f"  ALLOC_CONF   : {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
    print(f"  Checkpoints  : saved every epoch → Ctrl+C safe anytime")
    print(f"{'=' * 60}\n")

    model = YOLO(args.model)

    train_kwargs = {
        "data":             args.data,
        "epochs":           args.epochs,
        "batch":            args.batch,
        "imgsz":            args.imgsz,
        "device":           "0",
        "workers":          DEFAULT_WORKERS,   # 0 = no subprocess spawning on Windows
        "project":          args.output,
        "name":             DEFAULT_NAME,
        "exist_ok":         True,              # Reuse same dir, don't create visdrone_fast-2

        # ── Early stopping ─────────────────────────────────────────────────
        "patience":         20,

        # ── Checkpointing: save last.pt every epoch ────────────────────────
        "save":             True,
        "save_period":      1,

        # ── Memory & speed optimizations ───────────────────────────────────
        "amp":              True,              # FP16 AMP on Tensor Cores
        "cache":            False,             # Disable shared memory cache (paging issue)
        "optimizer":        "AdamW",
        "cos_lr":           True,
        "lr0":              0.001,
        "lrf":              0.01,
        "warmup_epochs":    3.0,
        "weight_decay":     0.0005,
        "momentum":         0.937,

        # ── Augmentation (balanced for speed + small-object accuracy) ──────
        "mosaic":           1.0,
        "flipud":           0.5,
        "fliplr":           0.5,
        "scale":            0.5,
        "hsv_h":            0.015,
        "hsv_s":            0.7,
        "hsv_v":            0.4,
        "translate":        0.1,
        "mixup":            0.0,               # Disabled for speed
        "copy_paste":       0.0,               # Disabled for speed
        "degrees":          0.0,               # Disabled for speed

        # ── Loss weights (small-object drone imagery tuned) ────────────────
        "box":              7.5,
        "cls":              0.5,
        "dfl":              1.5,

        # ── Validation & plots ─────────────────────────────────────────────
        "val":              True,
        "plots":            True,
        "verbose":          True,
    }

    try:
        results = model.train(**train_kwargs)
        _print_done(args.output, DEFAULT_NAME)
    except KeyboardInterrupt:
        _print_interrupt(args.output, DEFAULT_NAME)
    except Exception as e:
        print(f"\n[Train] ✗ Training failed: {e}")
        _print_interrupt(args.output, DEFAULT_NAME)
        raise


def _print_done(output_dir: str, name: str):
    best = Path(output_dir) / name / "weights" / "best.pt"
    last = Path(output_dir) / name / "weights" / "last.pt"
    print(f"\n{'=' * 60}")
    print(f"[Train] ✓ Training complete!")
    print(f"  Best weights : {best}")
    print(f"  Last weights : {last}")
    print(f"\n  To resume from a break:")
    print(f"  python train_detector_fast.py --resume")
    print(f"{'=' * 60}")


def _print_interrupt(output_dir: str, name: str):
    last = Path(output_dir) / name / "weights" / "last.pt"
    print(f"\n[Train] ⚡ Interrupted — checkpoint saved at:")
    print(f"  {last}")
    print(f"\n  Resume when ready:")
    print(f"  python train_detector_fast.py --resume")


if __name__ == "__main__":
    main()
