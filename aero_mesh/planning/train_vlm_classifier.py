"""
AERO MESH — VLM CLIP Architectural Classifier Training
=======================================================
Fine-tunes a CLIP image encoder to classify aerial building image patches
into 7 architectural types for the Shape Grammar Engine's asset selection.

Architectural types:
  0: high_rise_residential
  1: office_tower
  2: commercial_shopfront
  3: industrial_warehouse
  4: villa_house
  5: apartment_block
  6: mixed_use

The fine-tuning adds a lightweight MLP classification head on top of
CLIP's frozen ViT image encoder. Only the head (and optionally the last
few encoder layers) are trained — making this very fast (~2 hours on GPU).

CHECKPOINTING & RESUME:
  - Saves checkpoint every --save-every epochs
  - Saves 'best.pt' when validation accuracy improves
  - Ctrl+C (SIGINT) saves checkpoint before exit
  - Resume: python train_vlm_classifier.py --resume

DATASET:
  Provide aerial building image crops (224×224 RGB, any common format).
  Directory structure:
    --dataset path/to/building_patches/
      high_rise_residential/  (*.jpg, *.png)
      office_tower/
      commercial_shopfront/
      industrial_warehouse/
      villa_house/
      apartment_block/
      mixed_use/

  If --dataset is not provided, uses synthetic coloured-patch data for testing.

Usage:
  # Train with your dataset:
  python aero_mesh/planning/train_vlm_classifier.py --dataset path/to/patches/

  # Resume:
  python aero_mesh/planning/train_vlm_classifier.py --dataset path/to/patches/ --resume

  # Synthetic test (no dataset needed):
  python aero_mesh/planning/train_vlm_classifier.py --epochs 5

  # Inference test after training:
  python aero_mesh/planning/train_vlm_classifier.py --infer path/to/image.jpg
"""

import os
import sys
import json
import signal
import argparse
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ── Class Labels ──────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "high_rise_residential",
    "office_tower",
    "commercial_shopfront",
    "industrial_warehouse",
    "villa_house",
    "apartment_block",
    "mixed_use",
]
NUM_CLASSES = len(CLASS_NAMES)

# Text prompts used for zero-shot CLIP reference
CLIP_CLASS_PROMPTS = [
    "aerial view of a high-rise residential apartment building",
    "aerial view of a tall office or corporate tower",
    "aerial view of commercial shops and storefronts",
    "aerial view of an industrial warehouse or factory",
    "aerial view of a detached house or villa",
    "aerial view of a mid-rise apartment block",
    "aerial view of a mixed-use building with shops and flats",
]


def parse_args():
    p = argparse.ArgumentParser(description="Train VLM CLIP Architectural Classifier")
    p.add_argument("--dataset",    type=str, default=None,
                   help="Path to dataset root with class subdirectories")
    p.add_argument("--base-model", type=str,
                   default="openai/clip-vit-base-patch32",
                   help="HuggingFace CLIP model name")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch",      type=int,   default=32)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--save-dir",   type=str,   default="checkpoints/vlm")
    p.add_argument("--save-every", type=int,   default=5)
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--ckpt",       type=str,   default=None)
    p.add_argument("--device",     type=str,   default="auto")
    p.add_argument("--n-synthetic",type=int,   default=2000,
                   help="Synthetic samples per class if no dataset provided")
    p.add_argument("--infer",      type=str,   default=None,
                   help="Path to image to classify (inference mode)")
    return p.parse_args()


# ── Classification Head ───────────────────────────────────────────────────────

def build_classifier_head(embed_dim: int, num_classes: int):
    """Lightweight MLP on top of CLIP image embeddings."""
    try:
        import torch.nn as nn
        return nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
    except ImportError:
        return None


# ── Synthetic Dataset ─────────────────────────────────────────────────────────

def generate_synthetic_patches(n_per_class: int, img_size: int = 224):
    """
    Generate synthetic coloured image patches for testing the training pipeline.
    Each class gets a distinct dominant colour palette.
    Returns: (N, 3, H, W) float32, labels (N,) int
    """
    np.random.seed(0)
    colours = [
        [0.7, 0.4, 0.3],  # high_rise_residential — warm red-brown
        [0.3, 0.5, 0.8],  # office_tower — cool blue-grey
        [0.8, 0.7, 0.2],  # commercial_shopfront — warm yellow
        [0.5, 0.5, 0.5],  # industrial_warehouse — neutral grey
        [0.4, 0.7, 0.4],  # villa_house — green (garden)
        [0.6, 0.5, 0.7],  # apartment_block — muted purple
        [0.7, 0.6, 0.5],  # mixed_use — beige
    ]
    patches, labels = [], []
    for cls_idx, col in enumerate(colours):
        for _ in range(n_per_class):
            base = np.array(col, dtype=np.float32)
            noise = np.random.randn(3, img_size, img_size).astype(np.float32) * 0.08
            patch = (base[:, None, None] + noise).clip(0, 1)
            patches.append(patch)
            labels.append(cls_idx)
    return np.stack(patches), np.array(labels, dtype=np.int64)


# ── Dataset Loader ────────────────────────────────────────────────────────────

def load_real_dataset(dataset_dir: str, processor, img_size: int = 224):
    """
    Load aerial building patch dataset from class subdirectories.
    Applies CLIP image preprocessing.
    Supports standard class names as well as AID dataset directory names.
    Returns: (features_np, labels_np) as raw pixel tensors
    """
    try:
        from PIL import Image
        import torch
    except ImportError:
        return None, None

    features, labels = [], []
    dataset_path = Path(dataset_dir)
    if (dataset_path / "data").exists() and (dataset_path / "data").is_dir() and not (dataset_path / CLASS_NAMES[0]).exists():
        dataset_path = dataset_path / "data"

    AID_ALIASES = {
        "high_rise_residential": ["high_rise_residential", "DenseResidential"],
        "office_tower":          ["office_tower", "Center"],
        "commercial_shopfront":  ["commercial_shopfront", "Commercial"],
        "industrial_warehouse":  ["industrial_warehouse", "Industrial"],
        "villa_house":           ["villa_house", "SparseResidential"],
        "apartment_block":       ["apartment_block", "MediumResidential"],
        "mixed_use":             ["mixed_use", "School", "Square"],
    }

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        aliases = AID_ALIASES.get(cls_name, [cls_name])
        img_files = []
        found_dirs = []
        for alias in aliases:
            cls_dir = dataset_path / alias
            if cls_dir.exists():
                found_dirs.append(alias)
                for ext in ("*.jpg", "*.png", "*.jpeg", "*.webp"):
                    img_files.extend(list(cls_dir.glob(ext)))
        
        if not img_files:
            print(f"[VLM] Warning: no images found for class '{cls_name}' (searched: {aliases})")
            continue
        
        print(f"[VLM] Loading {len(img_files)} images for class '{cls_name}' (from {found_dirs})...")
        for img_path in img_files:
            try:
                img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
                arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
                features.append(arr.transpose(2, 0, 1))         # (3, H, W)
                labels.append(cls_idx)
            except Exception as e:
                print(f"[VLM] Skipping {img_path}: {e}")

    if not features:
        return None, None
    print(f"[VLM] Successfully loaded {len(features)} total real aerial images across classes.")
    return np.stack(features), np.array(labels, dtype=np.int64)


# ── Checkpoint Helpers ────────────────────────────────────────────────────────

def find_latest_checkpoint(save_dir: str):
    d = Path(save_dir)
    latest = d / "latest.pt"
    if latest.exists():
        return str(latest)
    ckpts = sorted(d.glob("ckpt_ep*.pt"))
    return str(ckpts[-1]) if ckpts else None


def save_checkpoint(
    head, optimizer, scheduler, clip_model,
    epoch, best_acc, config, save_dir
) -> str:
    import torch
    d = Path(save_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"ckpt_ep{epoch:03d}.pt"
    data = {
        "epoch":           epoch,
        "head_state":      head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_acc":        best_acc,
        "config":          config,
        "class_names":     CLASS_NAMES,
    }
    # Also save CLIP encoder state for any fine-tuned layers
    if clip_model is not None:
        data["clip_state"] = clip_model.state_dict()
    torch.save(data, path)
    torch.save(data, d / "latest.pt")
    return str(path)


# ── Inference Helper ──────────────────────────────────────────────────────────

def classify_image(image_path: str, ckpt_path: str, device: str = "cpu") -> str:
    """Classify a single aerial building image patch."""
    try:
        import torch
        from PIL import Image
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        print("[VLM] Missing: pip install transformers Pillow")
        return "unknown"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    base = cfg.get("base_model", "openai/clip-vit-base-patch32")

    processor = CLIPProcessor.from_pretrained(base)
    clip_model = CLIPModel.from_pretrained(base).to(device)
    embed_dim  = clip_model.config.projection_dim

    head = build_classifier_head(embed_dim, NUM_CLASSES).to(device)
    head.load_state_dict(ckpt["head_state"])
    head.eval()
    clip_model.eval()

    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feats  = clip_model.get_image_features(**inputs)
        logits = head(feats)
        pred   = logits.argmax(-1).item()

    class_name = CLASS_NAMES[pred]
    print(f"[VLM] Predicted class: {class_name}")
    return class_name


# ── Main Training ─────────────────────────────────────────────────────────────

def train(args):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("[train_vlm] PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    # Optional CLIP
    clip_model  = None
    processor   = None
    embed_dim   = 512   # CLIP ViT-B/32 image embedding dim

    try:
        from transformers import CLIPProcessor, CLIPModel
        processor  = CLIPProcessor.from_pretrained(args.base_model)
        clip_model_name = args.base_model
        _CLIP_AVAILABLE = True
    except Exception as e:
        print(f"[train_vlm] CLIP not available ({e}). Using synthetic RGB features.")
        _CLIP_AVAILABLE = False
        embed_dim = 3 * 224 * 224  # Fallback: flat pixel features

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device

    print(f"\n{'='*60}")
    print(f"  AERO MESH — VLM CLIP Architectural Classifier Training")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    print(f"  CLIP model  : {args.base_model if _CLIP_AVAILABLE else 'N/A (synthetic)'}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Classes     : {NUM_CLASSES} ({', '.join(CLASS_NAMES[:3])}…)")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.dataset and Path(args.dataset).exists() and _CLIP_AVAILABLE:
        print(f"[train_vlm] Loading real dataset from {args.dataset}…")
        raw_features, raw_labels = load_real_dataset(args.dataset, processor)
        if raw_features is None:
            print("[train_vlm] No images found in dataset. Using synthetic data.")
            raw_features, raw_labels = generate_synthetic_patches(args.n_synthetic)
    else:
        if args.dataset:
            print(f"[train_vlm] Dataset dir not found or CLIP unavailable. "
                  f"Using {args.n_synthetic}×{NUM_CLASSES} synthetic samples.")
        else:
            print(f"[train_vlm] No dataset provided. "
                  f"Using {args.n_synthetic}×{NUM_CLASSES} synthetic samples.")
        raw_features, raw_labels = generate_synthetic_patches(args.n_synthetic)

    # ── Extract CLIP embeddings ────────────────────────────────────────────────
    if _CLIP_AVAILABLE and raw_features.shape[1] == 3 and raw_features.shape[2] == 224:
        print("[train_vlm] Extracting CLIP image embeddings…")
        from transformers import CLIPModel
        clip_model = CLIPModel.from_pretrained(args.base_model).to(device)
        clip_model.eval()
        embed_dim  = clip_model.config.projection_dim
        all_embeds = []
        bs = 64
        with torch.no_grad():
            for i in range(0, len(raw_features), bs):
                batch = torch.from_numpy(raw_features[i:i+bs]).float().to(device)
                # Normalise to [0,1] then convert to CLIP expected range
                pixel_values = batch
                feats = clip_model.vision_model(pixel_values=pixel_values).pooler_output
                feats = clip_model.visual_projection(feats)
                all_embeds.append(feats.cpu().numpy())
        features = np.vstack(all_embeds).astype(np.float32)
        print(f"[train_vlm] CLIP embeddings extracted: {features.shape}")
        # Freeze CLIP for training (only train head)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
    else:
        # Fallback: flatten raw pixel features
        features = raw_features.reshape(len(raw_features), -1).astype(np.float32)
        embed_dim = features.shape[1]
        clip_model = None
        print(f"[train_vlm] Using flat pixel features: dim={embed_dim}")

    labels = raw_labels

    # Train/val split
    n_val = max(NUM_CLASSES, int(len(features) * 0.15))
    idx   = np.random.permutation(len(features))
    tr_idx, val_idx = idx[n_val:], idx[:n_val]

    X_tr  = torch.from_numpy(features[tr_idx]).float()
    y_tr  = torch.from_numpy(labels[tr_idx]).long()
    X_val = torch.from_numpy(features[val_idx]).float()
    y_val = torch.from_numpy(labels[val_idx]).long()

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=args.batch, shuffle=True
    )
    val_loader   = DataLoader(
        TensorDataset(X_val, y_val), batch_size=args.batch * 4
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    head      = build_classifier_head(embed_dim, NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    loss_fn   = nn.CrossEntropyLoss()

    start_epoch = 0
    best_acc    = 0.0
    config      = vars(args)
    config["class_names"] = CLASS_NAMES

    # ── Resume ────────────────────────────────────────────────────────────────
    ckpt_path = args.ckpt or (find_latest_checkpoint(args.save_dir) if args.resume else None)
    if ckpt_path and Path(ckpt_path).exists():
        print(f"[train_vlm] Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        head.load_state_dict(ckpt["head_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc    = ckpt.get("best_acc", 0.0)
        print(f"[train_vlm] Resuming from epoch {start_epoch}, best_acc={best_acc:.3f}")
    elif args.resume:
        print("[train_vlm] --resume set but no checkpoint found. Starting fresh.")

    # ── Ctrl+C Handler ────────────────────────────────────────────────────────
    _current_epoch = [start_epoch]
    _current_acc   = [0.0]

    def handle_sigint(sig, frame):
        print(f"\n[train_vlm] Ctrl+C — saving checkpoint at epoch "
              f"{_current_epoch[0]}…")
        save_checkpoint(
            head, optimizer, scheduler, None,
            _current_epoch[0], best_acc, config, args.save_dir
        )
        print("[train_vlm] Saved. Exiting cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    # ── Training Loop ─────────────────────────────────────────────────────────
    print(f"[train_vlm] Training head on {len(X_tr)} samples, "
          f"val on {len(X_val)} samples.")
    for epoch in range(start_epoch, args.epochs):
        _current_epoch[0] = epoch
        head.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = loss_fn(head(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validation accuracy
        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = head(X_b).argmax(-1)
                correct += (preds == y_b).sum().item()
                total   += len(y_b)
        acc = correct / max(total, 1)
        _current_acc[0] = acc

        print(f"  Epoch {epoch+1:03d}/{args.epochs}  val_acc={acc:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if acc > best_acc:
            best_acc  = acc
            best_path = Path(args.save_dir) / "best.pt"
            save_checkpoint(head, optimizer, scheduler, None,
                            epoch, best_acc, config, args.save_dir)
            import shutil
            shutil.copy(Path(args.save_dir) / "latest.pt", best_path)
            print(f"  ★ New best! acc={best_acc:.3f} → {best_path}")

        if (epoch + 1) % args.save_every == 0:
            ckpt_saved = save_checkpoint(
                head, optimizer, scheduler, None,
                epoch, best_acc, config, args.save_dir
            )
            print(f"  ✓ Checkpoint → {ckpt_saved}")

    save_checkpoint(head, optimizer, scheduler, None,
                    args.epochs - 1, best_acc, config, args.save_dir)
    print(f"\n[train_vlm] Training complete!")
    print(f"  Best val_acc  : {best_acc:.3f}")
    print(f"  Best model    : {Path(args.save_dir) / 'best.pt'}")
    print(f"\n  To classify an image:")
    print(f"    python aero_mesh/planning/train_vlm_classifier.py "
          f"--infer path/to/image.jpg")


if __name__ == "__main__":
    args = parse_args()
    if args.infer:
        ckpt = args.ckpt or str(Path(args.save_dir) / "best.pt")
        classify_image(args.infer, ckpt, args.device if args.device != "auto" else "cpu")
    else:
        train(args)
