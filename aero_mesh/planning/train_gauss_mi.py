"""
AERO MESH — GauSS-MI Uncertainty Head Training Script
======================================================
Trains the GaussianUncertaintyHead MLP that predicts per-Gaussian
uncertainty from view history features.

CHECKPOINTING & RESUME:
  - Saves checkpoint every --save-every epochs to --save-dir
  - Also saves 'best.pt' when validation loss improves
  - Ctrl+C (SIGINT) is handled: writes final checkpoint before exit
  - Resume from latest checkpoint: python train_gauss_mi.py --resume
  - Resume from specific: python train_gauss_mi.py --resume --ckpt path/to/ckpt.pt

DATASET:
  The trainer uses synthetic training data generated from pairs of:
    - Gaussian states (position, scale, rotation, opacity, SH variance)
    - Ground-truth uncertainty labels derived from multi-view colour consistency
  
  To use your own data, provide a directory with .npy feature/label files:
    --dataset path/to/gauss_mi_data/
      features.npy   → (N, 20) float32
      labels.npy     → (N,) float32 in [0, 1]

Usage:
  # Generate synthetic data and train:
  python aero_mesh/planning/train_gauss_mi.py

  # Resume from latest checkpoint:
  python aero_mesh/planning/train_gauss_mi.py --resume

  # Resume from specific checkpoint:
  python aero_mesh/planning/train_gauss_mi.py --resume --ckpt checkpoints/gauss_mi/ckpt_ep020.pt

  # Full options:
  python aero_mesh/planning/train_gauss_mi.py \\
      --epochs 50 --lr 1e-3 --batch 256 \\
      --save-dir checkpoints/gauss_mi/ \\
      --save-every 5
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

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train GauSS-MI Uncertainty Head")
    p.add_argument("--dataset",    type=str, default=None,
                   help="Path to dataset dir with features.npy + labels.npy. "
                        "If not provided, uses synthetic data.")
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--batch",      type=int, default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--save-dir",   type=str, default="checkpoints/gauss_mi")
    p.add_argument("--save-every", type=int, default=5,
                   help="Save checkpoint every N epochs")
    p.add_argument("--resume",     action="store_true",
                   help="Resume from latest checkpoint in --save-dir")
    p.add_argument("--ckpt",       type=str, default=None,
                   help="Path to specific checkpoint to resume from")
    p.add_argument("--n-synthetic",type=int, default=100_000,
                   help="Number of synthetic samples to generate if no dataset provided")
    p.add_argument("--val-split",  type=float, default=0.1,
                   help="Fraction of data held out for validation")
    p.add_argument("--device",     type=str, default="auto")
    return p.parse_args()


# ── Synthetic Data Generation ─────────────────────────────────────────────────

def generate_synthetic_data(n: int) -> tuple:
    """
    Generate synthetic (features, labels) for GauSS-MI training.

    Features (N, 20):
      positions(3), scales(3), rotations(4), view_count(1), opacity(1),
      colour_var(3), sh_var(3), mean_colour_var(1), barely_seen(1)

    Labels (N,): uncertainty in [0, 1]
      High uncertainty when:
        - view_count < 3 (barely seen from any angle)
        - colour_var > 0.2 (appearance is inconsistent across views)
        - scale is large (Gaussian represents large uncertain region)
        - opacity is low (floater / ghost Gaussian)
    """
    np.random.seed(42)
    positions   = np.random.randn(n, 3).astype(np.float32) * 10.0
    scales      = np.abs(np.random.randn(n, 3)).astype(np.float32) * 0.1
    rotations   = (np.random.randn(n, 4).astype(np.float32))
    rotations  /= np.linalg.norm(rotations, axis=1, keepdims=True)
    view_count  = np.random.exponential(3, (n, 1)).astype(np.float32)
    view_count  = np.clip(view_count / 20.0, 0, 1)
    opacity     = np.random.beta(2, 2, (n, 1)).astype(np.float32)
    colour_var  = np.abs(np.random.randn(n, 3)).astype(np.float32) * 0.1
    sh_var      = np.abs(np.random.randn(n, 3)).astype(np.float32) * 0.05
    mean_cvar   = colour_var.mean(axis=1, keepdims=True)
    barely_seen = (view_count < 0.15).astype(np.float32)

    features = np.concatenate([
        positions / 100.0, scales * 10.0, rotations,
        view_count, opacity, colour_var, sh_var, mean_cvar, barely_seen
    ], axis=1).astype(np.float32)

    # Ground-truth uncertainty label
    raw_vc   = view_count.squeeze() * 20.0
    vc_raw   = np.random.exponential(3, n)
    labels   = np.zeros(n, dtype=np.float32)
    labels  += 0.40 * (1.0 / (1.0 + vc_raw / 5.0))            # view-count contribution
    labels  += 0.35 * colour_var.mean(axis=1)                  # appearance variance
    labels  += 0.15 * scales.mean(axis=1)                      # scale contribution
    labels  += 0.10 * (1.0 - opacity.squeeze())                # opacity contribution
    labels  += np.random.randn(n).astype(np.float32) * 0.02   # noise
    labels   = np.clip(labels, 0, 1).astype(np.float32)

    return features[:, :20], labels


# ── Checkpoint Helpers ────────────────────────────────────────────────────────

def find_latest_checkpoint(save_dir: str) -> str | None:
    """Return path to the latest checkpoint, or None if none exist."""
    d = Path(save_dir)
    if not d.exists():
        return None
    latest = d / "latest.pt"
    if latest.exists():
        return str(latest)
    ckpts = sorted(d.glob("ckpt_ep*.pt"))
    return str(ckpts[-1]) if ckpts else None


def save_checkpoint(
    model, optimizer, scheduler, epoch: int,
    val_loss: float, best_val: float, config: dict, save_dir: str
) -> str:
    import torch
    d = Path(save_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"ckpt_ep{epoch:03d}.pt"
    data = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "val_loss":        val_loss,
        "best_val":        best_val,
        "config":          config,
    }
    torch.save(data, path)
    torch.save(data, d / "latest.pt")
    return str(path)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("[train_gauss_mi] PyTorch not installed. Run:")
        print("  pip install torch")
        sys.exit(1)

    from aero_mesh.planning.gauss_mi import GaussianUncertaintyHead

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    print(f"\n{'='*60}")
    print(f"  AERO MESH — GauSS-MI Uncertainty Head Training")
    print(f"{'='*60}")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch     : {args.batch}")
    print(f"  LR        : {args.lr}")
    print(f"  Save Dir  : {args.save_dir}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.dataset and Path(args.dataset).exists():
        feat_path = Path(args.dataset) / "features.npy"
        lbl_path  = Path(args.dataset) / "labels.npy"
        if feat_path.exists() and lbl_path.exists():
            features = np.load(str(feat_path))
            labels   = np.load(str(lbl_path))
            print(f"[train_gauss_mi] Loaded dataset: {features.shape}")
        else:
            print("[train_gauss_mi] Dataset dir missing features.npy or labels.npy. "
                  "Using synthetic data.")
            features, labels = generate_synthetic_data(args.n_synthetic)
    else:
        print(f"[train_gauss_mi] Generating {args.n_synthetic} synthetic samples…")
        features, labels = generate_synthetic_data(args.n_synthetic)
        print(f"[train_gauss_mi] Synthetic data generated: {features.shape}")

    # Train/val split
    n_val = max(1, int(len(features) * args.val_split))
    idx   = np.random.permutation(len(features))
    train_idx, val_idx = idx[n_val:], idx[:n_val]

    X_tr = torch.from_numpy(features[train_idx]).float()
    y_tr = torch.from_numpy(labels[train_idx]).float().unsqueeze(1)
    X_val = torch.from_numpy(features[val_idx]).float()
    y_val = torch.from_numpy(labels[val_idx]).float().unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=args.batch, shuffle=True, drop_last=False
    )
    val_loader   = DataLoader(
        TensorDataset(X_val, y_val), batch_size=args.batch * 4, shuffle=False
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = GaussianUncertaintyHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    loss_fn   = nn.MSELoss()

    start_epoch = 0
    best_val    = float("inf")
    config      = vars(args)

    # ── Resume ────────────────────────────────────────────────────────────────
    ckpt_path = args.ckpt or (find_latest_checkpoint(args.save_dir) if args.resume else None)
    if ckpt_path and Path(ckpt_path).exists():
        print(f"[train_gauss_mi] Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val    = ckpt.get("best_val", float("inf"))
        print(f"[train_gauss_mi] Resuming from epoch {start_epoch}, best_val={best_val:.6f}")
    elif args.resume:
        print("[train_gauss_mi] --resume specified but no checkpoint found. Starting fresh.")

    # ── Ctrl+C Signal Handler ─────────────────────────────────────────────────
    _interrupted = [False]
    _current_epoch = [start_epoch]
    _current_val_loss = [float("inf")]

    def handle_sigint(sig, frame):
        print(f"\n[train_gauss_mi] Ctrl+C detected! Saving checkpoint at epoch "
              f"{_current_epoch[0]}…")
        save_checkpoint(
            model, optimizer, scheduler,
            _current_epoch[0], _current_val_loss[0], best_val, config, args.save_dir
        )
        print("[train_gauss_mi] Checkpoint saved. Exiting cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    # ── Training Loop ─────────────────────────────────────────────────────────
    print(f"[train_gauss_mi] Training on {len(X_tr)} samples, "
          f"validating on {len(X_val)} samples.")
    for epoch in range(start_epoch, args.epochs):
        _current_epoch[0] = epoch
        model.train()
        train_losses = []
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b)
                val_losses.append(loss_fn(pred, y_b).item())
        val_loss = np.mean(val_losses)
        _current_val_loss[0] = val_loss
        tr_loss  = np.mean(train_losses)

        print(f"  Epoch {epoch+1:03d}/{args.epochs}  "
              f"train_loss={tr_loss:.6f}  val_loss={val_loss:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            best_path = Path(args.save_dir) / "best.pt"
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_loss": val_loss, "config": config,
            }, best_path)
            print(f"  ★ New best model saved → {best_path}")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_saved = save_checkpoint(
                model, optimizer, scheduler,
                epoch, val_loss, best_val, config, args.save_dir
            )
            print(f"  ✓ Checkpoint saved → {ckpt_saved}")

    # Final checkpoint
    save_checkpoint(
        model, optimizer, scheduler,
        args.epochs - 1, val_loss, best_val, config, args.save_dir
    )
    print(f"\n[train_gauss_mi] Training complete!")
    print(f"  Best val_loss : {best_val:.6f}")
    print(f"  Best model    : {Path(args.save_dir) / 'best.pt'}")
    print(f"\n  To use in AERO MESH:")
    print(f"    gmi = GaussMI(checkpoint_path='{Path(args.save_dir) / 'best.pt'}')")


if __name__ == "__main__":
    train(parse_args())
