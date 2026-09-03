"""
update_training_status.py
─────────────────────────
Reads the live YOLO training outputs and writes
  web/public/training_status.json

So every metric shown in the frontend is derived from actual training data,
not hardcoded. Run once to generate the initial file, or loop with --watch.

Usage:
    python scripts/update_training_status.py           # single update
    python scripts/update_training_status.py --watch   # poll every 10 s
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# ── Paths (relative to project root, discovered from this script's location) ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RUNS_DIR     = PROJECT_ROOT / "aero_mesh" / "training" / "runs" / "visdrone_fast"
RESULTS_CSV  = RUNS_DIR / "results.csv"
ARGS_YAML    = RUNS_DIR / "args.yaml"
WEIGHTS_DIR  = RUNS_DIR / "weights"
DATASET_YAML = PROJECT_ROOT / "yolo_visdrone" / "dataset.yaml"
OUT_JSON     = PROJECT_ROOT / "web" / "public" / "training_status.json"


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_args_yaml(path: Path) -> dict:
    """Parse YAML key: value pairs without requiring PyYAML."""
    data: dict = {}
    if not path.exists():
        return data
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, _, v = line.partition(":")
                data[k.strip()] = v.strip()
    return data


def parse_dataset_yaml(path: Path) -> dict:
    """Extract nc and names list from dataset YAML."""
    result: dict = {"nc": 0, "names": [], "nTrain": 0, "nVal": 0}
    if not path.exists():
        return result
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    for line in content.splitlines():
        s = line.strip()
        if s.startswith("nc:"):
            try:
                result["nc"] = int(s.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif s.startswith("names:"):
            raw = s.split(":", 1)[1].strip()
            if raw.startswith("["):
                raw = raw.strip("[]")
                result["names"] = [n.strip().strip("'\"") for n in raw.split(",")]
        elif s.startswith("train:"):
            train_path = Path(s.split(":", 1)[1].strip())
            if train_path.exists():
                result["nTrain"] = sum(1 for f in train_path.iterdir()
                                       if f.suffix.lower() in (".jpg", ".png", ".jpeg"))
        elif s.startswith("val:"):
            val_path = Path(s.split(":", 1)[1].strip())
            if val_path.exists():
                result["nVal"] = sum(1 for f in val_path.iterdir()
                                     if f.suffix.lower() in (".jpg", ".png", ".jpeg"))
    return result


def parse_results_csv(path: Path) -> dict:
    """Read the last row of results.csv and return a metrics dict."""
    empty = {
        "epoch": 0, "totalEpochs": 100,
        "mAP50": 0.0, "mAP5095": 0.0,
        "precision": 0.0, "recall": 0.0,
        "trainLoss": 0.0, "valLoss": 0.0,
        "trainingMin": 0.0,
    }
    if not path.exists():
        return empty

    rows: list[list[str]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    if len(rows) < 2:
        return empty

    header = [h.strip() for h in rows[0]]
    last   = [v.strip() for v in rows[-1]]

    def col(name: str, default: float = 0.0) -> float:
        try:
            idx = next(i for i, h in enumerate(header) if name in h)
            return float(last[idx])
        except (StopIteration, ValueError, IndexError):
            return default

    return {
        "epoch":       int(col("epoch")),
        "trainingMin": col("time") / 60.0,
        "trainLoss":   col("train/box_loss"),
        "precision":   col("precision"),
        "recall":      col("recall"),
        "mAP50":       col("mAP50(B)"),
        "mAP5095":     col("mAP50-95(B)"),
        "valLoss":     col("val/box_loss"),
    }


def count_epochs_completed() -> int:
    """Count epoch checkpoints in weights dir."""
    if not WEIGHTS_DIR.exists():
        return 0
    return sum(1 for f in WEIGHTS_DIR.iterdir()
               if f.name.startswith("epoch") and f.suffix == ".pt")


def find_weights() -> dict[str, str]:
    """Return paths to best.pt and last.pt if they exist."""
    result: dict[str, str] = {"bestWeights": "", "lastWeights": ""}
    best = WEIGHTS_DIR / "best.pt"
    last = WEIGHTS_DIR / "last.pt"
    if best.exists():
        result["bestWeights"] = str(best)
    if last.exists():
        result["lastWeights"] = str(last)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

def build_status() -> dict:
    """Assemble the full training_status.json payload."""
    metrics  = parse_results_csv(RESULTS_CSV)
    args     = parse_args_yaml(ARGS_YAML)
    dataset  = parse_dataset_yaml(DATASET_YAML)
    weights  = find_weights()

    # Extract model name from args.yaml (key: model)
    model_raw = args.get("model", "yolo11m.pt")
    # Strip to basename without extension for display
    model_name = Path(model_raw.strip("\"'")).stem if model_raw else "yolo11m"

    # Total epochs from args.yaml
    try:
        total_epochs = int(args.get("epochs", "100"))
    except ValueError:
        total_epochs = 100

    # Image size
    try:
        imgsz = int(args.get("imgsz", "640"))
    except ValueError:
        imgsz = 640

    # Batch size
    try:
        batch = int(args.get("batch", "8"))
    except ValueError:
        batch = 8

    metrics["totalEpochs"] = total_epochs

    status = {
        # ── Identity ──────────────────────────────────────────────────────
        "projectName":   "AERO MESH — VisDrone Dynamic Object Detection",
        "problemStatement": "SIH PS #26158",
        "modelName":     model_name,
        "imgsz":         imgsz,
        "batch":         batch,

        # ── Dataset ───────────────────────────────────────────────────────
        "datasetName":   "VisDrone2019-DET",
        "datasetYaml":   str(DATASET_YAML),
        "nc":            dataset["nc"],
        "classNames":    dataset["names"],
        "nTrain":        dataset["nTrain"],
        "nVal":          dataset["nVal"],

        # ── Weights ───────────────────────────────────────────────────────
        **weights,
        "epochsCompleted": count_epochs_completed(),

        # ── Live Metrics (from results.csv last row) ───────────────────
        **metrics,

        # ── Meta ──────────────────────────────────────────────────────────
        "updatedAt":     time.strftime("%Y-%m-%dT%H:%M:%S"),
        "csvExists":     RESULTS_CSV.exists(),
        "weightsExist":  (WEIGHTS_DIR / "last.pt").exists(),
    }
    return status


def write_status() -> dict:
    """Build and write the JSON. Returns the payload."""
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = build_status()
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    epoch = payload.get("epoch", 0)
    total = payload.get("totalEpochs", 100)
    mAP   = payload.get("mAP50", 0) * 100
    print(f"[Status] Written -> {OUT_JSON}")
    print(f"         Epoch {epoch}/{total} | mAP50={mAP:.1f}% | "
          f"model={payload['modelName']} | classes={payload['nc']}")
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Update training_status.json")
    parser.add_argument("--watch", action="store_true",
                        help="Poll every 10s while training runs")
    parser.add_argument("--interval", type=int, default=10,
                        help="Poll interval in seconds (with --watch)")
    args = parser.parse_args()

    write_status()

    if args.watch:
        print(f"[Status] Watching every {args.interval}s ... (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(args.interval)
                write_status()
        except KeyboardInterrupt:
            print("[Status] Stopped.")


if __name__ == "__main__":
    main()
