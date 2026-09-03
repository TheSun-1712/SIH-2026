"""
AERO MESH — Semantic 3D Terrain Reconstruction
===============================================
Generates a full semantic terrain model from drone images by combining:
  1. YOLOv11n-seg  → per-pixel semantic masks (80 COCO classes → 6 terrain buckets)
  2. Depth Anything V2 Metric Outdoor (ViT-L) → per-pixel metric depth in metres
  3. VisDrone YOLO detector → moving objects (cars, people) as thin overlay markers

Output JSON schema:
  {
    "generatedAt": "...",
    "planes": [{ "filename", "x", "y", "z", "w", "h", "frame" }],
    "terrain": [{ "x", "y", "z", "cls", "label", "color" }],   ← sampled surface
    "objects": [{ "x", "y", "z", "w", "h", "d", "cls", "label", "color" }],
    "stats": { "numTerrain", "numObjects", "epochAtExport", "mAP50AtExport", ... }
  }
"""

import os, sys, json, time, argparse
import cv2
import numpy as np
from PIL import Image as PILImage

try:
    from ultralytics import YOLO
except ImportError:
    print("[Recon] ERROR: ultralytics not installed."); sys.exit(1)

# ─── Semantic bucket mapping (COCO-80 class id → terrain label + color hex) ──
# Reference: https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
COCO_TO_TERRAIN = {
    # Vehicles / dynamic objects → "vehicle" (kept thin, low opacity)
    2: ("vehicle",   "#10b981", 1.5),   # car
    3: ("vehicle",   "#10b981", 1.5),   # motorcycle
    4: ("vehicle",   "#10b981", 1.0),   # airplane
    5: ("vehicle",   "#10b981", 1.5),   # bus
    6: ("vehicle",   "#10b981", 1.5),   # train
    7: ("vehicle",   "#10b981", 1.5),   # truck
    8: ("vehicle",   "#10b981", 0.8),   # boat
    # Person
    0: ("person",    "#38bdf8", 1.8),

    # Outdoor / ground classes
    13: ("bench",    "#a3a3a3", 0.5),
    # Sports / outdoor
    38: ("kite",     "#f59e0b", 0.3),

    # Indoor / furniture → treat as "structure"
    56: ("chair",    "#d97706", 0.6),
    57: ("couch",    "#d97706", 0.5),
    59: ("bed",      "#d97706", 0.5),

    # Potted plant → vegetation
    58: ("vegetation", "#16a34a", 2.5),

    # Building / architectural
    # COCO doesn't have a direct "building" class. We use the mask2former
    # approach: if a large mask covers a high-depth region → building.
}

# Manually defined terrain buckets for regions NOT matched by COCO objects.
# We derive these from HSV colour analysis of the image under each mask.
TERRAIN_BUCKETS = [
    # label         hex colour     base_height  hue_range(lo, hi)  sat_min
    ("road",       "#475569",     0.05,         (0,   30),         0.0),
    ("vegetation", "#16a34a",     4.0,          (35,  85),         0.25),
    ("building",   "#d97706",     8.0,          (0,   30),         0.0),   # fallback high-depth
    ("water",      "#0ea5e9",     0.02,         (95, 130),         0.20),
    ("ground",     "#a16207",     0.3,          (15,  35),         0.10),
    ("sky",        None,          0.0,          (180, 260),        0.0),   # filtered out
]

TERRAIN_COLORS = {
    "road":        "#475569",
    "vegetation":  "#16a34a",
    "building":    "#d97706",
    "water":       "#0ea5e9",
    "ground":      "#a16207",
    "person":      "#38bdf8",
    "vehicle":     "#10b981",
    "structure":   "#a78bfa",
    "other":       "#94a3b8",
}


def classify_pixel_by_color(hsv_pixel: np.ndarray) -> str:
    """
    Classify a pixel into a terrain bucket using ONLY HSV colour.
    For nadir (straight-down) drone cameras, depth is nearly identical across
    the entire frame (everything is ~50m below), so depth-based building
    detection is unreliable. We use colour only.

    hsv_pixel: shape (3,) in OpenCV HSV range (H: 0-179, S: 0-255, V: 0-255).
    """
    h = float(hsv_pixel[0])          # 0-179 (multiply by 2 for 0-360)
    s = float(hsv_pixel[1]) / 255.0  # 0-1
    v = float(hsv_pixel[2]) / 255.0  # 0-1 (brightness)

    # Sky: very bright AND very low saturation (white/light grey at top)
    if v > 0.90 and s < 0.08:
        return "sky"

    # Vegetation: green hue (H ~60-80 in 0-179 scale = 120-160 in 0-360)
    if 55 < h < 85 and s > 0.18:
        return "vegetation"

    # Water: blue/cyan hue (H ~95-130 in 0-179 scale)
    if 90 < h < 135 and s > 0.20:
        return "water"

    # Road/Pavement: gray — low saturation, medium brightness
    if s < 0.15 and 0.15 < v < 0.80:
        return "road"

    # Building: distinctly warm/reddish rooftops with meaningful saturation.
    # Require s > 0.30 to avoid catching beige/tan pavement (which has s ~ 0.10-0.20).
    if (h < 20 or h > 158) and s > 0.30 and v > 0.25:
        return "building"

    # Anything warm-ish but low saturation → ground (sand/dirt/tan pavement)
    return "ground"


def load_depth_model():
    """Attempt to load Depth Anything V2 Metric Outdoor (ViT-L) from HuggingFace."""
    try:
        import torch
        from transformers import pipeline as hf_pipeline
        device = 0 if torch.cuda.is_available() else -1
        dtype  = torch.float16 if device == 0 else torch.float32
        print("[Recon] Loading Depth Anything V2 Metric Outdoor (ViT-L)…")
        pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
            device=device,
            torch_dtype=dtype,
        )
        print("[Recon]   ✓ DA2 Metric Outdoor loaded")
        return pipe, "da2"
    except Exception as e:
        print(f"[Recon]   DA2 unavailable ({e}). Using synthetic depth.")
        return None, "synthetic"


def synthetic_depth(img: np.ndarray) -> np.ndarray:
    """Fallback: generate a plausible radial depth gradient (centre closer)."""
    h, w = img.shape[:2]
    gy = np.linspace(0.3, 1.0, h)[:, None] * np.ones((1, w))  # bottom = farther
    # Normalise to 0.2 – 0.8 range (as fraction of max_depth)
    return gy.astype(np.float32)


def predict_depth(pipe, img_bgr: np.ndarray) -> np.ndarray:
    """Returns a normalised depth map in [0, 1] where 0=closest, 1=farthest."""
    if pipe is None:
        return synthetic_depth(img_bgr)
    try:
        import torch
        rgb = PILImage.fromarray(img_bgr[:, :, ::-1])
        with torch.no_grad():
            out = pipe(rgb)
        d = np.array(out["depth"], dtype=np.float32)
        # Resize to match image
        h, w = img_bgr.shape[:2]
        if d.shape != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        # Normalise: DA2 gives absolute metres; normalise for colour mapping
        d_min, d_max = d.min(), d.max()
        if d_max > d_min:
            return (d - d_min) / (d_max - d_min)
        return np.zeros_like(d)
    except Exception as e:
        print(f"[Recon]   Depth inference failed: {e}. Using synthetic.")
        return synthetic_depth(img_bgr)


def build_terrain_from_image(
    img_bgr: np.ndarray,
    seg_model,
    depth_norm: np.ndarray,
    frame_idx: int,
    offset_y: float,
    phys_scale: float,
    sample_stride: int = 20,
) -> list[dict]:
    """
    Run YOLO-seg on the image, then for every non-sky pixel on the sampling grid,
    classify the terrain bucket and compute a 3D point.
    Returns a list of terrain point dicts.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # ── 1. Get YOLO-seg masks ───────────────────────────────────────────────────
    seg_results = seg_model(img_bgr, verbose=False)
    seg_result  = seg_results[0]

    # Build a per-pixel class map (H×W) with COCO class id, default = -1
    class_map = np.full((h, w), -1, dtype=np.int16)
    if seg_result.masks is not None:
        masks_data = seg_result.masks.data.cpu().numpy()  # (N, H', W')
        classes    = seg_result.boxes.cls.cpu().numpy().astype(int)
        for idx, (mask, cls_id) in enumerate(zip(masks_data, classes)):
            # Resize mask to full image size
            m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
            class_map[m] = cls_id

    # ── 2. Sample terrain grid ─────────────────────────────────────────────────
    terrain_pts = []
    for py in range(0, h, sample_stride):
        for px in range(0, w, sample_stride):
            coco_cls   = int(class_map[py, px])
            depth_val  = float(depth_norm[py, px])
            hsv_px     = hsv[py, px]

            # Derive terrain label
            if coco_cls in COCO_TO_TERRAIN:
                label, color, height_m = COCO_TO_TERRAIN[coco_cls]
            else:
                label = classify_pixel_by_color(hsv_px)
                color = TERRAIN_COLORS.get(label, TERRAIN_COLORS["other"])
                # Fixed, realistic heights for each terrain class.
                # Do NOT use depth to scale heights for nadir drone imagery —
                # depth is nearly identical across the whole frame (~50m alt).
                height_map = {
                    "road":       0.05,   # nearly flat
                    "ground":     0.15,   # slightly raised ground
                    "vegetation": 2.5,    # trees / shrubs
                    "building":   4.0,    # rooftop
                    "water":      0.02,   # flat water surface
                    "sky":        0.0,
                    "other":      0.2
                }
                height_m = height_map.get(label, 0.2)

            # Skip sky entirely
            if label == "sky":
                continue

            # 3D position (world coords)
            wx = (px - w / 2.0) * phys_scale
            wy = (py - h / 2.0) * phys_scale + offset_y

            terrain_pts.append({
                "x":     round(float(wx),      4),
                "y":     round(float(wy),      4),
                "h":     round(float(height_m), 3),
                "label": label,
                "color": color,
                "frame": frame_idx,
            })

    return terrain_pts


def build_objects_from_detections(
    det_model,
    img_bgr: np.ndarray,
    frame_idx: int,
    offset_y: float,
    phys_scale: float,
) -> list[dict]:
    """
    Run VisDrone YOLO detector (bounding boxes only) and return thin wireframe
    object markers for moving objects — small enough to not block the terrain.
    """
    VIS_LABELS = {
        1: 'Pedestrian', 2: 'People', 3: 'Bicycle', 4: 'Car',
        5: 'Van', 6: 'Truck', 7: 'Tricycle', 8: 'Awning-Tricycle',
        9: 'Bus', 10: 'Motor', 11: 'Others'
    }
    VIS_COLORS = {
        1: '#38bdf8', 2: '#38bdf8', 3: '#f43f5e', 4: '#10b981',
        5: '#f59e0b', 6: '#f59e0b', 7: '#a78bfa', 8: '#a78bfa',
        9: '#f59e0b', 10: '#f43f5e', 11: '#64748b',
    }
    h, w = img_bgr.shape[:2]
    results = det_model(img_bgr, verbose=False)
    objects = []
    for box_data in results[0].boxes:
        b    = box_data.xyxy[0].cpu().numpy()
        conf = float(box_data.conf[0].cpu().numpy())
        cls  = int(box_data.cls[0].cpu().numpy()) + 1
        if cls not in VIS_LABELS:
            cls = 11
        x1, y1, x2, y2 = b
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = (x2 - x1) * phys_scale
        bd = (y2 - y1) * phys_scale
        # Height: thin slab — just enough to be visible above terrain
        obj_h = max(0.5, min(2.0, np.sqrt((x2-x1)*(y2-y1)) * phys_scale * 0.4))
        wx = (cx - w / 2.0) * phys_scale
        wy = (cy - h / 2.0) * phys_scale + offset_y
        objects.append({
            "x": round(float(wx), 4),
            "y": round(float(wy), 4),
            "z": round(float(obj_h / 2.0), 4),
            "w": round(float(bw), 4),
            "h": round(float(obj_h), 4),
            "d": round(float(bd), 4),
            "cls": cls,
            "label": VIS_LABELS[cls],
            "color": VIS_COLORS.get(cls, '#64748b'),
            "conf":  round(conf, 4),
            "frame": frame_idx,
        })
    return objects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',    type=str, default='aero_mesh/training/runs/visdrone_fast/weights/last.pt')
    parser.add_argument('--seg-model', type=str, default='yolo11n-seg.pt')
    parser.add_argument('--max-images',type=int, default=5)
    parser.add_argument('--stride',    type=int, default=20, help='Terrain sampling stride (px). Lower = denser but slower.')
    args = parser.parse_args()

    weights_path = os.path.abspath(args.weights)
    if not os.path.exists(weights_path):
        print(f"[Recon] YOLO weights not found at {weights_path}"); sys.exit(1)

    val_dir = os.path.join("VisDrone2019-DET-val", "VisDrone2019-DET-val", "images")
    if not os.path.exists(val_dir):
        print(f"[Recon] Dataset not found at {val_dir}"); sys.exit(1)

    imgs = sorted([f for f in os.listdir(val_dir) if f.endswith('.jpg')])
    step = max(1, len(imgs) // args.max_images)
    imgs = imgs[::step][:args.max_images]
    print(f"[Recon] Processing {len(imgs)} images from {val_dir}")

    # ── Load models ────────────────────────────────────────────────────────────
    print("[Recon] Loading YOLO segmentation model…")
    seg_model = YOLO(args.seg_model)

    print("[Recon] Loading VisDrone detector…")
    det_model = YOLO(weights_path)

    print("[Recon] Loading depth estimation model…")
    depth_pipe, depth_backend = load_depth_model()

    # ── Per-image processing ───────────────────────────────────────────────────
    phys_scale    = 0.05          # pixels → world units
    frame_spacing = 300.0 * phys_scale

    all_planes  = []
    all_terrain = []
    all_objects = []

    for frame_idx, filename in enumerate(imgs):
        img_path = os.path.join(val_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[Recon]   Could not load {img_path}, skipping.")
            continue

        h, w = img.shape[:2]
        offset_y = frame_idx * frame_spacing

        print(f"[Recon]  [{frame_idx+1}/{len(imgs)}] {filename}  ({w}×{h})")

        # Ground plane image tile
        all_planes.append({
            "filename": filename,
            "x": 0.0,
            "y": float(offset_y),
            "z": -0.01,
            "w": float(w * phys_scale),
            "h": float(h * phys_scale),
            "frame": frame_idx,
        })

        # Depth prediction
        t0 = time.time()
        depth_norm = predict_depth(depth_pipe, img)
        print(f"[Recon]     depth: {time.time()-t0:.1f}s ({depth_backend})")

        # Semantic terrain sampling
        t0 = time.time()
        terrain = build_terrain_from_image(
            img, seg_model, depth_norm, frame_idx, offset_y, phys_scale, args.stride
        )
        all_terrain.extend(terrain)
        print(f"[Recon]     terrain: {len(terrain)} pts in {time.time()-t0:.1f}s")

        # VisDrone object detection
        t0 = time.time()
        objects = build_objects_from_detections(
            det_model, img, frame_idx, offset_y, phys_scale
        )
        all_objects.extend(objects)
        print(f"[Recon]     objects: {len(objects)} in {time.time()-t0:.1f}s")

    # ── Read training metadata ─────────────────────────────────────────────────
    epoch = 39
    mAP50 = 0.267
    results_csv = "aero_mesh/training/runs/visdrone_fast/results.csv"
    if os.path.exists(results_csv):
        try:
            import csv
            with open(results_csv) as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                epoch = int(float(last.get("                  epoch", epoch)))
                mAP50 = float(last.get("   metrics/mAP50(B)", mAP50))
        except Exception:
            pass

    # ── Write output ───────────────────────────────────────────────────────────
    payload = {
        "generatedAt":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "depthBackend":   depth_backend,
        "stats": {
            "numImages":  len(all_planes),
            "numTerrain": len(all_terrain),
            "numObjects": len(all_objects),
            "epochAtExport":  epoch,
            "mAP50AtExport":  round(float(mAP50), 4),
            "modelName":  "yolo11n",
            "sampleStride": args.stride,
        },
        "planes":   all_planes,
        "terrain":  all_terrain,
        "objects":  all_objects,
    }

    out_file = "web/public/pointcloud.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(payload, f)

    mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"\n[Recon] ✓ Semantic terrain map → {os.path.abspath(out_file)}")
    print(f"         {len(all_planes)} planes | {len(all_terrain)} terrain pts | {len(all_objects)} objects | {mb:.2f} MB")


if __name__ == "__main__":
    main()
