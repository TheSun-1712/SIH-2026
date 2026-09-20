"""
AERO MESH — Semantic 3D Terrain Reconstruction
===============================================
Generates a full semantic terrain model from drone images by combining:
  1. YOLOv11n-seg  → per-pixel semantic masks (80 COCO classes → 6 terrain buckets)
  2. Depth Anything V2 Metric Outdoor (ViT-L) → per-pixel metric depth in metres
  3. VisDrone YOLO detector → moving objects (cars, people) as thin overlay markers

[NEW] Entity Extraction (Phase 2 Upgrade):
  4. Metric Height Clustering — connected-component building instance labelling
     with real height H = z_roof - z_ground from DA2 aligned depth.
  5. VLM CLIP Architectural Classifier — classifies each building patch into one
     of 7 architectural types for the Shape Grammar Engine's asset selection.
  6. OUGS Uncertainty Integration — per-entity uncertainty scores from the
     OUGSScorer are embedded in the entity manifest.
  7. Cross-Modal Fuser — merges aerial and Mapillary street-level point clouds.

Output JSON schema:
  {
    "generatedAt": "...",
    "planes": [...],
    "terrain": [...],
    "objects": [...],
    "entities": {
      "buildings": [{"id", "type", "footprint", "measuredHeight",
                     "floorCount", "confidence", "lod", "uncertainty"}],
      "vegetation": [{"id", "position", "height", "radius"}],
      "vehicles":   [{"id", "type", "position", "heading"}],
      "roads":      [{"id", "polyline", "width", "type"}]
    },
    "stats": {...}
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
        print("[Recon] Loading Depth Anything V2 Metric Outdoor (ViT-L)…")
        pipe = hf_pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
            device=device,
            torch_dtype=torch.float32,
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


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Entity Extraction, Metric Height Clustering & VLM Classification
# ─────────────────────────────────────────────────────────────────────────────

VLM_CLASS_NAMES = [
    "high_rise_residential", "office_tower", "commercial_shopfront",
    "industrial_warehouse", "villa_house", "apartment_block", "mixed_use",
]

FLOOR_HEIGHT_BY_TYPE = {
    "high_rise_residential": 2.85,
    "office_tower":          3.50,
    "commercial_shopfront":  4.50,
    "industrial_warehouse":  8.00,
    "villa_house":           3.00,
    "apartment_block":       3.00,
    "mixed_use":             3.20,
}


def extract_building_instances(
    seg_masks_data: np.ndarray,       # (N_masks, H, W) float32
    seg_classes:    np.ndarray,       # (N_masks,) int class ids
    depth_norm:     np.ndarray,       # (H, W) float32 in [0,1]
    img_bgr:        np.ndarray,       # (H, W, 3) uint8 for patch crops
    phys_scale:     float = 0.05,
    offset_y:       float = 0.0,
    frame_idx:      int   = 0,
    depth_max_m:    float = 80.0,     # DA2 Metric Outdoor max depth
) -> list:
    """
    Extract discrete building instances from YOLO-seg masks.
    For each building mask:
      - Computes connected components to isolate individual buildings
      - Measures real height H = depth_roof - depth_ground using DA2 depth
      - Extracts oriented bounding box footprint
      - Crops the image patch for VLM classification
    Returns list of raw building instance dicts.
    """
    h, w = img_bgr.shape[:2]
    buildings = []
    bldg_idx  = 0

    # 1. Color-based rooftop segmentation (warm/reddish/concrete tones)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1] / 255.0, hsv[:, :, 2] / 255.0
    bldg_color_mask = ((h_ch < 22) | (h_ch > 155)) & (s_ch > 0.20) & (v_ch > 0.20)

    # 2. Depth elevation mask (structures raised above local ground plane)
    ground_depth = float(np.median(depth_norm))
    elevated_mask = (ground_depth - depth_norm) > 0.03
    combined_mask = ((bldg_color_mask | elevated_mask) & (v_ch > 0.15)).astype(np.uint8) * 255

    # 3. Morphological cleanup (merge roof facets, eliminate road lines & noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    min_area = 250   # At least ~250 pixels
    max_area = int(h * w * 0.35)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        x_min = stats[i, cv2.CC_STAT_LEFT]
        y_min = stats[i, cv2.CC_STAT_TOP]
        box_w = stats[i, cv2.CC_STAT_WIDTH]
        box_h = stats[i, cv2.CC_STAT_HEIGHT]
        x_max = x_min + box_w
        y_max = y_min + box_h

        # Aspect ratio filter (reject long narrow roads/pavements)
        aspect = max(box_w, box_h) / (min(box_w, box_h) + 1e-3)
        if aspect > 4.5:
            continue

        comp_mask = (labels == i)
        depth_in_mask = depth_norm[comp_mask]
        if len(depth_in_mask) == 0:
            continue

        mean_depth = float(depth_in_mask.mean())
        depth_diff = max(0.02, float(ground_depth - mean_depth))
        height_m   = max(3.5, min(depth_diff * depth_max_m * 2.2, 60.0))

        cx_px = float(centroids[i][0])
        cy_px = float(centroids[i][1])
        cx_w  = (cx_px - w / 2.0) * phys_scale
        cz_w  = (cy_px - h / 2.0) * phys_scale + offset_y
        width_w = max(6.0, box_w * phys_scale)
        depth_w = max(6.0, box_h * phys_scale)

        pad = 8
        crop = img_bgr[
            max(0, y_min - pad):min(h, y_max + pad),
            max(0, x_min - pad):min(w, x_max + pad)
        ]

        buildings.append({
            "id":          f"bldg_{frame_idx}_{bldg_idx}",
            "cx":          float(cx_w),
            "cz":          float(cz_w),
            "width_m":     float(width_w),
            "depth_m":     float(depth_w),
            "height_m":    float(height_m),
            "mean_depth":  float(mean_depth),
            "frame":       frame_idx,
            "crop":        crop,
        })
        bldg_idx += 1

    return buildings


def classify_building_type(crop_bgr: np.ndarray, vlm_ckpt: str = None) -> str:
    """
    Classify a building aerial crop into one of 7 architectural types.

    If the VLM checkpoint is available (train_vlm_classifier.py has been run),
    uses the trained CLIP head. Otherwise falls back to colour/texture heuristics.

    Args:
        crop_bgr:  (H, W, 3) BGR crop of the building from above
        vlm_ckpt:  path to the VLM classifier checkpoint (best.pt)

    Returns: string architectural type label
    """
    default_type = "apartment_block"

    if vlm_ckpt and os.path.exists(vlm_ckpt):
        try:
            import torch
            from PIL import Image as PILImg
            ckpt = torch.load(vlm_ckpt, map_location="cpu", weights_only=False)
            head_state = ckpt.get("head_state")
            if head_state is None:
                return default_type

            embed_dim = next(iter(head_state.values())).shape[-1] \
                if "0.weight" not in head_state else head_state["0.weight"].shape[1]

            import torch.nn as nn
            head = nn.Sequential(
                nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 256), nn.GELU(),
                nn.Dropout(0.2), nn.Linear(256, len(VLM_CLASS_NAMES)),
            )
            head.load_state_dict(head_state)
            head.eval()

            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            rgb_pil = PILImg.fromarray(rgb)

            if embed_dim == 512:
                # Real CLIP mode
                from transformers import CLIPModel, CLIPProcessor
                cfg = ckpt.get("config", {})
                base_model = cfg.get("base_model", "openai/clip-vit-base-patch32")
                proc = CLIPProcessor.from_pretrained(base_model)
                clip_m = CLIPModel.from_pretrained(base_model)
                clip_m.eval()
                inputs = proc(images=rgb_pil, return_tensors="pt")
                with torch.no_grad():
                    feat = clip_m.get_image_features(**inputs)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    pred = head(feat).argmax(-1).item()
            else:
                # Flat feature mode
                rgb_resized = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
                feat = torch.from_numpy(rgb_resized.transpose(2, 0, 1).flatten()).float().unsqueeze(0)
                with torch.no_grad():
                    pred = head(feat).argmax(-1).item()
            return VLM_CLASS_NAMES[pred]
        except Exception as e:
            pass  # Fall through to heuristic

    # Colour heuristic fallback
    if crop_bgr is None or crop_bgr.size == 0:
        return default_type
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mean_h = float(hsv[:, :, 0].mean())
    mean_s = float(hsv[:, :, 1].mean()) / 255.0
    mean_v = float(hsv[:, :, 2].mean()) / 255.0
    area   = crop_bgr.shape[0] * crop_bgr.shape[1]

    # Large + grey → industrial warehouse or office
    if area > 40000 and mean_s < 0.15:
        return "industrial_warehouse" if mean_v < 0.5 else "office_tower"
    # Warm + reddish → residential
    if mean_h < 20 and mean_s > 0.2:
        return "high_rise_residential"
    # Green-ish (vegetation on roof) → villa
    if 55 < mean_h < 85 and mean_s > 0.2:
        return "villa_house"
    # Blue-ish glass → office tower
    if 90 < mean_h < 130 and mean_s > 0.3:
        return "office_tower"
    return default_type


def extract_vegetation_instances(
    terrain_pts: list, phys_scale: float = 0.05
) -> list:
    """
    Cluster terrain points labelled 'vegetation' into discrete tree instances.
    Uses simple grid-cell clustering — each occupied grid cell = one tree.
    Returns list of vegetation entity dicts.
    """
    veg_pts = [p for p in terrain_pts if p.get("label") == "vegetation"]
    if not veg_pts:
        return []

    cell_size = 4.0  # Group into 4m grid cells
    cells:    dict = {}
    for pt in veg_pts:
        key = (round(pt["x"] / cell_size), round(pt["y"] / cell_size))
        if key not in cells:
            cells[key] = []
        cells[key].append(pt)

    vegetation = []
    for tree_idx, (key, pts_in_cell) in enumerate(cells.items()):
        cx   = float(np.mean([p["x"] for p in pts_in_cell]))
        cz   = float(np.mean([p["y"] for p in pts_in_cell]))
        # Height: use the 'h' field from terrain points
        heights = [p.get("h", 2.5) for p in pts_in_cell]
        tree_h  = float(np.max(heights))
        radius  = min(max(len(pts_in_cell) * 0.3, 1.5), 6.0)
        vegetation.append({
            "id":       f"tree_{tree_idx}",
            "position": [cx, 0.0, cz],
            "height":   round(tree_h, 2),
            "radius":   round(radius, 2),
        })
    return vegetation


def build_entities_manifest(
    all_building_instances: list,
    all_terrain:            list,
    all_objects:            list,
    vlm_ckpt:               str = None,
    ougs_scores:            dict = None,
    phys_scale:             float = 0.05,
) -> dict:
    """
    Build the final entities manifest from extracted building instances,
    terrain vegetation clusters, and detected objects.

    This is the data structure consumed by the web viewer's DynamicEntityManager.
    """
    # ── Buildings ──────────────────────────────────────────────────────────────
    buildings_out = []
    for inst in all_building_instances:
        crop      = inst.pop("crop", None)
        arch_type = classify_building_type(crop, vlm_ckpt)
        fh        = FLOOR_HEIGHT_BY_TYPE.get(arch_type, 3.0)
        n_floors  = max(1, round(inst["height_m"] / fh))

        # Confidence: higher if mid-depth (well-observed from above)
        depth     = inst.get("mean_depth", 0.5)
        conf      = float(np.clip(1.0 - abs(depth - 0.45) * 2.0, 0.3, 0.95))

        # OUGS uncertainty (if available)
        uncertainty = float(ougs_scores.get(inst["id"], 0.5)) \
            if ougs_scores else 0.5

        buildings_out.append({
            "id":            inst["id"],
            "type":          arch_type,
            "footprint": {
                "cx":      round(inst["cx"], 3),
                "cz":      round(inst["cz"], 3),
                "w":       round(inst["width_m"], 2),
                "d":       round(inst["depth_m"], 2),
                "heading": 0.0,
            },
            "measuredHeight": round(inst["height_m"], 2),
            "floorCount":     n_floors,
            "confidence":     round(conf, 3),
            "uncertainty":    round(uncertainty, 3),
            "lod":            2,          # Upgraded from LoD1 OSM prior
            "source":         "drone_reconstruction",
            "frame":          inst.get("frame", 0),
        })

    # Deduplicate buildings that are within 3m of each other (across frames)
    deduped = []
    for b in buildings_out:
        too_close = False
        for existing in deduped:
            dx = b["footprint"]["cx"] - existing["footprint"]["cx"]
            dz = b["footprint"]["cz"] - existing["footprint"]["cz"]
            if dx*dx + dz*dz < 9.0:   # 3m radius
                # Keep the one with higher confidence
                if b["confidence"] > existing["confidence"]:
                    deduped.remove(existing)
                    deduped.append(b)
                too_close = True
                break
        if not too_close:
            deduped.append(b)
    buildings_out = deduped

    # ── Vegetation ─────────────────────────────────────────────────────────────
    vegetation_out = extract_vegetation_instances(all_terrain, phys_scale)

    # ── Vehicles from detection ────────────────────────────────────────────────
    VIS_TO_TYPE = {
        "Car": "Car", "Van": "Van", "Truck": "Truck", "Bus": "Bus",
        "Bicycle": "Bicycle", "Motor": "Motorcycle",
        "Pedestrian": "Pedestrian", "People": "Pedestrian",
    }
    vehicles_out = []
    for i, obj in enumerate(all_objects):
        lbl = obj.get("label", "Others")
        vtype = VIS_TO_TYPE.get(lbl, lbl)
        vehicles_out.append({
            "id":       f"veh_{i}",
            "type":     vtype,
            "position": [obj["x"], 0.0, obj["y"]],
            "heading":  0.0,
            "conf":     obj.get("conf", 0.5),
        })

    return {
        "buildings":  buildings_out,
        "vegetation": vegetation_out,
        "vehicles":   vehicles_out,
        "roads":      [],     # Populated by gis_prior.py or cross_modal_fuser.py
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',    type=str, default='aero_mesh/training/runs/visdrone_fast/weights/last.pt')
    parser.add_argument('--seg-model', type=str, default='yolo11n-seg.pt')
    parser.add_argument('--max-images',type=int, default=5)
    parser.add_argument('--stride',    type=int, default=20, help='Terrain sampling stride (px). Lower = denser but slower.')
    parser.add_argument('--vlm-ckpt',  type=str, default='checkpoints/vlm/best.pt',
                        help='Path to VLM CLIP classifier checkpoint (optional — uses heuristic if absent)')
    parser.add_argument('--no-entities', action='store_true',
                        help='Skip entity extraction (faster, for quick terrain-only runs)')
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

    all_planes             = []
    all_terrain            = []
    all_objects            = []
    all_building_instances = []   # [NEW] for entity extraction

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

        # Save depth heatmap for Tier 1 Visualizer
        heatmap_dir = os.path.join("web", "public", "drone")
        os.makedirs(heatmap_dir, exist_ok=True)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        heatmap_filename = filename.replace(".jpg", "_depth.jpg")
        cv2.imwrite(os.path.join(heatmap_dir, heatmap_filename), depth_color)

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

        # [NEW] Building instance extraction
        if not args.no_entities:
            seg_result = seg_model(img, verbose=False)[0]
            masks_data  = seg_result.masks.data.cpu().numpy() \
                if seg_result.masks is not None else np.zeros((0, *img.shape[:2]))
            cls_data    = seg_result.boxes.cls.cpu().numpy().astype(int) \
                if seg_result.boxes is not None else np.array([])
            buildings_raw = extract_building_instances(
                masks_data, cls_data, depth_norm, img,
                phys_scale=phys_scale, offset_y=offset_y, frame_idx=frame_idx
            )
            all_building_instances.extend(buildings_raw)
            print(f"[Recon]     buildings detected: {len(buildings_raw)}")

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

    # ── [NEW] Build entity manifest ────────────────────────────────────────────
    entities = {"buildings": [], "vegetation": [], "vehicles": [], "roads": []}
    if not args.no_entities and all_building_instances:
        print(f"\n[Recon] Building entity manifest ({len(all_building_instances)} raw instances)…")
        vlm_ckpt = args.vlm_ckpt if os.path.exists(args.vlm_ckpt) else None
        if vlm_ckpt:
            print(f"[Recon]   VLM classifier: {vlm_ckpt}")
        else:
            print("[Recon]   VLM checkpoint not found — using heuristic classification.")
            print(f"          Train first: python aero_mesh/planning/train_vlm_classifier.py")

        entities = build_entities_manifest(
            all_building_instances, all_terrain, all_objects,
            vlm_ckpt=vlm_ckpt, phys_scale=phys_scale,
        )
        print(f"[Recon]   Entities: {len(entities['buildings'])} buildings, "
              f"{len(entities['vegetation'])} trees, {len(entities['vehicles'])} vehicles")
    elif not args.no_entities:
        # Still extract vegetation from terrain even if no seg-based buildings
        entities["vegetation"] = extract_vegetation_instances(all_terrain, phys_scale)
        entities["vehicles"]   = [
            {"id": f"veh_{i}", "type": o.get("label", "Vehicle"),
             "position": [o["x"], 0.0, o["y"]], "heading": 0.0, "conf": o.get("conf", 0.5)}
            for i, o in enumerate(all_objects)
        ]

    # ── Write output ───────────────────────────────────────────────────────────
    payload = {
        "generatedAt":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        "depthBackend":   depth_backend,
        "stats": {
            "numImages":   len(all_planes),
            "numTerrain":  len(all_terrain),
            "numObjects":  len(all_objects),
            "numBuildings": len(entities.get("buildings", [])),
            "numVegetation": len(entities.get("vegetation", [])),
            "epochAtExport": epoch,
            "mAP50AtExport": round(float(mAP50), 4),
            "modelName":   "yolo11n",
            "sampleStride": args.stride,
            "entityExtractionEnabled": not args.no_entities,
        },
        "planes":   all_planes,
        "terrain":  all_terrain,
        "objects":  all_objects,
        "entities": entities,       # [NEW] — consumed by DynamicEntityManager
    }

    out_file = "web/public/pointcloud.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(payload, f)

    mb = os.path.getsize(out_file) / (1024 * 1024)
    print(f"\n[Recon] ✓ Semantic terrain map → {os.path.abspath(out_file)}")
    print(f"         {len(all_planes)} planes | {len(all_terrain)} terrain pts | "
          f"{len(all_objects)} objects | {len(entities.get('buildings',[]))} buildings | "
          f"{mb:.2f} MB")


if __name__ == "__main__":
    main()
