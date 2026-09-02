"""
Stage 7: Dynamic Object Detection & Motion Masking
Uses RAFT-Large optical flow (GPU/CUDA) for dense flow estimation,
ego-motion compensation via estimated pose delta, and SAM2 instance segmentation
for precise dynamic object masks. Masks out moving objects from depth maps.

GPU Backend: RAFT-Large (torchvision) → Farneback (CPU fallback)
Segmentation: SAM2 (Meta, GPU) → bounding-box crop masking fallback
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from aero_mesh.core.frame import Frame

# ---------------------------------------------------------------------------
# RAFT-Large GPU Backend (torchvision)
# ---------------------------------------------------------------------------
_RAFT_MODEL = None
_RAFT_TRANSFORMS = None
_RAFT_DEVICE = "cpu"
_RAFT_LOADED = False

def _init_raft():
    """Lazily load RAFT-Large from torchvision (GPU preferred)."""
    global _RAFT_MODEL, _RAFT_TRANSFORMS, _RAFT_DEVICE, _RAFT_LOADED
    if _RAFT_LOADED:
        return _RAFT_MODEL is not None
    _RAFT_LOADED = True
    try:
        import torch
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        from torchvision.models.optical_flow import raft_large

        _RAFT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        weights = Raft_Large_Weights.DEFAULT
        _RAFT_MODEL = raft_large(weights=weights, progress=False).to(_RAFT_DEVICE)
        _RAFT_MODEL.eval()
        _RAFT_TRANSFORMS = weights.transforms()
        print(f"[Motion] RAFT-Large loaded on {_RAFT_DEVICE}")
        return True
    except Exception as e:
        print(f"[Motion] RAFT-Large unavailable: {e}. Falling back to Farneback.")
        return False

# ---------------------------------------------------------------------------
# SAM2 Segmentation Backend (Meta)
# ---------------------------------------------------------------------------
_SAM2_MODEL = None
_SAM2_LOADED = False

def _init_sam2():
    """Lazily load SAM2 image predictor for precise object masking."""
    global _SAM2_MODEL, _SAM2_LOADED
    if _SAM2_LOADED:
        return _SAM2_MODEL is not None
    _SAM2_LOADED = True
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # SAM2-large for best accuracy; requires model checkpoint download
        _SAM2_MODEL = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        _SAM2_MODEL.model.to(device)
        print(f"[Motion] SAM2-Hiera-Large loaded on {device}")
        return True
    except Exception as e:
        print(f"[Motion] SAM2 unavailable: {e}. Using bbox crop masking.")
        return False


class DynamicObjectFilter:
    """
    GPU-accelerated dynamic object detection and masking.
    
    Pipeline:
      1. YOLOv11x VisDrone Detector for high-confidence dynamic objects (vehicles, pedestrians)
      2. RAFT-Large dense optical flow (GPU) for remaining unclassified motion
      3. Ego-motion compensation via ESKF pose delta
      4. SAM2 instance segmentation for precise object boundaries (optional)
      5. Mask out dynamic regions in depth maps and confidence maps
    """

    def __init__(self, motion_threshold: float = 2.5, use_sam2: bool = True, use_yolo: bool = True):
        self.motion_threshold = motion_threshold
        self.use_sam2 = use_sam2
        self.use_yolo = use_yolo
        self.detector = None

        if self.use_yolo:
            try:
                from aero_mesh.training.detector_inference import VisDroneDetector
                # We load it lazily or initialize here
                self.detector = VisDroneDetector()
            except ImportError:
                print("[Motion] YOLOv11x VisDroneDetector not found. Falling back to RAFT only.")
                self.detector = None

    # ------------------------------------------------------------------
    # Optical Flow
    # ------------------------------------------------------------------

    def _compute_raft_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow using RAFT-Large on GPU.
        Returns flow magnitude map (HxW float32).
        """
        import torch

        def _to_tensor(img: np.ndarray) -> torch.Tensor:
            """BGR ndarray → (1, 3, H, W) float32 tensor."""
            rgb = img[:, :, ::-1].astype(np.float32)
            t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            return t.to(_RAFT_DEVICE)

        # Resize to multiple of 8 (RAFT requirement)
        h, w = img1.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        img1_pad = cv2.copyMakeBorder(img1, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        img2_pad = cv2.copyMakeBorder(img2, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        t1 = _to_tensor(img1_pad)
        t2 = _to_tensor(img2_pad)

        if _RAFT_TRANSFORMS is not None:
            t1, t2 = _RAFT_TRANSFORMS(t1, t2)

        with torch.no_grad():
            flow_predictions = _RAFT_MODEL(t1, t2)

        flow = flow_predictions[-1].squeeze().cpu().numpy()  # (2, H+pad, W+pad)
        flow = flow[:, :h, :w]  # Remove padding

        mag = np.sqrt(flow[0] ** 2 + flow[1] ** 2).astype(np.float32)
        return mag

    def _compute_farneback_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Dense Farneback optical flow — CPU fallback."""
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 5, 25, 5, 7, 1.5, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return mag.astype(np.float32)

    def compute_optical_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Dispatch to RAFT-Large (GPU) or Farneback (CPU)."""
        if _init_raft():
            try:
                return self._compute_raft_flow(img1, img2)
            except Exception as e:
                print(f"[Motion] RAFT error: {e}. Using Farneback.")
        return self._compute_farneback_flow(img1, img2)

    # ------------------------------------------------------------------
    # Ego-motion compensation
    # ------------------------------------------------------------------

    def _estimate_ego_flow(self, img1: np.ndarray,
                            f1: Frame, f2: Frame) -> Optional[float]:
        """
        Estimate expected background flow magnitude from ESKF pose delta.
        Returns expected ego-motion flow threshold in pixels.
        """
        if f1.estimated_pose is None or f2.estimated_pose is None:
            return None
        delta_pos = np.linalg.norm(
            f2.estimated_pose.position - f1.estimated_pose.position
        )
        h, w = img1.shape[:2]
        # Rough projection: 500px focal, 25m altitude → px_per_meter ≈ 20
        approx_altitude = abs(f1.estimated_pose.position[2]) + 1.0
        focal = 500.0
        px_per_meter = focal / approx_altitude
        ego_flow = delta_pos * px_per_meter
        return float(ego_flow)

    # ------------------------------------------------------------------
    # SAM2 segmentation
    # ------------------------------------------------------------------

    def _refine_mask_with_sam2(self, img: np.ndarray,
                                coarse_mask: np.ndarray) -> np.ndarray:
        """
        Uses SAM2 to refine coarse motion mask into precise instance boundaries.
        Finds connected components in coarse mask → prompts SAM2 with bboxes.
        """
        if not _init_sam2() or _SAM2_MODEL is None:
            return coarse_mask

        try:
            import torch
            # Find connected components (candidate dynamic objects)
            coarse_uint = (coarse_mask * 255).astype(np.uint8)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(coarse_uint)

            refined = np.zeros_like(coarse_mask, dtype=bool)
            rgb = img[:, :, ::-1]  # BGR→RGB for SAM2

            with torch.inference_mode():
                _SAM2_MODEL.set_image(rgb)
                for comp_id in range(1, n_labels):
                    area = stats[comp_id, cv2.CC_STAT_AREA]
                    if area < 50:  # Skip tiny blobs
                        continue
                    x, y, bw, bh = (stats[comp_id, cv2.CC_STAT_LEFT],
                                    stats[comp_id, cv2.CC_STAT_TOP],
                                    stats[comp_id, cv2.CC_STAT_WIDTH],
                                    stats[comp_id, cv2.CC_STAT_HEIGHT])
                    bbox = np.array([x, y, x + bw, y + bh], dtype=np.float32)
                    masks, _, _ = _SAM2_MODEL.predict(
                        box=bbox[None],
                        multimask_output=False
                    )
                    refined |= masks[0].astype(bool)
            return refined
        except Exception as e:
            print(f"[Motion] SAM2 refinement error: {e}")
            return coarse_mask

    # ------------------------------------------------------------------
    # Main filter
    # ------------------------------------------------------------------

    def filter_dynamic_objects(self, frames: List[Frame]) -> List[Frame]:
        """
        Applies RAFT optical flow + ego-motion compensation + SAM2 masking
        to all consecutive frame pairs. Zeroes dynamic regions in depth maps.
        """
        total_masked = 0.0

        for i in range(1, len(frames)):
            f_prev, f_curr = frames[i - 1], frames[i]

            if f_prev.image_data is None or f_curr.image_data is None:
                continue

            img_prev = f_prev.image_data
            img_curr = f_curr.image_data

            # --- Dense optical flow ---
            flow_mag = self.compute_optical_flow(img_prev, img_curr)

            # --- Ego-motion adaptive threshold ---
            ego_flow = self._estimate_ego_flow(img_curr, f_prev, f_curr)
            if ego_flow is not None:
                # Background pixels have flow ≈ ego_flow; dynamic = significantly more
                dynamic_threshold = ego_flow + self.motion_threshold
            else:
                # Fallback: median-based threshold
                dynamic_threshold = float(np.median(flow_mag)) + self.motion_threshold

            # --- Coarse dynamic mask (Optical Flow) ---
            coarse_mask = flow_mag > dynamic_threshold

            # --- YOLOv11x Dynamic Detections ---
            if self.detector is not None:
                try:
                    yolo_mask = self.detector.detect_and_mask(img_curr, expand_pixels=4)
                    # Merge optical flow and YOLO detections
                    coarse_mask = coarse_mask | yolo_mask
                except Exception as e:
                    print(f"[Motion] YOLO inference failed: {e}")

            # --- Morphological cleanup ---
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            coarse_mask_uint = coarse_mask.astype(np.uint8) * 255
            coarse_mask_uint = cv2.morphologyEx(coarse_mask_uint, cv2.MORPH_CLOSE, kernel)
            coarse_mask_uint = cv2.morphologyEx(coarse_mask_uint, cv2.MORPH_OPEN, kernel)
            coarse_mask = coarse_mask_uint > 0

            # --- SAM2 mask refinement (optional) ---
            if self.use_sam2:
                final_mask = self._refine_mask_with_sam2(img_curr, coarse_mask)
            else:
                final_mask = coarse_mask

            # --- Apply mask to depth map and confidence ---
            if f_curr.depth_map is not None:
                f_curr.depth_map[final_mask] = 0.0
            if f_curr.depth_confidence is not None:
                f_curr.depth_confidence[final_mask] = 0.0

            mask_ratio = float(np.mean(final_mask))
            f_curr.metadata["dynamic_mask_ratio"] = mask_ratio
            f_curr.metadata["motion_flow_backend"] = "raft" if _RAFT_MODEL else "farneback"
            f_curr.metadata["sam2_used"] = _SAM2_MODEL is not None
            total_masked += mask_ratio

        if len(frames) > 1:
            avg = total_masked / (len(frames) - 1)
            print(f"[Motion] Dynamic object filtering complete. "
                  f"Avg mask ratio: {avg:.1%} "
                  f"({'RAFT' if _RAFT_MODEL else 'Farneback'} + "
                  f"{'SAM2' if _SAM2_MODEL else 'morphological'})")
        return frames
