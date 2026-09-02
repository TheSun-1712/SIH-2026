"""
Stage 4: Monocular Depth Infill & Metric Scale Alignment
Uses Depth Anything V2 Metric Outdoor (ViT-L) via HuggingFace Transformers on CUDA.
Fallback chain: DA2 Metric (ViT-L, GPU) → ZoeDepth (GPU) → Synthetic gradient.
Scale-shift alignment to SfM sparse points via robust least-squares.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from aero_mesh.core.frame import Frame

# ---------------------------------------------------------------------------
# GPU Depth Backend — lazy global initialization
# ---------------------------------------------------------------------------
_DEPTH_BACKEND: str = "none"       # "depth_anything_v2" | "zoedepth" | "none"
_DEPTH_MODEL = None                 # HF pipeline or ZoeDepth model
_TORCH_DEVICE: str = "cpu"
_MODEL_LOADED: bool = False


def _init_depth_model() -> str:
    """
    Attempts to load best available monocular metric depth model.
    Returns the backend name that was loaded.
    """
    global _DEPTH_BACKEND, _DEPTH_MODEL, _TORCH_DEVICE, _MODEL_LOADED
    if _MODEL_LOADED:
        return _DEPTH_BACKEND

    # ---- Attempt 1: Depth Anything V2 Metric Outdoor (ViT-L) ----
    try:
        import torch
        from transformers import pipeline as hf_pipeline

        _TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if _TORCH_DEVICE == "cuda" else torch.float32
        model_id = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"

        print(f"[Depth] Loading Depth Anything V2 Metric Outdoor (ViT-L) on {_TORCH_DEVICE}...")
        _DEPTH_MODEL = hf_pipeline(
            task="depth-estimation",
            model=model_id,
            device=0 if _TORCH_DEVICE == "cuda" else -1,
            torch_dtype=dtype,
        )
        _DEPTH_BACKEND = "depth_anything_v2"
        _MODEL_LOADED = True
        print(f"[Depth] ✓ DA2 Metric Outdoor (ViT-L) loaded on {_TORCH_DEVICE}")
        return _DEPTH_BACKEND

    except Exception as e:
        print(f"[Depth] DA2 unavailable: {e}")

    # ---- Attempt 2: ZoeDepth (metric, GPU) ----
    try:
        import torch
        _TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Depth] Loading ZoeDepth (NK) on {_TORCH_DEVICE}...")
        model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True,
                                trust_repo=True)
        model = model.to(_TORCH_DEVICE).eval()
        _DEPTH_MODEL = model
        _DEPTH_BACKEND = "zoedepth"
        _MODEL_LOADED = True
        print(f"[Depth] ✓ ZoeDepth (NK) loaded on {_TORCH_DEVICE}")
        return _DEPTH_BACKEND

    except Exception as e:
        print(f"[Depth] ZoeDepth unavailable: {e}")

    # ---- Fallback: Synthetic gradient ----
    _DEPTH_BACKEND = "synthetic"
    _MODEL_LOADED = True
    print("[Depth] ⚠ All ML depth models unavailable — using synthetic depth fallback.")
    return _DEPTH_BACKEND


# ---------------------------------------------------------------------------

class MonocularScaleAligner:
    """
    GPU-accelerated monocular metric depth prediction & scale alignment.
    Model priority: Depth Anything V2 Metric Outdoor (ViT-L) → ZoeDepth → synthetic.
    """

    def __init__(self, default_mono_model: str = "DepthAnythingV2"):
        self.model_name = default_mono_model

    # ------------------------------------------------------------------
    # Depth prediction
    # ------------------------------------------------------------------

    def predict_metric_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Predicts absolute metric depth map (meters) for a BGR image.
        Returns float32 HxW depth array in meters.
        """
        backend = _init_depth_model()

        if backend == "depth_anything_v2":
            return self._infer_da2(image)
        elif backend == "zoedepth":
            return self._infer_zoedepth(image)
        else:
            return self._synthetic_depth(image)

    def _infer_da2(self, image: np.ndarray) -> np.ndarray:
        """Run Depth Anything V2 Metric (HuggingFace Transformers pipeline)."""
        from PIL import Image as PILImage
        import torch
        rgb = image[:, :, ::-1].astype(np.uint8) if len(image.shape) == 3 else image
        pil = PILImage.fromarray(rgb)
        with torch.no_grad():
            result = _DEPTH_MODEL(pil)
        depth = np.array(result["depth"], dtype=np.float32)
        return depth  # Already in meters (Metric-Outdoor variant)

    def _infer_zoedepth(self, image: np.ndarray) -> np.ndarray:
        """Run ZoeDepth inference (torch hub model)."""
        from PIL import Image as PILImage
        import torch
        rgb = image[:, :, ::-1].astype(np.uint8) if len(image.shape) == 3 else image
        pil = PILImage.fromarray(rgb)
        with torch.no_grad():
            depth = _DEPTH_MODEL.infer_pil(pil)  # Returns metric meters
        return np.array(depth, dtype=np.float32)

    def _synthetic_depth(self, image: np.ndarray) -> np.ndarray:
        """Synthetic radial depth gradient — last resort fallback."""
        h, w = image.shape[:2]
        y, x = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        rel = 1.0 / (1.0 + dist * 0.005)
        return ((rel / rel.max()) * 25.0 + 5.0).astype(np.float32)

    # Backward-compatible alias
    def predict_relative_depth(self, image: np.ndarray) -> np.ndarray:
        metric = self.predict_metric_depth(image)
        mx = float(np.max(metric))
        return (metric / mx) if mx > 1e-6 else metric

    # ------------------------------------------------------------------
    # Scale & shift alignment (Weighted Least Squares)
    # ------------------------------------------------------------------

    def align_scale_and_shift(
        self,
        mono_depth: np.ndarray,
        sparse_metric_depth: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Robust WLS scale+shift alignment:
          argmin_{s,t} || w * ((s * D_mono + t) - D_sparse) ||^2
        where weights are inversely proportional to sparse depth magnitude.
        
        For DA2 Metric: depth is already in meters; s≈1.0, t≈0.0 expected.
        For ZoeDepth / relative models: s and t correct for scene-specific scale.
        """
        d_mono = mono_depth[valid_mask].flatten().astype(np.float64)
        d_ref = sparse_metric_depth[valid_mask].flatten().astype(np.float64)

        if len(d_mono) < 4:
            # No SfM points to align against — trust model output directly
            return mono_depth.astype(np.float64), 1.0, 0.0

        # Inverse-depth weighting: near SfM points weighted more
        weights = np.clip(1.0 / (d_ref + 1.0), 0.01, 10.0)
        W = np.diag(weights)
        A = np.column_stack([d_mono, np.ones_like(d_mono)])

        # WLS: (A^T W A)^{-1} A^T W b
        try:
            AtW = A.T @ W
            s_t = np.linalg.solve(AtW @ A, AtW @ d_ref)
            scale, shift = float(s_t[0]), float(s_t[1])
        except np.linalg.LinAlgError:
            s_t, _, _, _ = np.linalg.lstsq(A, d_ref, rcond=None)
            scale, shift = float(s_t[0]), float(s_t[1])

        scale = max(scale, 0.01)  # Prevent negative/zero scale
        aligned = scale * mono_depth.astype(np.float64) + shift
        return aligned, scale, shift

    # ------------------------------------------------------------------
    # Frame-level depth infill
    # ------------------------------------------------------------------

    def infill_frame_depth(self, frame: Frame, sparse_points_3d: np.ndarray) -> Frame:
        """
        Predicts metric depth for frame, aligns to SfM sparse points (WLS),
        and updates frame.depth_map and frame.depth_confidence.
        """
        img = frame.image_data if frame.image_data is not None else np.zeros((480, 640, 3), np.uint8)
        h, w = img.shape[:2]

        # --- Predict metric depth ---
        depth_pred = self.predict_metric_depth(img)

        # Resize if model output shape differs from image (common with ViT models)
        if depth_pred.shape != (h, w):
            depth_pred = cv2.resize(depth_pred, (w, h), interpolation=cv2.INTER_LINEAR)

        # --- Project SfM sparse points onto frame for scale supervision ---
        sparse_depth_map = np.zeros((h, w), dtype=np.float64)
        valid_mask = np.zeros((h, w), dtype=bool)

        if frame.estimated_pose is not None and len(sparse_points_3d) > 0:
            W2C = np.linalg.inv(frame.estimated_pose.transform_matrix())
            pts_h = np.column_stack([sparse_points_3d,
                                      np.ones(len(sparse_points_3d))])
            pts_cam = (W2C @ pts_h.T).T

            fx = fy = 500.0
            cx, cy = w / 2.0, h / 2.0

            for pt in pts_cam:
                z = pt[2]
                if z > 0.1:
                    u = int(round(fx * pt[0] / z + cx))
                    v = int(round(fy * pt[1] / z + cy))
                    if 0 <= u < w and 0 <= v < h:
                        sparse_depth_map[v, u] = z
                        valid_mask[v, u] = True

        # --- WLS scale-shift alignment ---
        aligned, scale, shift = self.align_scale_and_shift(
            depth_pred.astype(np.float64), sparse_depth_map, valid_mask
        )
        frame.depth_map = np.clip(aligned, 0.1, 500.0)

        # --- Per-pixel depth confidence ---
        # DA2 Metric baseline: 0.78 (high-quality metric predictions)
        # ZoeDepth baseline: 0.70
        # Synthetic baseline: 0.45
        backend = _init_depth_model()
        baselines = {"depth_anything_v2": 0.78, "zoedepth": 0.70, "synthetic": 0.45}
        base_conf = baselines.get(backend, 0.60)

        confidence = np.full((h, w), base_conf, dtype=np.float32)
        confidence[valid_mask] = 0.93   # SfM-observed pixels get highest trust
        frame.depth_confidence = confidence

        frame.metadata["depth_backend"] = backend
        frame.metadata["depth_device"] = _TORCH_DEVICE
        frame.metadata["scale_alignment"] = {"scale": round(scale, 4), "shift": round(shift, 4)}

        return frame
