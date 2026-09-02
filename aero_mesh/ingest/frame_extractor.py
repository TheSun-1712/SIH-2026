"""
Stage 1: Frame Extraction & Quality Filtering
Extracts video frames, evaluates Laplacian blur scores, and selects keyframes based on baseline movement and visual sharpness.
"""

import os
import cv2
import numpy as np
from typing import List, Optional
from aero_mesh.core.frame import Frame

def laplacian_sharpness(img: np.ndarray) -> float:
    """Calculates variance of Laplacian for blur estimation."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

class FrameExtractor:
    def __init__(self, sharpness_threshold: float = 80.0, min_keyframe_interval: float = 0.5):
        self.sharpness_threshold = sharpness_threshold
        self.min_keyframe_interval = min_keyframe_interval

    def extract_from_video(self, video_path: str, output_dir: str, target_fps: float = 2.0) -> List[Frame]:
        """Extracts frames from video file at target_fps, saving to output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_stride = max(1, int(round(video_fps / target_fps)))

        frames: List[Frame] = []
        frame_count = 0
        extracted_idx = 0

        while True:
            ret, mat = cap.read()
            if not ret:
                break

            if frame_count % frame_stride == 0:
                timestamp = frame_count / video_fps
                sharpness = laplacian_sharpness(mat)
                
                filename = f"frame_{extracted_idx:05d}.jpg"
                img_path = os.path.join(output_dir, filename)
                cv2.imwrite(img_path, mat)

                frame_obj = Frame(
                    frame_id=extracted_idx,
                    timestamp=timestamp,
                    image_path=img_path,
                    sharpness_score=sharpness,
                    metadata={"original_frame_index": frame_count}
                )
                frames.append(frame_obj)
                extracted_idx += 1

            frame_count += 1

        cap.release()
        return frames

    def select_keyframes(self, frames: List[Frame], min_baseline_meters: float = 2.0) -> List[Frame]:
        """Greedily selects keyframes that satisfy sharpness threshold and baseline displacement."""
        keyframes: List[Frame] = []
        last_selected_pose: Optional[np.ndarray] = None

        for frame in frames:
            # Filter out motion blurred frames
            if frame.sharpness_score < self.sharpness_threshold:
                frame.is_keyframe = False
                continue

            if frame.estimated_pose is not None:
                curr_pos = frame.estimated_pose.position
                if last_selected_pose is None:
                    frame.is_keyframe = True
                    keyframes.append(frame)
                    last_selected_pose = curr_pos
                else:
                    dist = float(np.linalg.norm(curr_pos - last_selected_pose))
                    if dist >= min_baseline_meters:
                        frame.is_keyframe = True
                        keyframes.append(frame)
                        last_selected_pose = curr_pos
                    else:
                        frame.is_keyframe = False
            else:
                # If pose is not yet available, fallback to temporal stride
                if not keyframes or (frame.timestamp - keyframes[-1].timestamp) >= self.min_keyframe_interval:
                    frame.is_keyframe = True
                    keyframes.append(frame)
                else:
                    frame.is_keyframe = False

        return keyframes
