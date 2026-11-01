"""
Engagement Analysis & Multimodal Fusion Module.
Analyzes visual indicators (face visibility rate, head pose alignment, motion dynamics)
and fuses them with acoustic indicators to assess classroom engagement.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    YAW_FORWARD_TOLERANCE,
    PITCH_FORWARD_TOLERANCE,
    WEIGHT_FACE_VISIBILITY,
    WEIGHT_HEAD_ORIENTATION,
    WEIGHT_AUDIO_INTERACTION
)
from modules.database import log_session_indicator


class VisualEngagementAnalyzer:
    """
    Extracts observable physical engagement metrics from computer vision frames:
    - Head pose yaw and pitch deviation using facial landmarks
    - Motion index via frame differencing
    - Aggregated face visibility rate
    """

    def __init__(self):
        self.prev_gray_frame: Optional[np.ndarray] = None

    def estimate_head_pose_from_landmarks(
        self,
        landmarks: Optional[List[Tuple[float, float]]]
    ) -> Dict[str, Any]:
        """
        Estimates yaw and pitch angles from 5 facial landmarks:
        [0]: right eye, [1]: left eye, [2]: nose, [3]: mouth right, [4]: mouth left.
        Returns yaw, pitch, and boolean is_looking_forward.
        """
        if not landmarks or len(landmarks) < 5:
            # If landmarks unavailable, assume neutral forward alignment
            return {
                "approx_yaw_deg": 0.0,
                "approx_pitch_deg": 0.0,
                "is_deviated": False,
                "confidence": "LOW (No landmarks)"
            }

        re = np.array(landmarks[0])
        le = np.array(landmarks[1])
        nose = np.array(landmarks[2])
        mr = np.array(landmarks[3])
        ml = np.array(landmarks[4])

        # 1. Yaw Estimation (Horizontal projection asymmetry)
        dx_re_nose = abs(nose[0] - re[0])
        dx_le_nose = abs(nose[0] - le[0])
        total_eye_dx = dx_re_nose + dx_le_nose + 1e-6
        yaw_ratio = (dx_le_nose - dx_re_nose) / total_eye_dx
        approx_yaw = round(float(yaw_ratio * 75.0), 1)

        # 2. Pitch Estimation (Vertical ratio)
        eye_mid_y = (re[1] + le[1]) / 2.0
        mouth_mid_y = (mr[1] + ml[1]) / 2.0
        nose_to_eyes_y = max(1e-6, abs(nose[1] - eye_mid_y))
        nose_to_mouth_y = max(1e-6, abs(mouth_mid_y - nose[1]))

        # Normal face has nose_to_mouth roughly 0.9 - 1.3 times nose_to_eyes
        pitch_ratio = (nose_to_eyes_y - nose_to_mouth_y) / (nose_to_eyes_y + nose_to_mouth_y + 1e-6)
        approx_pitch = round(float(pitch_ratio * 60.0), 1)

        is_yaw_deviated = abs(approx_yaw) >= YAW_FORWARD_TOLERANCE
        is_pitch_deviated = abs(approx_pitch) >= PITCH_FORWARD_TOLERANCE
        is_deviated = is_yaw_deviated or is_pitch_deviated

        return {
            "approx_yaw_deg": approx_yaw,
            "approx_pitch_deg": approx_pitch,
            "is_deviated": is_deviated,
            "confidence": "HIGH (5-point Landmark Geometric Ratio)"
        }

    def compute_motion_index(self, frame_bgr: np.ndarray) -> float:
        """
        Measures dynamic motion between consecutive video frames using normalized absolute differencing.
        Returns value between 0.0 (static picture / frozen) and 1.0 (extreme movement).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return 0.0

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray_frame is None or self.prev_gray_frame.shape != gray.shape:
            self.prev_gray_frame = gray
            return 0.0

        frame_delta = cv2.absdiff(self.prev_gray_frame, gray)
        self.prev_gray_frame = gray

        # Compute normalized intensity change
        mean_diff = np.mean(frame_delta)
        # Scale: mean difference of 25 corresponds to significant movement
        motion_index = float(np.clip(mean_diff / 25.0, 0.0, 1.0))
        return round(motion_index, 3)

    def analyze_frame_engagement(
        self,
        frame_bgr: np.ndarray,
        detected_faces: List[Dict[str, Any]],
        expected_student_count: int = 1
    ) -> Dict[str, Any]:
        """
        Computes aggregate visual engagement metrics for a given video frame.
        """
        num_faces = len(detected_faces)
        expected = max(1, expected_student_count)

        # 1. Face Visibility Rate
        face_visibility_rate = float(np.clip(num_faces / expected, 0.0, 1.0))

        # 2. Head Pose Orientation Rate
        if num_faces == 0:
            head_pose_deviation_rate = 1.0  # Nobody facing camera
            deviated_count = 0
            student_poses = []
        else:
            deviated_count = 0
            student_poses = []
            for face in detected_faces:
                pose = self.estimate_head_pose_from_landmarks(face.get("landmarks"))
                student_poses.append(pose)
                if pose["is_deviated"]:
                    deviated_count += 1
            head_pose_deviation_rate = float(deviated_count / num_faces)

        # 3. Motion Index
        motion_index = self.compute_motion_index(frame_bgr)

        # Forward alignment rate is inverse of deviation rate
        forward_alignment_rate = max(0.0, 1.0 - head_pose_deviation_rate)

        return {
            "face_visibility_rate": round(face_visibility_rate, 3),
            "head_pose_deviation_rate": round(head_pose_deviation_rate, 3),
            "forward_alignment_rate": round(forward_alignment_rate, 3),
            "motion_index": motion_index,
            "num_faces": num_faces,
            "deviated_faces_count": deviated_count,
            "student_poses": student_poses
        }


class MultimodalFusionEngine:
    """
    Fuses visual and acoustic engagement streams into a unified classroom engagement metric.
    """

    def __init__(
        self,
        weight_face: float = WEIGHT_FACE_VISIBILITY,
        weight_head: float = WEIGHT_HEAD_ORIENTATION,
        weight_audio: float = WEIGHT_AUDIO_INTERACTION
    ):
        self.weight_face = weight_face
        self.weight_head = weight_head
        self.weight_audio = weight_audio

    def fuse(
        self,
        face_visibility_rate: float,
        forward_alignment_rate: float,
        audio_energy_level: float = 0.0,
        audio_enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Computes weighted fused engagement score [0.0 - 1.0] and categorizes level.
        """
        face_val = float(np.clip(face_visibility_rate, 0.0, 1.0))
        align_val = float(np.clip(forward_alignment_rate, 0.0, 1.0))
        audio_val = float(np.clip(audio_energy_level, 0.0, 1.0))

        if audio_enabled:
            fused_score = (
                (face_val * self.weight_face) +
                (align_val * self.weight_head) +
                (audio_val * self.weight_audio)
            )
        else:
            # Rescale visual weights to sum to 1.0
            total_vis = self.weight_face + self.weight_head
            if total_vis > 0:
                fused_score = (
                    (face_val * (self.weight_face / total_vis)) +
                    (align_val * (self.weight_head / total_vis))
                )
            else:
                fused_score = (face_val + align_val) / 2.0

        fused_score = float(np.clip(fused_score, 0.0, 1.0))

        if fused_score >= 0.70:
            level = "HIGH"
            color = "green"
        elif fused_score >= 0.40:
            level = "MODERATE"
            color = "orange"
        else:
            level = "LOW"
            color = "red"

        return {
            "fused_score": round(fused_score, 3),
            "level": level,
            "color": color,
            "components": {
                "face_visibility": round(face_val, 3),
                "head_alignment": round(align_val, 3),
                "audio_interaction": round(audio_val, 3) if audio_enabled else None
            }
        }

    def record_snapshot(
        self,
        session_id: str,
        visual_metrics: Dict[str, Any],
        audio_metrics: Optional[Dict[str, Any]] = None,
        audio_enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Fuses current frame metrics and records them into the SQLite database.
        """
        face_vis = visual_metrics.get("face_visibility_rate", 0.0)
        head_align = visual_metrics.get("forward_alignment_rate", 0.0)
        head_dev = visual_metrics.get("head_pose_deviation_rate", 0.0)
        motion = visual_metrics.get("motion_index", 0.0)

        if audio_enabled and audio_metrics:
            audio_energy = audio_metrics.get("energy_level", 0.0)
            is_active_speech = audio_metrics.get("is_active_speech", False)
        else:
            audio_energy = 0.0
            is_active_speech = False

        fusion_result = self.fuse(
            face_visibility_rate=face_vis,
            forward_alignment_rate=head_align,
            audio_energy_level=audio_energy,
            audio_enabled=audio_enabled
        )

        log_session_indicator(
            session_id=session_id,
            face_visibility=face_vis,
            head_pose_deviation=head_dev,
            motion_index=motion,
            audio_energy=audio_energy,
            is_active_speech=is_active_speech,
            fused_score=fusion_result["fused_score"]
        )

        return fusion_result
