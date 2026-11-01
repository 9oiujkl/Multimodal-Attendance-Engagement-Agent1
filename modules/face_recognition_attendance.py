"""
Face Recognition & Attendance Tracking Module.
Performs real-time multi-face detection, SFace 128-d cosine matching,
and temporal sustained detection verification before recording attendance.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    YUNET_MODEL_PATH,
    SFACE_MODEL_PATH,
    FACE_DET_SCORE_THRESHOLD,
    FACE_DET_NMS_THRESHOLD,
    FACE_DET_TOP_K,
    COSINE_SIMILARITY_THRESHOLD,
    SUSTAINED_DETECTION_FRAMES
)
from modules.database import record_or_update_attendance


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two 1D vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


class FaceAttendanceSystem:
    """
    Handles face detection, embedding matching against enrolled students,
    sustained frame buffer temporal stability checks, and attendance persistence.
    """

    def __init__(self, sustained_frames_required: int = SUSTAINED_DETECTION_FRAMES):
        self.sustained_frames_required = sustained_frames_required
        self.yunet = None
        self.sface = None
        self.haar = None

        # Tracks consecutive frames a student is observed: {student_id: count}
        self.detection_streak: Dict[str, int] = {}
        # Tracks validated students in current session: {student_id: first_validated_datetime}
        self.validated_students: Dict[str, datetime] = {}

        self._init_models()

    def _init_models(self):
        """Initializes ONNX models if available; else falls back to OpenCV Haar."""
        if YUNET_MODEL_PATH.exists() and SFACE_MODEL_PATH.exists():
            try:
                self.yunet = cv2.FaceDetectorYN.create(
                    model=str(YUNET_MODEL_PATH),
                    config="",
                    input_size=(320, 320),
                    score_threshold=FACE_DET_SCORE_THRESHOLD,
                    nms_threshold=FACE_DET_NMS_THRESHOLD,
                    top_k=FACE_DET_TOP_K
                )
                self.sface = cv2.FaceRecognizerSF.create(
                    model=str(SFACE_MODEL_PATH),
                    config=""
                )
                return
            except Exception as e:
                print(f"[Warning] Failed to initialize ONNX models: {e}. Falling back to Haar.")

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in image_bgr.
        Returns a list of dictionaries with 'box': (x, y, w, h), 'landmarks': (optional 5 points), 'raw_face': face representation.
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        detected = []

        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(image_bgr)
            if faces is not None and len(faces) > 0:
                for face in faces:
                    x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    score = float(face[14])
                    # 5 landmarks: right_eye, left_eye, nose, mouth_right, mouth_left
                    landmarks = [
                        (float(face[4]), float(face[5])),
                        (float(face[6]), float(face[7])),
                        (float(face[8]), float(face[9])),
                        (float(face[10]), float(face[11])),
                        (float(face[12]), float(face[13]))
                    ]
                    detected.append({
                        "box": (max(0, x), max(0, y), fw, fh),
                        "score": score,
                        "landmarks": landmarks,
                        "raw_face": face
                    })
                return detected

        # Fallback using Haar Cascade
        if self.haar is not None:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            for (x, y, fw, fh) in faces:
                detected.append({
                    "box": (x, y, fw, fh),
                    "score": 0.90,
                    "landmarks": None,
                    "raw_face": (x, y, fw, fh)
                })

        return detected

    def extract_feature(self, image_bgr: np.ndarray, face_info: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extracts 128-dimensional embedding from detected face."""
        if self.sface is not None and face_info.get("raw_face") is not None and len(face_info["raw_face"]) >= 15:
            try:
                aligned = self.sface.alignCrop(image_bgr, face_info["raw_face"])
                feat = self.sface.feature(aligned).flatten().astype(np.float32)
                norm = np.linalg.norm(feat)
                if norm > 0:
                    feat = feat / norm
                return feat
            except Exception:
                pass

        # Offline fallback feature extractor: 64-bin histogram + 64 block spatial averages
        x, y, fw, fh = face_info["box"]
        h, w = image_bgr.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + fw), min(h, y + fh)
        if x2 <= x1 or y2 <= y1:
            return None

        roi = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
        equalized = cv2.equalizeHist(resized)
        hist, _ = np.histogram(equalized, bins=64, range=(0, 256), density=True)
        blocks = cv2.resize(equalized, (8, 8), interpolation=cv2.INTER_AREA).flatten().astype(np.float32)
        blocks = (blocks - np.mean(blocks)) / (np.std(blocks) + 1e-6)
        desc = np.concatenate([hist.astype(np.float32), blocks])
        norm = np.linalg.norm(desc)
        if norm > 0:
            desc = desc / norm
        return desc

    def match_face(
        self,
        embedding: np.ndarray,
        enrolled_embeddings: Dict[str, np.ndarray],
        threshold: float = COSINE_SIMILARITY_THRESHOLD
    ) -> Tuple[Optional[str], float]:
        """
        Compares face embedding against all enrolled student embeddings.
        Returns (best_student_id, best_similarity_score) if above threshold, else (None, best_similarity_score).
        """
        if not enrolled_embeddings or embedding is None:
            return None, 0.0

        best_student = None
        best_score = -1.0

        for student_id, enrolled_vec in enrolled_embeddings.items():
            sim = cosine_similarity(embedding, enrolled_vec)
            if sim > best_score:
                best_score = sim
                best_student = student_id

        if best_score >= threshold:
            return best_student, best_score
        return None, best_score

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        session_id: Optional[str],
        enrolled_embeddings: Dict[str, np.ndarray],
        persist_attendance: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Processes a single camera/video frame:
        1. Detects faces
        2. Matches against enrolled roster
        3. Updates sustained temporal detection streaks
        4. Persists verified attendance to SQLite database
        5. Annotates bounding boxes and status HUD onto the frame.

        Returns (annotated_frame, list_of_detections, summary_stats).
        """
        annotated = frame_bgr.copy()
        h, w = frame_bgr.shape[:2]
        now = datetime.now()

        faces = self.detect_faces(frame_bgr)
        seen_in_this_frame = set()
        detections_info = []

        for face_info in faces:
            x, y, fw, fh = face_info["box"]
            embedding = self.extract_feature(frame_bgr, face_info)
            student_id, sim_score = self.match_face(embedding, enrolled_embeddings)

            is_confirmed = False
            streak_count = 0

            if student_id:
                seen_in_this_frame.add(student_id)
                self.detection_streak[student_id] = self.detection_streak.get(student_id, 0) + 1
                streak_count = self.detection_streak[student_id]

                if streak_count >= self.sustained_frames_required:
                    is_confirmed = True
                    if student_id not in self.validated_students:
                        self.validated_students[student_id] = now

                    if persist_attendance and session_id:
                        record_or_update_attendance(
                            session_id=session_id,
                            student_id=student_id,
                            detection_time=now,
                            sustained_count=streak_count,
                            status="PRESENT"
                        )

            # Visual annotations
            if is_confirmed:
                box_color = (0, 220, 0)      # Solid Green for Verified Present
                label = f"{student_id} (VERIFIED: {sim_score:.2f})"
            elif student_id:
                box_color = (0, 215, 255)    # Gold/Yellow for Verifying Stability Buffer
                label = f"{student_id} (Buffering {streak_count}/{self.sustained_frames_required})"
            else:
                box_color = (60, 60, 220)    # Soft Red/Orange for Unregistered / Unknown
                label = f"Unknown ({sim_score:.2f})" if sim_score > 0 else "Unknown"

            # Draw rectangle and background banner
            cv2.rectangle(annotated, (x, y), (x + fw, y + fh), box_color, 2)
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x, max(0, y - text_h - 6)), (x + text_w + 6, max(0, y)), box_color, -1)
            cv2.putText(
                annotated,
                label,
                (x + 3, max(0, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # Draw landmarks if available
            landmarks = face_info.get("landmarks")
            if landmarks:
                for lx, ly in landmarks:
                    cv2.circle(annotated, (int(lx), int(ly)), 2, (0, 255, 255), -1)

            detections_info.append({
                "student_id": student_id,
                "confidence": sim_score,
                "is_confirmed": is_confirmed,
                "streak": streak_count,
                "box": (x, y, fw, fh),
                "landmarks": landmarks
            })

        # Decay streaks for students not observed in this frame
        for student_id in list(self.detection_streak.keys()):
            if student_id not in seen_in_this_frame:
                self.detection_streak[student_id] = max(0, self.detection_streak[student_id] - 1)

        summary = {
            "total_faces_detected": len(faces),
            "recognized_count": len([d for d in detections_info if d["student_id"] is not None]),
            "confirmed_present_count": len([d for d in detections_info if d["is_confirmed"]]),
            "timestamp": now.isoformat()
        }

        return annotated, detections_info, summary

    def reset_session_cache(self):
        """Clears local frame streak buffer and validated cache for a new session."""
        self.detection_streak.clear()
        self.validated_students.clear()
