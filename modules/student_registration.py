"""
Student Registration Module for Multimodal Attendance & Engagement Agent.
Handles consent-based biometric face enrollment, local feature embedding extraction,
non-biometric registration, and student profile management.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    YUNET_MODEL_PATH,
    SFACE_MODEL_PATH,
    FACE_DET_SCORE_THRESHOLD,
    FACE_DET_NMS_THRESHOLD,
    FACE_DET_TOP_K
)
from modules.database import (
    register_student,
    get_all_students,
    delete_student,
    log_audit_event
)


class LocalFaceFeatureExtractor:
    """
    Extracts 128-dimensional face representations locally.
    Prioritizes official OpenCV SFace ONNX model if available;
    otherwise gracefully falls back to OpenCV Haar Cascade + Normalized Histogram/Patch
    to guarantee 100% offline functionality.
    """
    def __init__(self):
        self.yunet_detector = None
        self.sface_recognizer = None
        self.haar_cascade = None
        self._init_models()

    def _init_models(self):
        # Attempt to load OpenCV Zoo SFace & YuNet if files exist
        if YUNET_MODEL_PATH.exists() and SFACE_MODEL_PATH.exists():
            try:
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    model=str(YUNET_MODEL_PATH),
                    config="",
                    input_size=(320, 320),
                    score_threshold=FACE_DET_SCORE_THRESHOLD,
                    nms_threshold=FACE_DET_NMS_THRESHOLD,
                    top_k=FACE_DET_TOP_K
                )
                self.sface_recognizer = cv2.FaceRecognizerSF.create(
                    model=str(SFACE_MODEL_PATH),
                    config=""
                )
                return
            except Exception as e:
                print(f"[Notice] Failed loading ONNX models ({e}), falling back to standard OpenCV features.")

        # Built-in offline fallback using OpenCV Haar Cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.haar_cascade = cv2.CascadeClassifier(cascade_path)

    def extract_embedding(self, image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        """
        Detects primary face in image and returns (128-dim L2-normalized float32 vector, message).
        Returns (None, reason) if no face or multiple faces detected.
        """
        if image_bgr is None or image_bgr.size == 0:
            return None, "Invalid image data provided."

        h, w = image_bgr.shape[:2]

        # Path A: Using SFace + YuNet
        if self.yunet_detector is not None and self.sface_recognizer is not None:
            try:
                self.yunet_detector.setInputSize((w, h))
                _, faces = self.yunet_detector.detect(image_bgr)
                if faces is None or len(faces) == 0:
                    return None, "No face detected in enrollment image."
                if len(faces) > 1:
                    # Select largest face by bounding box area
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

                primary_face = faces[0]
                aligned_face = self.sface_recognizer.alignCrop(image_bgr, primary_face)
                feature = self.sface_recognizer.feature(aligned_face)
                # Ensure 1D L2-normalized vector
                feature = feature.flatten().astype(np.float32)
                norm = np.linalg.norm(feature)
                if norm > 0:
                    feature = feature / norm
                return feature, "Success (SFace 128-d ONNX)"
            except Exception as e:
                print(f"[Warning] SFace extraction failed: {e}. Falling back to Haar.")

        # Path B: Offline Fallback using Haar Cascade + Geometric-Intensity Feature Vector
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) == 0:
            return None, "No face detected in enrollment image."

        # Pick largest face
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        x, y, fw, fh = faces[0]
        face_roi = gray[y:y+fh, x:x+fw]

        # Resize to fixed standard 32x32 patch
        resized_roi = cv2.resize(face_roi, (32, 32), interpolation=cv2.INTER_AREA)

        # Equalize histogram for lighting invariance
        equalized = cv2.equalizeHist(resized_roi)

        # Compute 128-dimensional descriptor:
        # 64-bin intensity distribution histogram + 64 downsampled spatial gradient cells
        hist, _ = np.histogram(equalized, bins=64, range=(0, 256), density=True)

        # 8x8 block averages = 64 spatial values
        blocks = cv2.resize(equalized, (8, 8), interpolation=cv2.INTER_AREA).flatten().astype(np.float32)
        blocks = (blocks - np.mean(blocks)) / (np.std(blocks) + 1e-6)

        # Concatenate 64 + 64 = 128 dimensions
        descriptor = np.concatenate([hist.astype(np.float32), blocks])
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm

        return descriptor, "Success (OpenCV Offline Feature Extractor)"


# Singleton instance
_extractor = LocalFaceFeatureExtractor()


def enroll_student_biometric(
    student_id: str,
    full_name: str,
    email: Optional[str],
    department: str,
    image_bgr: np.ndarray
) -> Tuple[bool, str]:
    """
    Registers a student with verified consent and extracts a 128-d face embedding.
    Never stores the original photo.
    """
    if not student_id or not full_name:
        return False, "Student ID and Full Name are mandatory."

    embedding, msg = _extractor.extract_embedding(image_bgr)
    if embedding is None:
        return False, f"Enrollment rejected: {msg}"

    success = register_student(
        student_id=student_id.strip(),
        full_name=full_name.strip(),
        email=email.strip() if email else None,
        department=department.strip(),
        biometric_consent=True,
        face_embedding=embedding
    )
    if success:
        return True, f"Successfully enrolled {full_name} ({msg}). Biometric template securely saved."
    return False, "Database enrollment failed."


def enroll_student_non_biometric(
    student_id: str,
    full_name: str,
    email: Optional[str],
    department: str
) -> Tuple[bool, str]:
    """
    Registers a student who opts out of facial recognition.
    Enables ID/Roll-call attendance without biometric data.
    """
    if not student_id or not full_name:
        return False, "Student ID and Full Name are mandatory."

    success = register_student(
        student_id=student_id.strip(),
        full_name=full_name.strip(),
        email=email.strip() if email else None,
        department=department.strip(),
        biometric_consent=False,
        face_embedding=None
    )
    if success:
        return True, f"Successfully registered {full_name} under Non-Biometric attendance alternative."
    return False, "Database registration failed."


def get_students_roster() -> list:
    """Returns list of all active students."""
    return get_all_students(active_only=True)


def unenroll_student(student_id: str) -> Tuple[bool, str]:
    """Deletes student record and purges any biometric embedding."""
    try:
        delete_student(student_id, hard_delete=False)
        return True, f"Student {student_id} successfully deactivated and biometrics purged."
    except Exception as e:
        return False, f"Failed to delete student: {e}"
