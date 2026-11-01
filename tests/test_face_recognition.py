"""
Unit Tests for Face Recognition & Attendance Tracking Module.
"""

import unittest
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.face_recognition_attendance import (
    cosine_similarity,
    FaceAttendanceSystem
)
from modules.database import init_database


class TestFaceRecognition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_database()
        cls.system = FaceAttendanceSystem(sustained_frames_required=3)

    def test_01_cosine_similarity(self):
        """Tests mathematical correctness of cosine similarity calculation."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v4 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        # Identical vectors -> 1.0
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0, places=5)
        # Orthogonal vectors -> 0.0
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0, places=5)
        # Opposite vectors -> -1.0
        self.assertAlmostEqual(cosine_similarity(v1, v4), -1.0, places=5)
        # Zero vector -> 0.0 (safe guard against divide-by-zero)
        v_zero = np.zeros(3, dtype=np.float32)
        self.assertEqual(cosine_similarity(v1, v_zero), 0.0)

    def test_02_match_face_against_roster(self):
        """Tests roster lookup and threshold matching."""
        target_vec = np.random.randn(128).astype(np.float32)
        target_vec /= np.linalg.norm(target_vec)

        other_vec = np.random.randn(128).astype(np.float32)
        other_vec /= np.linalg.norm(other_vec)

        roster = {
            "STUDENT-001": other_vec,
            "STUDENT-002": target_vec
        }

        # Query with exact target vector
        matched_id, score = self.system.match_face(target_vec, roster, threshold=0.363)
        self.assertEqual(matched_id, "STUDENT-002")
        self.assertGreater(score, 0.99)

        # Query with an unregistered random vector
        unregistered_vec = np.random.randn(128).astype(np.float32)
        unregistered_vec /= np.linalg.norm(unregistered_vec)
        # Highly unlikely to match above 0.8
        matched_unreg, _ = self.system.match_face(unregistered_vec, roster, threshold=0.90)
        self.assertIsNone(matched_unreg)

    def test_03_sustained_detection_tracking(self):
        """Tests that attendance requires N consecutive frames before verification."""
        self.system.reset_session_cache()
        student_id = "TEST-STREAK-001"
        embedding = np.random.randn(128).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        roster = {student_id: embedding}

        # Simulate frame 1 match
        self.system.detection_streak[student_id] = 1
        self.assertLess(self.system.detection_streak[student_id], self.system.sustained_frames_required)

        # Simulate frame 2 match
        self.system.detection_streak[student_id] += 1
        self.assertLess(self.system.detection_streak[student_id], self.system.sustained_frames_required)

        # Simulate frame 3 match (meets threshold = 3)
        self.system.detection_streak[student_id] += 1
        self.assertGreaterEqual(self.system.detection_streak[student_id], self.system.sustained_frames_required)

        # Reset cache
        self.system.reset_session_cache()
        self.assertEqual(len(self.system.detection_streak), 0)


if __name__ == "__main__":
    unittest.main()
