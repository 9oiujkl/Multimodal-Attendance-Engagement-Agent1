"""
Unit Tests for Engagement Analysis, Audio Signal Processing & Multimodal Fusion.
"""

import unittest
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.engagement_analysis import (
    VisualEngagementAnalyzer,
    MultimodalFusionEngine
)
from modules.audio_analysis import (
    compute_rms,
    compute_zcr,
    AcousticAnalyzer,
    generate_synthetic_audio
)


class TestEngagementAndAudio(unittest.TestCase):

    def setUp(self):
        self.visual_analyzer = VisualEngagementAnalyzer()
        self.fusion_engine = MultimodalFusionEngine()
        self.audio_analyzer = AcousticAnalyzer(prolonged_silence_sec=2.0)  # Shorter for fast testing

    def test_01_head_pose_estimation(self):
        """Tests landmark-based yaw and pitch deviation calculation."""
        # Forward aligned simulated landmarks
        # Right eye, Left eye, Nose, Mouth right, Mouth left
        forward_landmarks = [
            (100.0, 100.0),  # Right eye
            (140.0, 100.0),  # Left eye
            (120.0, 120.0),  # Nose (centered between eyes)
            (105.0, 140.0),  # Mouth right
            (135.0, 140.0)   # Mouth left
        ]
        res_forward = self.visual_analyzer.estimate_head_pose_from_landmarks(forward_landmarks)
        self.assertFalse(res_forward["is_deviated"])
        self.assertLess(abs(res_forward["approx_yaw_deg"]), 15.0)

        # Heavily turned head (yaw deviation)
        deviated_landmarks = [
            (90.0, 100.0),   # Right eye
            (105.0, 100.0),  # Left eye
            (95.0, 120.0),   # Nose heavily shifted to one side
            (92.0, 140.0),   # Mouth right
            (102.0, 140.0)   # Mouth left
        ]
        res_deviated = self.visual_analyzer.estimate_head_pose_from_landmarks(deviated_landmarks)
        self.assertTrue(res_deviated["is_deviated"])

    def test_02_motion_index_differencing(self):
        """Tests frame differencing motion detection."""
        f1 = np.zeros((100, 100, 3), dtype=np.uint8)
        f2 = np.zeros((100, 100, 3), dtype=np.uint8)

        # First frame initializes reference
        m0 = self.visual_analyzer.compute_motion_index(f1)
        self.assertEqual(m0, 0.0)

        # Identical second frame has 0 motion
        m1 = self.visual_analyzer.compute_motion_index(f2)
        self.assertEqual(m1, 0.0)

        # Mutated frame has positive motion
        f3 = np.ones((100, 100, 3), dtype=np.uint8) * 150
        m2 = self.visual_analyzer.compute_motion_index(f3)
        self.assertGreater(m2, 0.1)

    def test_03_multimodal_fusion(self):
        """Tests score computation with and without audio modality."""
        # High engagement scenario
        high_res = self.fusion_engine.fuse(
            face_visibility_rate=1.0,
            forward_alignment_rate=1.0,
            audio_energy_level=0.8,
            audio_enabled=True
        )
        self.assertGreaterEqual(high_res["fused_score"], 0.70)
        self.assertEqual(high_res["level"], "HIGH")

        # Low engagement scenario
        low_res = self.fusion_engine.fuse(
            face_visibility_rate=0.2,
            forward_alignment_rate=0.1,
            audio_energy_level=0.0,
            audio_enabled=True
        )
        self.assertLess(low_res["fused_score"], 0.40)
        self.assertEqual(low_res["level"], "LOW")

        # Audio disabled mode (weights re-normalize to 1.0)
        no_audio_res = self.fusion_engine.fuse(
            face_visibility_rate=0.8,
            forward_alignment_rate=0.8,
            audio_enabled=False
        )
        self.assertAlmostEqual(no_audio_res["fused_score"], 0.80, places=2)

    def test_04_audio_rms_and_zcr(self):
        """Tests acoustic mathematical formulas on pure sinusoids and silence."""
        silence = np.zeros(1000, dtype=np.float32)
        self.assertEqual(compute_rms(silence), 0.0)
        self.assertEqual(compute_zcr(silence), 0.0)

        # Sine wave with amplitude 0.5 -> RMS = 0.5 / sqrt(2) ~ 0.3535
        t = np.linspace(0, 1, 1000, endpoint=False)
        sine = (0.5 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)
        rms_val = compute_rms(sine)
        self.assertAlmostEqual(rms_val, 0.3535, places=2)

    def test_05_prolonged_silence_detection(self):
        """Tests silence alert triggering after prolonged quiet interval."""
        sr, silence_audio = generate_synthetic_audio(duration_sec=3.0, scenario="prolonged_silence")
        # Process in 0.5 second chunks
        chunk_len = int(sr * 0.5)
        alert_fired = False

        for i in range(0, len(silence_audio), chunk_len):
            chunk = silence_audio[i:i + chunk_len]
            metrics = self.audio_analyzer.process_chunk(chunk, sample_rate=sr)
            if metrics["prolonged_silence_alert"]:
                alert_fired = True

        self.assertTrue(alert_fired, "Prolonged silence alert should have triggered after 2.0s silence")


if __name__ == "__main__":
    unittest.main()
