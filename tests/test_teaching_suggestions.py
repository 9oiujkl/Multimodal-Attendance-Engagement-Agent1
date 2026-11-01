"""
Unit Tests for Pedagogical Suggestions Engine.
"""

import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.database import init_database, create_session, get_db_connection
from modules.teaching_suggestions import PedagogicalAdvisoryEngine


class TestTeachingSuggestions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_database()
        cls.session_id = "TEST_SUGGESTION_SESSION"
        create_session(cls.session_id, "Pedagogy Testing", "Dr. Maria Montessori")

    def setUp(self):
        self.engine = PedagogicalAdvisoryEngine(cooldown_seconds=10)
        with get_db_connection() as conn:
            conn.execute("DELETE FROM teaching_suggestions WHERE session_id = ?", (self.session_id,))
            conn.commit()

    def test_01_silence_suggestion_generation(self):
        """Tests that prolonged silence triggers an appropriate concept check suggestion."""
        visual_metrics = {"face_visibility_rate": 0.9, "head_pose_deviation_rate": 0.1, "motion_index": 0.1}
        audio_metrics = {"prolonged_silence_alert": True, "speech_ratio": 0.0}

        suggestions = self.engine.evaluate_indicators(
            session_id=self.session_id,
            visual_metrics=visual_metrics,
            audio_metrics=audio_metrics,
            fused_score=0.45
        )

        self.assertEqual(len(suggestions), 1)
        self.assertIn("Acoustic lull", suggestions[0]["observed_indicator"])
        self.assertIn("open-ended question", suggestions[0]["suggested_action"])

    def test_02_cooldown_suppression(self):
        """Tests that duplicate identical suggestions are suppressed during the cooldown window."""
        visual_metrics = {"face_visibility_rate": 0.9, "head_pose_deviation_rate": 0.8, "motion_index": 0.1}
        audio_metrics = {"prolonged_silence_alert": False, "speech_ratio": 0.5}

        # First evaluation generates head deviation prompt
        suggs1 = self.engine.evaluate_indicators(self.session_id, visual_metrics, audio_metrics, 0.40)
        self.assertEqual(len(suggs1), 1)

        # Immediate second evaluation inside cooldown should return empty
        suggs2 = self.engine.evaluate_indicators(self.session_id, visual_metrics, audio_metrics, 0.40)
        self.assertEqual(len(suggs2), 0)

    def test_03_database_persistence_and_retrieval(self):
        """Verifies that generated suggestions are recorded and retrievable from the database."""
        visual_metrics = {"face_visibility_rate": 0.95, "head_pose_deviation_rate": 0.05, "motion_index": 0.15}
        # Trigger high engagement
        self.engine.evaluate_indicators(self.session_id, visual_metrics, None, fused_score=0.88)

        history = self.engine.get_history(self.session_id)
        self.assertGreaterEqual(len(history), 1)
        self.assertIn("optimal", history[0]["suggested_action"].lower())


if __name__ == "__main__":
    unittest.main()
