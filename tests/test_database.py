"""
Unit Test Suite for SQLite Database & Student Registration Operations.
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATABASE_PATH
from modules.database import (
    init_database,
    get_db_connection,
    register_student,
    get_all_students,
    get_enrolled_embeddings,
    delete_student,
    create_session,
    end_session,
    record_or_update_attendance,
    manual_attendance_override,
    get_session_attendance,
    log_session_indicator,
    get_session_indicators,
    log_teaching_suggestion,
    get_session_suggestions
)
from modules.student_registration import (
    enroll_student_non_biometric,
    unenroll_student
)


class TestDatabaseAndRegistration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initializes database schema before running tests."""
        init_database()

    def setUp(self):
        """Cleans test rows before each test to maintain test isolation."""
        with get_db_connection() as conn:
            conn.execute("DELETE FROM audit_logs WHERE entity_id LIKE 'TEST%';")
            conn.execute("DELETE FROM teaching_suggestions WHERE session_id LIKE 'TEST%';")
            conn.execute("DELETE FROM session_indicators WHERE session_id LIKE 'TEST%';")
            conn.execute("DELETE FROM attendance WHERE session_id LIKE 'TEST%';")
            conn.execute("DELETE FROM sessions WHERE session_id LIKE 'TEST%';")
            conn.execute("DELETE FROM students WHERE student_id LIKE 'TEST%';")
            conn.commit()

    def test_01_init_database_tables_exist(self):
        """Verifies that all 6 tables exist in the SQLite database."""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = {row[0] for row in cursor.fetchall()}

        expected_tables = {
            "students",
            "sessions",
            "attendance",
            "session_indicators",
            "teaching_suggestions",
            "audit_logs"
        }
        for table in expected_tables:
            self.assertIn(table, tables, f"Expected table '{table}' missing from database.")

    def test_02_biometric_and_non_biometric_student_registration(self):
        """Tests registration with dummy face embedding and non-biometric registration."""
        # 1. Non-biometric registration
        success, msg = enroll_student_non_biometric(
            student_id="TEST-MCA-001",
            full_name="Alice Smith",
            email="alice@example.edu",
            department="MCA"
        )
        self.assertTrue(success)

        # 2. Biometric registration with a 128-d dummy normalized vector
        dummy_embedding = np.random.randn(128).astype(np.float32)
        dummy_embedding /= np.linalg.norm(dummy_embedding)

        reg_success = register_student(
            student_id="TEST-MCA-002",
            full_name="Bob Jones",
            email="bob@example.edu",
            department="MCA",
            biometric_consent=True,
            face_embedding=dummy_embedding
        )
        self.assertTrue(reg_success)

        # Verify active roster
        roster = get_all_students(active_only=True)
        student_ids = [s["student_id"] for s in roster]
        self.assertIn("TEST-MCA-001", student_ids)
        self.assertIn("TEST-MCA-002", student_ids)

        # Verify embeddings retrieval
        embeddings_dict = get_enrolled_embeddings()
        self.assertNotIn("TEST-MCA-001", embeddings_dict, "Non-biometric student should not have embedding")
        self.assertIn("TEST-MCA-002", embeddings_dict)
        self.assertEqual(embeddings_dict["TEST-MCA-002"].shape, (128,))

    def test_03_session_lifecycle_and_attendance_tracking(self):
        """Tests session creation, attendance recording, and duration calculation."""
        session_id = "TEST_SESSION_101"
        student_id = "TEST-MCA-003"

        # Register student
        register_student(student_id, "Charlie Brown", "charlie@example.edu", "MCA", biometric_consent=False)

        # Create session
        created = create_session(session_id, "AI Lab", "Dr. Alan Turing")
        self.assertTrue(created)

        # First observation at T0
        t0 = datetime.now()
        record_or_update_attendance(session_id, student_id, t0, sustained_count=1)

        records = get_session_attendance(session_id)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["student_id"], student_id)
        self.assertEqual(records[0]["status"], "PRESENT")

        # Second observation at T0 + 10 seconds
        t1 = t0 + timedelta(seconds=10)
        record_or_update_attendance(session_id, student_id, t1, sustained_count=5)

        updated_records = get_session_attendance(session_id)
        self.assertAlmostEqual(updated_records[0]["duration_seconds"], 10.0, delta=1.0)

        # End session
        self.assertTrue(end_session(session_id))

    def test_04_manual_attendance_override(self):
        """Tests that instructors can manually correct attendance with audit logging."""
        session_id = "TEST_SESSION_102"
        student_id = "TEST-MCA-004"

        register_student(student_id, "Diana Prince", "diana@example.edu", "MCA")
        create_session(session_id, "Computer Vision", "Prof. Feifei Li")

        # Manual mark as PRESENT (e.g. for non-biometric student present via roll-call)
        success = manual_attendance_override(
            session_id=session_id,
            student_id=student_id,
            new_status="PRESENT",
            reason="Verified in person via student ID card",
            admin_name="Prof. Feifei Li"
        )
        self.assertTrue(success)

        records = get_session_attendance(session_id)
        match = [r for r in records if r["student_id"] == student_id][0]
        self.assertEqual(match["status"], "PRESENT")
        self.assertEqual(match["is_manual_override"], 1)
        self.assertEqual(match["override_reason"], "Verified in person via student ID card")

    def test_05_indicators_and_teaching_suggestions_logging(self):
        """Tests logging and retrieval of multimodal indicators and pedagogical suggestions."""
        session_id = "TEST_SESSION_103"
        create_session(session_id, "Machine Learning", "Dr. Andrew Ng")

        # Log indicator
        log_session_indicator(
            session_id=session_id,
            face_visibility=0.85,
            head_pose_deviation=0.12,
            motion_index=0.20,
            audio_energy=0.35,
            is_active_speech=True,
            fused_score=0.82
        )

        indicators = get_session_indicators(session_id)
        self.assertEqual(len(indicators), 1)
        self.assertAlmostEqual(indicators[0]["fused_engagement_score"], 0.82, places=2)

        # Log teaching suggestion
        log_teaching_suggestion(
            session_id=session_id,
            observed_indicator="Audio silence > 25 seconds after lecture topic change",
            pedagogical_reason="Learners may need a moment to digest or formulate questions",
            suggested_action="Ask an open-ended concept check question",
            confidence_note="High confidence (audio energy < 0.015 for 25s)"
        )

        suggestions = get_session_suggestions(session_id)
        self.assertEqual(len(suggestions), 1)
        self.assertIn("open-ended concept check", suggestions[0]["suggested_action"])


if __name__ == "__main__":
    unittest.main()
