"""
Unit Tests for Reporting & PDF Generation Module.
"""

import unittest
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import REPORTS_DIR
from modules.database import (
    init_database,
    create_session,
    register_student,
    record_or_update_attendance,
    log_session_indicator,
    log_teaching_suggestion
)
from modules.reporting import (
    get_session_summary_dataframe,
    export_attendance_csv,
    generate_session_pdf_report
)


class TestReporting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_database()
        cls.session_id = "TEST_REPORT_SESSION"
        create_session(cls.session_id, "Data Science 101", "Dr. Grace Hopper")

        # Setup test students and attendance
        register_student("TEST-DS-01", "Ada Lovelace", "ada@example.edu", "MCA", biometric_consent=True)
        register_student("TEST-DS-02", "Charles Babbage", "charles@example.edu", "MCA", biometric_consent=False)

        from datetime import datetime
        now = datetime.now()
        record_or_update_attendance(cls.session_id, "TEST-DS-01", now, sustained_count=5, status="PRESENT")
        log_session_indicator(cls.session_id, 0.85, 0.15, 0.20, 0.40, True, 0.80)
        log_teaching_suggestion(
            cls.session_id,
            "High focus",
            "Optimal pacing",
            "Maintain current delivery pace",
            "High confidence"
        )

    def test_01_summary_dataframe_structure(self):
        """Verifies session attendance DataFrame formatting and columns."""
        df = get_session_summary_dataframe(self.session_id)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreaterEqual(len(df), 2)
        expected_cols = {"student_id", "full_name", "status", "biometric_consent", "is_manual_override"}
        self.assertTrue(expected_cols.issubset(set(df.columns)))

    def test_02_export_attendance_csv(self):
        """Tests that CSV export produces valid CSV file on disk."""
        csv_path = REPORTS_DIR / f"test_attendance_{self.session_id}.csv"
        exported_path = export_attendance_csv(self.session_id, csv_path)
        self.assertTrue(exported_path.exists())
        self.assertGreater(exported_path.stat().st_size, 50)

        # Read back and verify
        df_read = pd.read_csv(exported_path)
        self.assertIn("Ada Lovelace", df_read["full_name"].values)

    def test_03_generate_session_pdf_report(self):
        """Tests that ReportLab generates a complete, non-empty PDF report."""
        pdf_path = REPORTS_DIR / f"test_report_{self.session_id}.pdf"
        generated_path = generate_session_pdf_report(self.session_id, pdf_path)
        self.assertTrue(generated_path.exists())
        # A valid PDF with tables and headers will be at least 2 KB
        self.assertGreater(generated_path.stat().st_size, 2048)


if __name__ == "__main__":
    unittest.main()
