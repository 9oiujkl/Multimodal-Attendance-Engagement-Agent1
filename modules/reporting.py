"""
Reporting & Analytics Export Module.
Generates CSV spreadsheets, analytics summaries, and publication-ready PDF reports
using ReportLab with full attendance records, engagement timelines,
pedagogical logs, and privacy compliance disclosures.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import REPORTS_DIR, ETHICAL_DISCLAIMER
from modules.database import (
    get_session_attendance,
    get_session_indicators,
    get_session_suggestions,
    get_all_sessions
)


def get_session_summary_dataframe(session_id: str) -> pd.DataFrame:
    """Retrieves session attendance records and formats them as a clean pandas DataFrame."""
    records = get_session_attendance(session_id)
    if not records:
        return pd.DataFrame(columns=[
            "student_id", "full_name", "department", "biometric_consent",
            "status", "first_seen", "last_seen", "duration_seconds", "is_manual_override", "override_reason"
        ])

    df = pd.DataFrame(records)
    df["biometric_consent"] = df["biometric_consent"].apply(lambda x: "Yes" if x else "Opt-Out")
    df["is_manual_override"] = df["is_manual_override"].apply(lambda x: "Yes" if x else "No")
    df["duration_minutes"] = (df["duration_seconds"] / 60.0).round(1)
    return df


def get_indicators_dataframe(session_id: str) -> pd.DataFrame:
    """Retrieves timeline telemetry records as a pandas DataFrame."""
    records = get_session_indicators(session_id)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df


def export_attendance_csv(session_id: str, output_path: Optional[Path] = None) -> Path:
    """Exports session attendance to a CSV file."""
    df = get_session_summary_dataframe(session_id)
    if output_path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPORTS_DIR / f"attendance_{session_id}.csv"
    df.to_csv(output_path, index=False)
    return output_path


def generate_session_pdf_report(session_id: str, output_path: Optional[Path] = None) -> Path:
    """
    Builds a professional, comprehensive PDF session report with:
    - Session metadata & Instructor info
    - Executive attendance & engagement KPIs
    - Complete student attendance roster
    - Observable engagement telemetry summary
    - AI-assisted teaching suggestions log
    - Strict privacy & ethical disclaimer
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = REPORTS_DIR / f"report_{session_id}.pdf"

    # Fetch session info
    sessions = get_all_sessions()
    session_info = next((s for s in sessions if s["session_id"] == session_id), None)
    course_name = session_info["course_name"] if session_info else "Classroom Session"
    instructor_name = session_info["instructor_name"] if session_info else "Instructor"
    start_time = session_info["start_time"] if session_info else datetime.now().isoformat()
    end_time = session_info.get("end_time") or "In Progress"

    # Fetch attendance and indicators
    att_df = get_session_summary_dataframe(session_id)
    indicators = get_session_indicators(session_id)
    suggestions = get_session_suggestions(session_id)

    # Compute KPIs
    total_students = len(att_df)
    present_count = len(att_df[att_df["status"] == "PRESENT"]) if total_students > 0 else 0
    att_rate = (present_count / total_students * 100) if total_students > 0 else 0.0
    override_count = len(att_df[att_df["is_manual_override"] == "Yes"]) if total_students > 0 else 0

    avg_engagement = (
        float(pd.DataFrame(indicators)["fused_engagement_score"].mean() * 100)
        if indicators and len(indicators) > 0 else 0.0
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2D3748")
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#718096"),
        fontName="Helvetica-Oblique"
    )

    elements = []

    # Title Banner
    elements.append(Paragraph("Multimodal Attendance & Engagement Audit Report", title_style))
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Confidential Educational Audit Record", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=12))

    # Session Metadata Table
    meta_data = [
        [Paragraph("<b>Course:</b>", body_style), Paragraph(str(course_name), body_style),
         Paragraph("<b>Session ID:</b>", body_style), Paragraph(str(session_id), body_style)],
        [Paragraph("<b>Instructor:</b>", body_style), Paragraph(str(instructor_name), body_style),
         Paragraph("<b>Start Time:</b>", body_style), Paragraph(str(start_time)[:19], body_style)],
        [Paragraph("<b>Status:</b>", body_style), Paragraph(str(session_info.get("status", "ACTIVE") if session_info else "ACTIVE"), body_style),
         Paragraph("<b>End Time:</b>", body_style), Paragraph(str(end_time)[:19], body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[70, 195, 75, 195])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, -1), (-1, -1), 1, colors.HexColor("#E2E8F0")),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 10))

    # KPI Summary Cards
    elements.append(Paragraph("Executive Summary & Key Performance Indicators", h2_style))
    kpi_data = [
        ["Total Enrolled", "Present Students", "Attendance Rate", "Avg Fused Engagement", "Manual Overrides"],
        [f"{total_students}", f"{present_count}", f"{att_rate:.1f}%", f"{avg_engagement:.1f}%", f"{override_count}"]
    ]
    kpi_table = Table(kpi_data, colWidths=[105, 105, 105, 115, 105])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, 1), 12),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#EDF2F7")),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 12))

    # Attendance Roster Table
    elements.append(Paragraph("Student Attendance Verification Roster", h2_style))
    if total_students == 0:
        elements.append(Paragraph("No student records available for this session.", body_style))
    else:
        roster_rows = [
            ["ID", "Full Name", "Dept", "Consent", "Status", "Duration", "Override", "Notes"]
        ]
        for _, row in att_df.iterrows():
            roster_rows.append([
                str(row["student_id"]),
                str(row["full_name"])[:18],
                str(row["department"]),
                str(row["biometric_consent"]),
                str(row["status"]),
                f"{row['duration_minutes']}m",
                str(row["is_manual_override"]),
                str(row["override_reason"] or "-")[:20]
            ])

        roster_table = Table(roster_rows, colWidths=[65, 110, 45, 50, 60, 50, 55, 100])
        roster_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2D3748")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("ALIGN", (2, 1), (6, -1), "CENTER"),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ]
        # Alternating row colors
        for i in range(1, len(roster_rows)):
            bg = colors.HexColor("#FFFFFF") if i % 2 != 0 else colors.HexColor("#F7FAFC")
            roster_style.append(("BACKGROUND", (0, i), (-1, i), bg))
            if roster_rows[i][4] == "PRESENT":
                roster_style.append(("TEXTCOLOR", (4, i), (4, i), colors.HexColor("#22543D")))
            else:
                roster_style.append(("TEXTCOLOR", (4, i), (4, i), colors.HexColor("#742A2A")))

        roster_table.setStyle(TableStyle(roster_style))
        elements.append(roster_table)

    elements.append(Spacer(1, 12))

    # Pedagogical Suggestions Log
    elements.append(Paragraph("AI-Assisted Pedagogical Feedback & Interventions", h2_style))
    if not suggestions:
        elements.append(Paragraph("No automated intervention alerts triggered during this session.", body_style))
    else:
        sug_rows = [["Time", "Observed Indicator", "Pedagogical Recommendation"]]
        for s in suggestions[:8]:  # Display up to latest 8
            sug_rows.append([
                str(s["timestamp"])[11:19],
                str(s["observed_indicator"])[:40],
                str(s["suggested_action"])[:70]
            ])
        sug_table = Table(sug_rows, colWidths=[60, 185, 290])
        sug_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A5568")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ]))
        elements.append(sug_table)

    elements.append(Spacer(1, 14))

    # Privacy & Ethical Compliance Box
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E0"), spaceAfter=6))
    elements.append(Paragraph("<b>Privacy Protection & Pedagogical Scope Disclaimer:</b>", body_style))
    elements.append(Paragraph(ETHICAL_DISCLAIMER, disclaimer_style))
    elements.append(Paragraph(
        "Biometric Policy: Raw facial imagery is never retained. Face embeddings are mathematically non-reversible. "
        "Non-biometric manual alternatives are guaranteed for all learners without academic penalty.",
        disclaimer_style
    ))

    # Build PDF document
    doc.build(elements)
    return output_path
