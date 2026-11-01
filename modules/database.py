"""
Database Module for Multimodal Attendance & Engagement Agent.
Manages SQLite database schema, connections, transactions, and CRUD operations
with strict foreign keys, parameterized queries, and audit logging.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATABASE_PATH


def get_db_connection() -> sqlite3.Connection:
    """Creates a database connection with foreign key support enabled."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Enables column access by name
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_database() -> None:
    """Initializes the database schema with all required tables."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Students Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                email TEXT,
                department TEXT DEFAULT 'MCA',
                biometric_consent INTEGER NOT NULL DEFAULT 0,
                face_embedding BLOB,
                registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active INTEGER DEFAULT 1
            );
        """)

        # 2. Sessions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                course_name TEXT NOT NULL,
                instructor_name TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                status TEXT DEFAULT 'ACTIVE',
                modality_flags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 3. Attendance Records Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                student_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PRESENT',
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                duration_seconds REAL DEFAULT 0.0,
                sustained_detection_count INTEGER DEFAULT 0,
                is_manual_override INTEGER DEFAULT 0,
                override_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE,
                UNIQUE(session_id, student_id)
            );
        """)

        # 4. Session Indicators (Observable Behaviors) Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                face_visibility_rate REAL DEFAULT 0.0,
                head_pose_deviation_rate REAL DEFAULT 0.0,
                motion_index REAL DEFAULT 0.0,
                audio_energy_level REAL DEFAULT 0.0,
                is_active_speech INTEGER DEFAULT 0,
                fused_engagement_score REAL DEFAULT 0.0,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
        """)

        # 5. Teaching Suggestions Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teaching_suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                observed_indicator TEXT NOT NULL,
                pedagogical_reason TEXT NOT NULL,
                suggested_action TEXT NOT NULL,
                confidence_note TEXT NOT NULL,
                status TEXT DEFAULT 'ACTIVE',
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
        """)

        # 6. Audit Trail Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                entity_id TEXT,
                performed_by TEXT DEFAULT 'ADMIN',
                details TEXT
            );
        """)

        conn.commit()


# ==========================================
# Audit Logging Helper
# ==========================================
def log_audit_event(event_type: str, entity_id: str, details: str, performed_by: str = "ADMIN") -> None:
    """Appends an event to the immutable audit trail."""
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO audit_logs (event_type, entity_id, details, performed_by) VALUES (?, ?, ?, ?)",
            (event_type, entity_id, details, performed_by)
        )
        conn.commit()


# ==========================================
# Student CRUD Operations
# ==========================================
def register_student(
    student_id: str,
    full_name: str,
    email: Optional[str] = None,
    department: str = "MCA",
    biometric_consent: bool = False,
    face_embedding: Optional[np.ndarray] = None
) -> bool:
    """
    Registers a new student. If biometric consent is given, stores the embedding as raw bytes.
    Raw facial images are NEVER stored in the database.
    """
    embedding_blob = None
    if biometric_consent and face_embedding is not None:
        # Store as 32-bit float byte string
        embedding_blob = face_embedding.astype(np.float32).tobytes()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO students (student_id, full_name, email, department, biometric_consent, face_embedding, is_active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(student_id) DO UPDATE SET
                full_name=excluded.full_name,
                email=excluded.email,
                department=excluded.department,
                biometric_consent=excluded.biometric_consent,
                face_embedding=excluded.face_embedding,
                is_active=1;
            """,
            (student_id.strip(), full_name.strip(), email, department, int(biometric_consent), embedding_blob)
        )
        conn.commit()

    log_audit_event(
        "STUDENT_ENROLLED",
        student_id,
        f"Enrolled {full_name}. Biometric consent: {biometric_consent}"
    )
    return True


def get_all_students(active_only: bool = True) -> List[Dict[str, Any]]:
    """Retrieves list of registered students."""
    with get_db_connection() as conn:
        query = "SELECT student_id, full_name, email, department, biometric_consent, registration_date, is_active FROM students"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY student_id ASC"
        rows = conn.execute(query).fetchall()
        return [dict(row) for row in rows]


def get_enrolled_embeddings() -> Dict[str, np.ndarray]:
    """
    Returns a dictionary of {student_id: np.ndarray} for all active students
    who gave biometric consent and have an embedding saved.
    """
    embeddings = {}
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT student_id, face_embedding FROM students WHERE is_active = 1 AND biometric_consent = 1 AND face_embedding IS NOT NULL"
        )
        for row in cursor.fetchall():
            student_id = row["student_id"]
            blob = row["face_embedding"]
            if blob:
                emb = np.frombuffer(blob, dtype=np.float32)
                embeddings[student_id] = emb
    return embeddings


def delete_student(student_id: str, hard_delete: bool = False) -> bool:
    """Deletes or deactivates a student record and records an audit log."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if hard_delete:
            cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
        else:
            cursor.execute("UPDATE students SET is_active = 0, face_embedding = NULL WHERE student_id = ?", (student_id,))
        conn.commit()

    action = "HARD_DELETE" if hard_delete else "DEACTIVATE_AND_PURGE_BIOMETRICS"
    log_audit_event(action, student_id, f"Student record {student_id} modified.")
    return True


# ==========================================
# Session Operations
# ==========================================
def create_session(session_id: str, course_name: str, instructor_name: str, modality_flags: Optional[Dict[str, bool]] = None) -> bool:
    """Initializes a new classroom session record."""
    flags_json = json.dumps(modality_flags or {"vision": True, "audio": True})
    now = datetime.now().isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, course_name, instructor_name, start_time, status, modality_flags)
            VALUES (?, ?, ?, ?, 'ACTIVE', ?)
            """,
            (session_id, course_name, instructor_name, now, flags_json)
        )
        conn.commit()
    log_audit_event("SESSION_START", session_id, f"Course: {course_name}, Instructor: {instructor_name}")
    return True


def end_session(session_id: str) -> bool:
    """Closes an active classroom session."""
    now = datetime.now().isoformat()
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE sessions SET end_time = ?, status = 'COMPLETED' WHERE session_id = ?",
            (now, session_id)
        )
        conn.commit()
    log_audit_event("SESSION_END", session_id, "Session marked as COMPLETED")
    return True


def get_all_sessions() -> List[Dict[str, Any]]:
    """Fetches all session records ordered by start time descending."""
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM sessions ORDER BY start_time DESC").fetchall()
        return [dict(row) for row in rows]


# ==========================================
# Attendance Operations
# ==========================================
def record_or_update_attendance(
    session_id: str,
    student_id: str,
    detection_time: datetime,
    sustained_count: int = 1,
    status: str = "PRESENT"
) -> None:
    """
    Inserts a new attendance record or updates first_seen/last_seen timestamps
    and calculates duration observed in seconds.
    """
    iso_time = detection_time.isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, first_seen, last_seen, duration_seconds, sustained_detection_count FROM attendance WHERE session_id = ? AND student_id = ?",
            (session_id, student_id)
        )
        row = cursor.fetchone()

        if row is None:
            # First observation
            cursor.execute(
                """
                INSERT INTO attendance (session_id, student_id, status, first_seen, last_seen, duration_seconds, sustained_detection_count)
                VALUES (?, ?, ?, ?, ?, 1.0, ?)
                """,
                (session_id, student_id, status, iso_time, iso_time, sustained_count)
            )
        else:
            # Update last_seen and compute duration
            first_seen_dt = datetime.fromisoformat(row["first_seen"])
            duration = max(0.0, (detection_time - first_seen_dt).total_seconds())
            new_sustained = max(row["sustained_detection_count"], sustained_count)

            cursor.execute(
                """
                UPDATE attendance
                SET last_seen = ?, duration_seconds = ?, sustained_detection_count = ?, status = ?
                WHERE session_id = ? AND student_id = ?
                """,
                (iso_time, duration, new_sustained, status, session_id, student_id)
            )
        conn.commit()


def manual_attendance_override(
    session_id: str,
    student_id: str,
    new_status: str,
    reason: str,
    admin_name: str = "ADMIN"
) -> bool:
    """Allows an authorized instructor to manually correct attendance with reason logged."""
    now = datetime.now().isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO attendance (session_id, student_id, status, first_seen, last_seen, is_manual_override, override_reason)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(session_id, student_id) DO UPDATE SET
                status = excluded.status,
                is_manual_override = 1,
                override_reason = excluded.override_reason;
            """,
            (session_id, student_id, new_status, now, now, reason)
        )
        conn.commit()

    log_audit_event(
        "ATTENDANCE_OVERRIDE",
        f"{session_id}:{student_id}",
        f"Overridden to {new_status}. Reason: {reason}",
        performed_by=admin_name
    )
    return True


def get_session_attendance(session_id: str) -> List[Dict[str, Any]]:
    """Returns combined student details and attendance records for a session."""
    with get_db_connection() as conn:
        query = """
            SELECT
                s.student_id,
                s.full_name,
                s.department,
                s.biometric_consent,
                COALESCE(a.status, 'ABSENT') as status,
                a.first_seen,
                a.last_seen,
                COALESCE(a.duration_seconds, 0.0) as duration_seconds,
                COALESCE(a.is_manual_override, 0) as is_manual_override,
                a.override_reason
            FROM students s
            LEFT JOIN attendance a
                ON s.student_id = a.student_id AND a.session_id = ?
            WHERE s.is_active = 1
            ORDER BY s.student_id ASC;
        """
        rows = conn.execute(query, (session_id,)).fetchall()
        return [dict(row) for row in rows]


# ==========================================
# Indicators & Teaching Suggestions Logging
# ==========================================
def log_session_indicator(
    session_id: str,
    face_visibility: float,
    head_pose_deviation: float,
    motion_index: float,
    audio_energy: float,
    is_active_speech: bool,
    fused_score: float
) -> None:
    """Logs a periodic multimodal snapshot for trend visualization."""
    now = datetime.now().isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO session_indicators (
                session_id, timestamp, face_visibility_rate, head_pose_deviation_rate,
                motion_index, audio_energy_level, is_active_speech, fused_engagement_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id, now, float(face_visibility), float(head_pose_deviation),
                float(motion_index), float(audio_energy), int(is_active_speech), float(fused_score)
            )
        )
        conn.commit()


def log_teaching_suggestion(
    session_id: str,
    observed_indicator: str,
    pedagogical_reason: str,
    suggested_action: str,
    confidence_note: str
) -> None:
    """Inserts an AI-assisted teaching recommendation."""
    now = datetime.now().isoformat()
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO teaching_suggestions (
                session_id, timestamp, observed_indicator, pedagogical_reason,
                suggested_action, confidence_note, status
            ) VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE')
            """,
            (session_id, now, observed_indicator, pedagogical_reason, suggested_action, confidence_note)
        )
        conn.commit()


def get_session_suggestions(session_id: str) -> List[Dict[str, Any]]:
    """Retrieves all teaching suggestions generated for a session."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM teaching_suggestions WHERE session_id = ? ORDER BY timestamp DESC",
            (session_id,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_session_indicators(session_id: str) -> List[Dict[str, Any]]:
    """Retrieves timeline indicators for session visualization."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM session_indicators WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,)
        ).fetchall()
        return [dict(row) for row in rows]
