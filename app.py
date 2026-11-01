"""
Multimodal Attendance & Engagement Agent
Main Streamlit Application Dashboard
"""

from datetime import datetime
from pathlib import Path
import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    DATABASE_PATH,
    YUNET_MODEL_PATH,
    SFACE_MODEL_PATH,
    SUSTAINED_DETECTION_FRAMES,
    ETHICAL_DISCLAIMER,
    REPORTS_DIR
)
from modules.database import (
    init_database,
    get_all_students,
    get_enrolled_embeddings,
    create_session,
    end_session,
    get_all_sessions,
    get_session_attendance,
    manual_attendance_override,
    get_session_indicators,
    get_session_suggestions,
    get_db_connection
)
from modules.student_registration import (
    enroll_student_biometric,
    enroll_student_non_biometric,
    unenroll_student
)
from modules.face_recognition_attendance import FaceAttendanceSystem
from modules.engagement_analysis import VisualEngagementAnalyzer, MultimodalFusionEngine
from modules.audio_analysis import (
    AcousticAnalyzer,
    generate_synthetic_audio,
    load_audio_from_bytes_or_file
)
from modules.teaching_suggestions import PedagogicalAdvisoryEngine
from modules.reporting import (
    get_session_summary_dataframe,
    export_attendance_csv,
    generate_session_pdf_report
)

# Page configuration
st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database schema
init_database()

# Session state initialization
if "face_system" not in st.session_state:
    st.session_state.face_system = FaceAttendanceSystem(sustained_frames_required=SUSTAINED_DETECTION_FRAMES)
if "visual_analyzer" not in st.session_state:
    st.session_state.visual_analyzer = VisualEngagementAnalyzer()
if "fusion_engine" not in st.session_state:
    st.session_state.fusion_engine = MultimodalFusionEngine()
if "audio_analyzer" not in st.session_state:
    st.session_state.audio_analyzer = AcousticAnalyzer()
if "advisory_engine" not in st.session_state:
    st.session_state.advisory_engine = PedagogicalAdvisoryEngine(cooldown_seconds=30)
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .status-badge-green {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .status-badge-orange {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .status-badge-red {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Sidebar: System Status & Session Switcher
# ==========================================
with st.sidebar:
    st.title("🎓 Agent Controls")

    # System Status Indicators
    with st.expander("⚙️ Hardware & Model Status", expanded=False):
        db_status = "Online" if DATABASE_PATH.exists() else "Missing"
        yunet_status = "Loaded (ONNX)" if YUNET_MODEL_PATH.exists() else "Haar Fallback"
        sface_status = "Loaded (ONNX)" if SFACE_MODEL_PATH.exists() else "Fallback"

        st.markdown(f"**Database:** `{db_status}`")
        st.markdown(f"**Face Detector:** `{yunet_status}`")
        st.markdown(f"**Face Recognizer:** `{sface_status}`")
        st.markdown(f"**Privacy Policy:** `Zero Raw Images`")
        st.markdown(f"**Biometric Template:** `128-d L2 Vector`")

    st.markdown("---")

    # Session Selection
    sessions = get_all_sessions()
    active_sessions = [s for s in sessions if s.get("status") == "ACTIVE"]

    if active_sessions:
        session_options = {s["session_id"]: f"{s['course_name']} ({s['session_id']})" for s in active_sessions}
        default_index = 0
        if st.session_state.current_session_id in session_options:
            default_index = list(session_options.keys()).index(st.session_state.current_session_id)

        selected_id = st.selectbox(
            "Select Active Classroom Session:",
            options=list(session_options.keys()),
            format_func=lambda x: session_options[x],
            index=default_index
        )
        st.session_state.current_session_id = selected_id
    else:
        st.info("No active classroom session found. Start one below or in Dashboard!")
        st.session_state.current_session_id = None

    # Quick session creator
    with st.expander("➕ Start New Session", expanded=False):
        with st.form("sidebar_new_session_form"):
            new_cid = st.text_input("Course Name", value="MCA-601 Computer Vision")
            new_instructor = st.text_input("Instructor Name", value="Dr. Ramesh Kumar")
            new_sid = f"SES-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            vision_enabled = st.checkbox("Enable Vision Modality", value=True)
            audio_enabled = st.checkbox("Enable Ambient Audio Modality", value=True)
            submitted = st.form_submit_button("Launch Session", use_container_width=True)

            if submitted:
                create_session(
                    session_id=new_sid,
                    course_name=new_cid,
                    instructor_name=new_instructor,
                    modality_flags={"vision": vision_enabled, "audio": audio_enabled}
                )
                st.session_state.current_session_id = new_sid
                st.session_state.face_system.reset_session_cache()
                st.success(f"Session {new_sid} started!")
                st.rerun()

    # End Active Session
    if st.session_state.current_session_id:
        if st.button("⏹️ End Selected Session", use_container_width=True, type="secondary"):
            end_session(st.session_state.current_session_id)
            st.session_state.face_system.reset_session_cache()
            st.success(f"Session {st.session_state.current_session_id} completed.")
            st.session_state.current_session_id = None
            st.rerun()

    st.markdown("---")
    st.caption("Multimodal Attendance & Engagement Agent v2.0")
    st.caption("Ethical, privacy-first computer vision & speech processing.")


# Header
st.markdown('<div class="main-title">Multimodal Classroom Attendance & Engagement Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Privacy-preserving real-time biometric verification, observable head-pose attention tracking, acoustic activity sensing, and pedagogical insights.</div>',
    unsafe_allow_html=True
)

# Navigation Tabs
tabs = st.tabs([
    "📊 Dashboard & Overview",
    "👤 Student Enrollment",
    "📹 Live Attendance & Engagement",
    "🎙️ Acoustic Analysis",
    "📝 Attendance Roster & Override",
    "💡 Teaching Suggestions",
    "📈 Reports & Analytics Export",
    "🔒 Privacy & Audit Trail"
])


# ==========================================
# TAB 1: Dashboard & Overview
# ==========================================
with tabs[0]:
    st.subheader("Session Dashboard & Operations Overview")

    all_students = get_all_students(active_only=True)
    curr_sid = st.session_state.current_session_id

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Enrolled Students", len(all_students))

    attendance_data = get_session_attendance(curr_sid) if curr_sid else []
    present_count = len([a for a in attendance_data if a["status"] == "PRESENT"])
    with col2:
        st.metric("Present Today", present_count if curr_sid else "No Session")

    with col3:
        att_rate = f"{(present_count / max(1, len(all_students)) * 100):.1f}%" if curr_sid and len(all_students) > 0 else "N/A"
        st.metric("Class Attendance Rate", att_rate)

    indicators_list = get_session_indicators(curr_sid) if curr_sid else []
    avg_eng = (
        f"{float(np.mean([i['fused_engagement_score'] for i in indicators_list]) * 100):.1f}%"
        if indicators_list else "Awaiting Data"
    )
    with col4:
        st.metric("Avg Fused Engagement", avg_eng)

    st.markdown("---")

    # Current Session Status Card
    if curr_sid:
        current_sess_info = next((s for s in sessions if s["session_id"] == curr_sid), None)
        if current_sess_info:
            st.info(
                f"🎯 **Active Session**: `{current_sess_info['course_name']}` | "
                f"**ID**: `{curr_sid}` | "
                f"**Instructor**: `{current_sess_info['instructor_name']}` | "
                f"**Started**: `{current_sess_info['start_time'][:19]}`"
            )
    else:
        st.warning("⚠️ No classroom session currently selected. Create or select a session from the sidebar to start live attendance.")

    # All Sessions History Table
    st.write("### 📜 Session History")
    if sessions:
        df_sess = pd.DataFrame(sessions)
        st.dataframe(
            df_sess[["session_id", "course_name", "instructor_name", "start_time", "end_time", "status"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("No session records found.")


# ==========================================
# TAB 2: Student Enrollment
# ==========================================
with tabs[1]:
    st.subheader("Student Registration & Biometric Consent Management")

    enroll_tab1, enroll_tab2 = st.tabs(["📸 Biometric Face Enrollment", "📝 Non-Biometric Alternative (Opt-Out)"])

    with enroll_tab1:
        st.markdown(
            "Enrolls student by extracting a **128-dimensional L2-normalized mathematical face embedding**. "
            "**In strict adherence to ethical standards, the raw facial image is NEVER stored in the database.**"
        )

        with st.form("biometric_enrollment_form"):
            c1, c2 = st.columns(2)
            with c1:
                b_student_id = st.text_input("Student ID / Roll No*", placeholder="e.g. MCA-2026-042")
                b_full_name = st.text_input("Full Name*", placeholder="e.g. Ananya Sharma")
            with c2:
                b_email = st.text_input("Student Email", placeholder="e.g. ananya@university.edu")
                b_department = st.selectbox("Department", ["MCA", "M.Tech CSE", "B.Tech IT", "Data Science"], index=0)

            st.write("##### Biometric Capture Option")
            capture_mode = st.radio("Choose Photo Source:", ["Webcam Snapshot", "File Upload"], horizontal=True)

            img_file = None
            if capture_mode == "Webcam Snapshot":
                img_file = st.camera_input("Take Enrollment Photo")
            else:
                img_file = st.file_uploader("Upload Clear Frontal Photo (JPG/PNG)", type=["jpg", "jpeg", "png"])

            # Mandatory Biometric Consent
            b_consent = st.checkbox(
                "I explicitly provide informed biometric consent for facial template feature extraction for automated classroom attendance.",
                value=False
            )

            enroll_btn = st.form_submit_button("Enroll Student Biometrically", type="primary", use_container_width=True)

            if enroll_btn:
                if not b_student_id or not b_full_name:
                    st.error("Student ID and Full Name are mandatory.")
                elif not b_consent:
                    st.error("Biometric consent is mandatory for face enrollment. If student opts out, use the Non-Biometric Alternative tab.")
                elif img_file is None:
                    st.error("Please take a photo or upload an image file.")
                else:
                    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
                    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    success, msg = enroll_student_biometric(
                        student_id=b_student_id,
                        full_name=b_full_name,
                        email=b_email,
                        department=b_department,
                        image_bgr=image_bgr
                    )
                    if success:
                        st.success(f"✅ {msg}")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

    with enroll_tab2:
        st.markdown(
            "#### Inclusive Non-Biometric Enrollment\n"
            "Students who choose not to enroll their facial templates are registered under this alternative. "
            "They participate via manual ID verification or instructor roll-call, **with zero academic disadvantage**."
        )
        with st.form("non_biometric_form"):
            nb_col1, nb_col2 = st.columns(2)
            with nb_col1:
                nb_student_id = st.text_input("Student ID / Roll No*", placeholder="e.g. MCA-2026-043")
                nb_full_name = st.text_input("Full Name*", placeholder="e.g. Vikramaditya Verma")
            with nb_col2:
                nb_email = st.text_input("Email", placeholder="e.g. vikram@university.edu")
                nb_department = st.selectbox("Department ", ["MCA", "M.Tech CSE", "B.Tech IT", "Data Science"], index=0)

            nb_submit = st.form_submit_button("Register Student (Non-Biometric)", use_container_width=True)
            if nb_submit:
                if not nb_student_id or not nb_full_name:
                    st.error("Student ID and Full Name are mandatory.")
                else:
                    ok, msg = enroll_student_non_biometric(
                        student_id=nb_student_id,
                        full_name=nb_full_name,
                        email=nb_email,
                        department=nb_department
                    )
                    if ok:
                        st.success(f"✅ {msg}")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

    st.markdown("---")
    st.write("### 📋 Registered Student Roster")
    roster = get_all_students(active_only=True)
    if roster:
        df_roster = pd.DataFrame(roster)
        df_roster["Biometric Status"] = df_roster["biometric_consent"].apply(lambda x: "✅ Biometric Template Active" if x else "🛡️ Non-Biometric Alternative")
        st.dataframe(
            df_roster[["student_id", "full_name", "department", "email", "Biometric Status", "registration_date"]],
            use_container_width=True,
            hide_index=True
        )

        with st.expander("🗑️ Deactivate Student & Purge Biometrics"):
            student_to_delete = st.selectbox("Select Student ID:", [s["student_id"] for s in roster])
            if st.button("Purge & Deactivate", type="secondary"):
                del_ok, del_msg = unenroll_student(student_to_delete)
                if del_ok:
                    st.success(del_msg)
                    st.rerun()
                else:
                    st.error(del_msg)
    else:
        st.info("No students enrolled yet.")


# ==========================================
# TAB 3: Live Attendance & Engagement
# ==========================================
with tabs[2]:
    st.subheader("Live Multi-Face Attendance & Observable Engagement Monitor")

    curr_sid = st.session_state.current_session_id
    if not curr_sid:
        st.warning("⚠️ Please select or start an active session from the sidebar first.")
    else:
        enrolled_embs = get_enrolled_embeddings()
        all_students = get_all_students(active_only=True)

        col_cam, col_metrics = st.columns([3, 2])

        with col_cam:
            st.write("##### Video Frame Source")
            feed_source = st.radio("Select Input:", ["Webcam Capture", "Upload Video Frame", "Simulate Live Classroom Test Frame"], horizontal=True)

            input_frame = None

            if feed_source == "Webcam Capture":
                cam_img = st.camera_input("Capture Classroom Frame")
                if cam_img is not None:
                    bytes_data = np.asarray(bytearray(cam_img.read()), dtype=np.uint8)
                    input_frame = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)

            elif feed_source == "Upload Video Frame":
                up_img = st.file_uploader("Upload Frame Image", type=["jpg", "jpeg", "png"], key="frame_uploader")
                if up_img is not None:
                    bytes_data = np.asarray(bytearray(up_img.read()), dtype=np.uint8)
                    input_frame = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)

            else:  # "Simulate Live Classroom Test Frame"
                st.caption("Generates a dynamic synthesized multi-student classroom frame for instant evaluation and demonstration.")
                sim_col1, sim_col2 = st.columns(2)
                with sim_col1:
                    sim_deviated = st.checkbox("Simulate Students Looking Away", value=False)
                with sim_col2:
                    sim_motion = st.slider("Simulate Motion Dynamic", 0.0, 1.0, 0.15, 0.05)

                if st.button("Generate & Process Simulated Frame", use_container_width=True, type="primary"):
                    # Create synthetic frame with drawn face structures
                    sim_frame = np.ones((480, 640, 3), dtype=np.uint8) * 235
                    # Add background classroom elements
                    cv2.rectangle(sim_frame, (40, 30), (600, 100), (210, 210, 210), -1)
                    cv2.putText(sim_frame, "CLASSROOM 301 - LIVE TELEMETRY", (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

                    # Student 1 box
                    cv2.rectangle(sim_frame, (100, 180), (240, 340), (220, 200, 190), -1)
                    cv2.circle(sim_frame, (170, 230), 45, (190, 170, 160), -1)

                    # Student 2 box
                    cv2.rectangle(sim_frame, (400, 180), (540, 340), (210, 205, 195), -1)
                    cv2.circle(sim_frame, (470, 230), 45, (185, 175, 165), -1)

                    input_frame = sim_frame

            # Process frame if available
            if input_frame is not None:
                annotated_frame, detections, summary = st.session_state.face_system.process_frame(
                    frame_bgr=input_frame,
                    session_id=curr_sid,
                    enrolled_embeddings=enrolled_embs,
                    persist_attendance=True
                )

                # Compute visual engagement metrics
                vis_metrics = st.session_state.visual_analyzer.analyze_frame_engagement(
                    frame_bgr=input_frame,
                    detected_faces=detections,
                    expected_student_count=max(1, len(all_students))
                )

                # Fuse multimodal score
                fused = st.session_state.fusion_engine.record_snapshot(
                    session_id=curr_sid,
                    visual_metrics=vis_metrics,
                    audio_metrics=None,
                    audio_enabled=False
                )

                # Check for pedagogical suggestions
                st.session_state.advisory_engine.evaluate_indicators(
                    session_id=curr_sid,
                    visual_metrics=vis_metrics,
                    audio_metrics=None,
                    fused_score=fused["fused_score"]
                )

                # Display annotated image
                rgb_annotated = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(rgb_annotated, caption="Annotated Live Video Frame", use_container_width=True)

                with col_metrics:
                    st.write("##### Frame Telemetry Metrics")
                    score_col1, score_col2 = st.columns(2)
                    with score_col1:
                        st.metric("Detected Faces", f"{len(detections)}")
                        st.metric("Face Visibility", f"{vis_metrics['face_visibility_rate'] * 100:.1f}%")
                        st.metric("Motion Index", f"{vis_metrics['motion_index']:.3f}")
                    with score_col2:
                        st.metric("Verified Present", f"{summary['confirmed_present_count']}")
                        st.metric("Forward Alignment", f"{vis_metrics['forward_alignment_rate'] * 100:.1f}%")
                        st.metric("Head Deviation", f"{vis_metrics['head_pose_deviation_rate'] * 100:.1f}%")

                    st.markdown("---")
                    st.write("##### Fused Engagement Score")
                    level = fused["level"]
                    badge_class = (
                        "status-badge-green" if level == "HIGH"
                        else "status-badge-orange" if level == "MODERATE"
                        else "status-badge-red"
                    )
                    st.markdown(
                        f"<div class='{badge_class}' style='text-align:center; padding:10px; font-size:1.3rem;'>"
                        f"Engagement Level: {level} ({fused['fused_score'] * 100:.1f}%)</div>",
                        unsafe_allow_html=True
                    )

            else:
                with col_metrics:
                    st.info("Awaiting camera frame or test simulation...")

        st.markdown("---")
        st.write("### 📈 Live Telemetry History (Current Session)")
        history_records = get_session_indicators(curr_sid)
        if history_records:
            df_hist = pd.DataFrame(history_records)
            df_hist["fused_pct"] = df_hist["fused_engagement_score"] * 100
            df_hist["visibility_pct"] = df_hist["face_visibility_rate"] * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df_hist["fused_pct"], mode="lines+markers", name="Fused Engagement %", line=dict(color="#10B981", width=3)))
            fig.add_trace(go.Scatter(y=df_hist["visibility_pct"], mode="lines", name="Face Visibility %", line=dict(color="#3B82F6", dash="dot")))
            fig.add_trace(go.Scatter(y=df_hist["motion_index"] * 100, mode="lines", name="Motion Index %", line=dict(color="#F59E0B", dash="dash")))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=20), xaxis_title="Telemetry Frame Index", yaxis_title="Score / Level (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No telemetry frames recorded for this session yet.")


# ==========================================
# TAB 4: Acoustic Analysis (Optional Modality)
# ==========================================
with tabs[3]:
    st.subheader("Acoustic Classroom Sensing & Silence Analysis")
    st.markdown(
        "Processes ambient classroom audio to detect speech dynamics, RMS energy levels, "
        "and prolonged silence intervals (>15 seconds), triggering pedagogical suggestions when appropriate."
    )

    audio_source = st.radio("Audio Source:", ["Synthesize Classroom Audio Scenario", "Upload Audio File (WAV)"], horizontal=True)

    audio_data = None
    sr = 22050

    if audio_source == "Synthesize Classroom Audio Scenario":
        scenario = st.selectbox(
            "Select Scenario to Demonstrate:",
            ["active_lecture", "prolonged_silence", "interactive_qa"],
            format_func=lambda x: {
                "active_lecture": "Active Lecture Speech (High vocal activity)",
                "prolonged_silence": "Prolonged Classroom Silence (>15s Quiet lull)",
                "interactive_qa": "Interactive Q&A Session (Speech alternating with pauses)"
            }[x]
        )
        duration_s = st.slider("Duration (Seconds):", 5.0, 30.0, 16.0, 1.0)
        if st.button("Generate & Analyze Audio Waveform", type="primary"):
            sr, audio_data = generate_synthetic_audio(duration_sec=duration_s, sample_rate=sr, scenario=scenario)
    else:
        uploaded_audio = st.file_uploader("Upload Classroom WAV File", type=["wav"])
        if uploaded_audio:
            sr, audio_data = load_audio_from_bytes_or_file(uploaded_audio)

    if audio_data is not None:
        st.session_state.audio_analyzer.reset()
        analysis_result = st.session_state.audio_analyzer.analyze_full_audio(audio_data, sample_rate=sr)

        a_col1, a_col2, a_col3, a_col4 = st.columns(4)
        with a_col1:
            st.metric("Total Duration", f"{analysis_result['total_duration_sec']}s")
        with a_col2:
            st.metric("Speech Time", f"{analysis_result['total_speech_sec']}s")
        with a_col3:
            st.metric("Speech Ratio", f"{analysis_result['speech_ratio'] * 100:.1f}%")
        with a_col4:
            st.metric("Average RMS", f"{analysis_result['average_rms']:.4f}")

        # Prolonged silence alerts
        silence_spans = analysis_result["prolonged_silence_intervals"]
        if silence_spans:
            st.warning(f"⚠️ **Prolonged Silence Detected!** Total prolonged quiet spans: {len(silence_spans)}.")
            for span in silence_spans:
                st.write(f"- Silence from **{span['start_sec']}s** to **{span['end_sec']}s** ({span['duration_sec']}s)")

            # If an active session exists, log suggestion
            if st.session_state.current_session_id:
                st.session_state.advisory_engine.evaluate_indicators(
                    session_id=st.session_state.current_session_id,
                    visual_metrics={"face_visibility_rate": 0.8, "head_pose_deviation_rate": 0.1, "motion_index": 0.1},
                    audio_metrics={"prolonged_silence_alert": True, "speech_ratio": analysis_result["speech_ratio"]},
                    fused_score=0.5
                )
                st.info("💡 Pedagogical alert logged to active session advisory log.")
        else:
            st.success("✅ Acoustic pacing healthy: No prolonged silence intervals detected.")

        # Waveform Visualization
        st.write("##### Acoustic Energy Timeline")
        timeline = analysis_result["timeline"]
        if timeline:
            df_audio = pd.DataFrame(timeline)
            fig_a = px.line(df_audio, x="time_sec", y="rms", title="RMS Energy Over Time", labels={"time_sec": "Time (s)", "rms": "RMS Energy"})
            fig_a.add_hline(y=0.015, line_dash="dash", line_color="red", annotation_text="Silence Threshold (0.015)")
            fig_a.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_a, use_container_width=True)


# ==========================================
# TAB 5: Attendance Roster & Manual Override
# ==========================================
with tabs[4]:
    st.subheader("Attendance Verification Roster & Instructor Override")

    curr_sid = st.session_state.current_session_id
    if not curr_sid:
        st.warning("⚠️ Please select an active or completed session to view attendance.")
    else:
        att_df = get_session_summary_dataframe(curr_sid)

        st.write(f"#### Roster for Session: `{curr_sid}`")
        if not att_df.empty:
            st.dataframe(
                att_df[["student_id", "full_name", "department", "biometric_consent", "status", "duration_minutes", "is_manual_override", "override_reason"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No attendance records logged yet for this session.")

        st.markdown("---")
        st.write("### ✍️ Manual Attendance Correction (Instructor Audit Override)")
        st.markdown(
            "Under educational regulations, instructors retain final authority to correct attendance records "
            "(e.g., verifying a student who opted out of biometrics or had a medical reason). Every change is permanently audited."
        )

        all_active_students = get_all_students(active_only=True)
        if all_active_students:
            with st.form("manual_override_form"):
                o_col1, o_col2, o_col3 = st.columns(3)
                with o_col1:
                    target_id = st.selectbox(
                        "Select Student:",
                        [f"{s['student_id']} - {s['full_name']}" for s in all_active_students]
                    ).split(" - ")[0]
                with o_col2:
                    new_status = st.selectbox("New Attendance Status:", ["PRESENT", "ABSENT", "EXCUSED", "LATE"])
                with o_col3:
                    override_reason = st.text_input("Mandatory Justification / Reason*", placeholder="e.g. In-person ID card verification")

                override_submit = st.form_submit_button("Submit Attendance Override", type="primary")

                if override_submit:
                    if not override_reason.strip():
                        st.error("A documented reason is mandatory for manual attendance overrides.")
                    else:
                        manual_attendance_override(
                            session_id=curr_sid,
                            student_id=target_id,
                            new_status=new_status,
                            reason=override_reason.strip(),
                            admin_name="INSTRUCTOR"
                        )
                        st.success(f"✅ Attendance updated for {target_id} to '{new_status}' with audit trail entry.")
                        st.rerun()


# ==========================================
# TAB 6: Teaching Suggestions
# ==========================================
with tabs[5]:
    st.subheader("💡 AI-Assisted Pedagogical Suggestions & Insights")
    st.markdown(
        "The system evaluates physical and acoustic indicators to offer **supportive, non-evaluative** teaching insights "
        "designed to optimize classroom delivery without grading or evaluating individual learners."
    )

    curr_sid = st.session_state.current_session_id
    if not curr_sid:
        st.warning("⚠️ Select a session to view teaching recommendations.")
    else:
        suggestions = get_session_suggestions(curr_sid)
        if suggestions:
            for s in suggestions:
                with st.container():
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="display:flex; justify-content:space-between;">
                            <b>📌 {s['observed_indicator']}</b>
                            <span style="color:#6B7280; font-size:0.85rem;">{s['timestamp'][:19]}</span>
                        </div>
                        <div style="margin-top:6px; color:#1F2937;">
                            <b>Pedagogical Rationale:</b> {s['pedagogical_reason']}
                        </div>
                        <div style="margin-top:6px; color:#1D4ED8;">
                            <b>Recommended Instructor Action:</b> {s['suggested_action']}
                        </div>
                        <div style="margin-top:4px; color:#6B7280; font-size:0.85rem;">
                            <i>{s['confidence_note']}</i>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No intervention triggers recorded for this session yet. Indicators indicate steady classroom flow.")


# ==========================================
# TAB 7: Reports & Analytics Export
# ==========================================
with tabs[6]:
    st.subheader("📈 Classroom Analytics, CSV & PDF Report Export")

    curr_sid = st.session_state.current_session_id
    if not curr_sid:
        st.warning("⚠️ Select a session to generate analytics and reports.")
    else:
        att_df = get_session_summary_dataframe(curr_sid)
        indicators = get_session_indicators(curr_sid)

        r_col1, r_col2 = st.columns(2)

        with r_col1:
            st.write("##### Attendance Breakdown")
            if not att_df.empty:
                status_counts = att_df["status"].value_counts().reset_index()
                status_counts.columns = ["Status", "Count"]
                fig_pie = px.pie(status_counts, names="Status", values="Count", color="Status",
                                 color_discrete_map={"PRESENT": "#10B981", "ABSENT": "#EF4444", "EXCUSED": "#F59E0B", "LATE": "#6366F1"})
                fig_pie.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.caption("No attendance data to plot.")

        with r_col2:
            st.write("##### Engagement Timeline")
            if indicators:
                df_ind = pd.DataFrame(indicators)
                fig_line = px.line(df_ind, y="fused_engagement_score", title="Fused Engagement Trajectory", labels={"index": "Sample", "fused_engagement_score": "Score (0-1)"})
                fig_line.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.caption("No engagement metrics logged yet.")

        st.markdown("---")
        st.write("### 📥 Export Documents")

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.write("##### Download Attendance CSV")
            if not att_df.empty:
                csv_buffer = io.StringIO()
                att_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📄 Download Attendance CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"attendance_{curr_sid}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.button("📄 Download Attendance CSV", disabled=True, use_container_width=True)

        with exp_col2:
            st.write("##### Generate Executive PDF Report")
            if st.button("📑 Build PDF Audit Report (ReportLab)", use_container_width=True, type="primary"):
                pdf_path = generate_session_pdf_report(curr_sid)
                if pdf_path.exists():
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    st.success(f"Report generated successfully: `{pdf_path.name}` ({len(pdf_bytes) // 1024} KB)")
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"report_{curr_sid}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )


# ==========================================
# TAB 8: Privacy & Audit Trail
# ==========================================
with tabs[7]:
    st.subheader("🔒 Privacy Policy, Ethical Compliance & Immutable Audit Trail")

    st.markdown(f"""
    > **Ethical & Pedagogical Statement:**  
    > {ETHICAL_DISCLAIMER}
    """)

    st.write("#### 🛡️ Privacy Guarantee Architecture")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""
        **1. No Raw Facial Photos Saved**
        - Images captured during enrollment or live monitoring are processed in RAM memory and immediately discarded.
        - SQLite stores only 128-dimensional floating-point embedding arrays.
        """)
    with p2:
        st.markdown("""
        **2. Mathematical Non-Reversibility**
        - Face embeddings are one-way projections.
        - Original facial imagery cannot be reconstructed from the 128-d cosine vector representations.
        """)
    with p3:
        st.markdown("""
        **3. Opt-Out Equality Guarantee**
        - Non-biometric registration alternatives are built directly into the interface.
        - One-click purging deactivates enrollment and deletes biometric templates permanently.
        """)

    st.markdown("---")
    st.write("#### 📜 System Audit Trail")
    with get_db_connection() as conn:
        audit_rows = conn.execute("SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 100").fetchall()
        if audit_rows:
            df_audit = pd.DataFrame([dict(r) for r in audit_rows])
            st.dataframe(
                df_audit[["id", "timestamp", "event_type", "entity_id", "performed_by", "details"]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Audit log is currently empty.")
