"""
Central Configuration Module for Multimodal Attendance & Engagement Agent.
Manages file system paths, machine learning thresholds, audio DSP parameters,
and privacy enforcement rules.
"""

from pathlib import Path

# ==========================================
# 1. Base Paths & Directory Setup
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
REPORTS_DIR = BASE_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"

# Ensure runtime directories exist
for folder in [MODELS_DIR, DATABASE_DIR, REPORTS_DIR, ASSETS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Database path
DATABASE_PATH = DATABASE_DIR / "attendance_agent.db"

# Model weights paths
YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# ==========================================
# 2. Computer Vision Parameters
# ==========================================
# YuNet Face Detector
FACE_DET_SCORE_THRESHOLD = 0.60
FACE_DET_NMS_THRESHOLD = 0.30
FACE_DET_TOP_K = 50

# SFace Face Recognizer
# SFace produces 128-d L2 normalized embeddings.
# Cosine similarity range: -1.0 to 1.0 (Threshold >= 0.363 indicates positive match)
COSINE_SIMILARITY_THRESHOLD = 0.363

# Temporal stability requirement:
# A student must be consistently detected across N consecutive frames before attendance is validated.
SUSTAINED_DETECTION_FRAMES = 5

# Video Sampling & Sizing
INPUT_FRAME_WIDTH = 640
INPUT_FRAME_HEIGHT = 480
FRAME_PROCESS_INTERVAL = 3  # Process 1 frame every 3 frames for CPU efficiency

# Head Pose Angle Limits (degrees)
YAW_FORWARD_TOLERANCE = 25.0    # Left/Right deviation limit
PITCH_FORWARD_TOLERANCE = 20.0  # Up/Down deviation limit

# ==========================================
# 3. Audio Signal Processing Parameters
# ==========================================
AUDIO_SAMPLE_RATE = 22050
AUDIO_FRAME_LENGTH = 2048
AUDIO_HOP_LENGTH = 512

# Silence & Activity Thresholds (RMS Energy)
RMS_ENERGY_THRESHOLD = 0.015
ZERO_CROSSING_RATE_THRESHOLD = 0.08
PROLONGED_SILENCE_SECONDS = 15.0

# ==========================================
# 4. Multimodal Fusion Weights (Total = 1.0)
# ==========================================
WEIGHT_FACE_VISIBILITY = 0.40
WEIGHT_HEAD_ORIENTATION = 0.35
WEIGHT_AUDIO_INTERACTION = 0.25

# ==========================================
# 5. Privacy, Ethics & Audit Guidelines
# ==========================================
STORE_RAW_FACIAL_IMAGES = False
ALLOW_AUDIO_PROCESSING = True
ALLOW_NON_BIOMETRIC_ENROLLMENT = True

ETHICAL_DISCLAIMER = (
    "Observable behavioral indicators (head orientation, motion frequency, and ambient sound activity) "
    "are recorded for aggregate classroom feedback only. They do NOT measure psychological engagement, "
    "comprehension, intelligence, or student aptitude."
)
