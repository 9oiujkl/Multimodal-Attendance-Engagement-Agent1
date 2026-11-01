"""
Teaching Suggestions & Pedagogical Insights Engine.
Translates observable behavioral indicators into actionable, non-punitive,
pedagogical recommendations to support instructors in real time.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.database import log_teaching_suggestion, get_session_suggestions


class PedagogicalAdvisoryEngine:
    """
    Evaluates multimodal telemetry against pedagogical rules.
    Maintains a cooldown buffer to prevent overwhelming instructors with repetitive prompts.
    """

    def __init__(self, cooldown_seconds: int = 45):
        self.cooldown_seconds = cooldown_seconds
        # Stores {rule_key: last_triggered_datetime}
        self.last_triggered: Dict[str, datetime] = {}

    def _can_trigger(self, rule_key: str, now: datetime) -> bool:
        if rule_key not in self.last_triggered:
            return True
        return (now - self.last_triggered[rule_key]).total_seconds() >= self.cooldown_seconds

    def evaluate_indicators(
        self,
        session_id: str,
        visual_metrics: Dict[str, Any],
        audio_metrics: Optional[Dict[str, Any]] = None,
        fused_score: float = 0.50
    ) -> List[Dict[str, Any]]:
        """
        Evaluates current classroom telemetry and returns newly generated suggestions.
        Automatically logs suggestions to the SQLite database.
        """
        now = datetime.now()
        new_suggestions = []

        face_vis = visual_metrics.get("face_visibility_rate", 0.0)
        head_dev = visual_metrics.get("head_pose_deviation_rate", 0.0)
        motion = visual_metrics.get("motion_index", 0.0)

        audio_alert = False
        speech_ratio = 0.0
        if audio_metrics:
            audio_alert = audio_metrics.get("prolonged_silence_alert", False)
            speech_ratio = audio_metrics.get("speech_ratio", 0.0)

        # Rule 1: Prolonged Silence Detected
        if audio_alert and self._can_trigger("PROLONGED_SILENCE", now):
            suggestion = {
                "session_id": session_id,
                "observed_indicator": "Acoustic lull / prolonged silence exceeding 15 seconds",
                "pedagogical_reason": "Learners may hesitate to interrupt, need time to digest complex material, or feel hesitant to ask questions.",
                "suggested_action": "Pause lecture delivery and ask a targeted open-ended question (e.g., 'What would happen if we change this parameter?').",
                "confidence_note": "High confidence (Continuous RMS energy below silence threshold for >15s)."
            }
            self.last_triggered["PROLONGED_SILENCE"] = now
            new_suggestions.append(suggestion)

        # Rule 2: High Head Pose Deviation (Looking Away)
        if head_dev >= 0.50 and self._can_trigger("HEAD_DEVIATION", now):
            suggestion = {
                "session_id": session_id,
                "observed_indicator": f"{int(head_dev * 100)}% of detected students exhibit head pose deviation away from the board/screen",
                "pedagogical_reason": "Attention is shifting away from the primary instructional display, suggesting cognitive fatigue or visual monotony.",
                "suggested_action": "Switch teaching modality: demonstrate a live practical code snippet, draw on a digital whiteboard, or run a quick concept poll.",
                "confidence_note": "High confidence (Facial landmark horizontal/vertical geometric ratio)."
            }
            self.last_triggered["HEAD_DEVIATION"] = now
            new_suggestions.append(suggestion)

        # Rule 3: Low Face Visibility
        if face_vis < 0.35 and visual_metrics.get("num_faces", 0) > 0 and self._can_trigger("LOW_VISIBILITY", now):
            suggestion = {
                "session_id": session_id,
                "observed_indicator": f"Face visibility is at {int(face_vis * 100)}% of expected classroom attendance",
                "pedagogical_reason": "Several students may have stepped out, or camera field-of-view angle is partially occluded.",
                "suggested_action": "Check camera framing and conduct an interactive attendance roll-call check.",
                "confidence_note": "Medium confidence (Multi-face detector count vs roster size)."
            }
            self.last_triggered["LOW_VISIBILITY"] = now
            new_suggestions.append(suggestion)

        # Rule 4: Restlessness / High Physical Motion
        if motion >= 0.45 and self._can_trigger("HIGH_MOTION", now):
            suggestion = {
                "session_id": session_id,
                "observed_indicator": f"Elevated classroom motion index ({motion:.2f}) observed across consecutive frames",
                "pedagogical_reason": "High physical movement often indicates restlessness from prolonged passive sitting.",
                "suggested_action": "Introduce a 2-minute active breakout exercise: 'Turn to your neighbor and explain the last concept.'",
                "confidence_note": "Medium confidence (Inter-frame pixel differencing)."
            }
            self.last_triggered["HIGH_MOTION"] = now
            new_suggestions.append(suggestion)

        # Rule 5: Sustained High Engagement Flow
        if fused_score >= 0.82 and self._can_trigger("HIGH_ENGAGEMENT", now):
            suggestion = {
                "session_id": session_id,
                "observed_indicator": f"Strong multimodal engagement flow ({int(fused_score * 100)}%) maintained",
                "pedagogical_reason": "Instructional pacing, visual cues, and explanation clarity are effectively aligned with student attention.",
                "suggested_action": "Pacing is optimal! Maintain current teaching tempo and segue into the next learning objective.",
                "confidence_note": "High confidence (Fused landmark alignment and attendance stability)."
            }
            self.last_triggered["HIGH_ENGAGEMENT"] = now
            new_suggestions.append(suggestion)

        # Persist suggestions to database
        for s in new_suggestions:
            log_teaching_suggestion(
                session_id=s["session_id"],
                observed_indicator=s["observed_indicator"],
                pedagogical_reason=s["pedagogical_reason"],
                suggested_action=s["suggested_action"],
                confidence_note=s["confidence_note"]
            )

        return new_suggestions

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Fetches all logged suggestions for a session from the database."""
        return get_session_suggestions(session_id)
