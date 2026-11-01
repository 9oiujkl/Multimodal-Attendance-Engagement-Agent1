"""
Audio Signal Processing & Acoustic Interaction Analysis Module.
Analyzes ambient classroom audio: RMS energy levels, zero-crossing rates,
speech presence, and prolonged silence detection.
Uses SciPy and Python's standard wave library for zero-dependency offline reliability.
"""

import io
import wave
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import scipy.io.wavfile as wavfile

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_FRAME_LENGTH,
    AUDIO_HOP_LENGTH,
    RMS_ENERGY_THRESHOLD,
    ZERO_CROSSING_RATE_THRESHOLD,
    PROLONGED_SILENCE_SECONDS
)


def load_audio_from_bytes_or_file(source: Any) -> Tuple[int, np.ndarray]:
    """
    Loads audio from either a file path, BytesIO, or raw bytes.
    Returns (sample_rate, normalized_mono_float32_array).
    """
    if isinstance(source, (str, Path)):
        sr, data = wavfile.read(str(source))
    elif isinstance(source, (bytes, bytearray)):
        bio = io.BytesIO(source)
        sr, data = wavfile.read(bio)
    elif hasattr(source, "read"):
        source.seek(0)
        sr, data = wavfile.read(source)
    else:
        raise ValueError(f"Unsupported audio source type: {type(source)}")

    # Convert multi-channel to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Normalize integer PCM data to float32 range [-1.0, 1.0]
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        audio = (data / max_val).astype(np.float32)
    else:
        audio = data.astype(np.float32)

    return sr, audio


def compute_rms(signal: np.ndarray) -> float:
    """Computes Root-Mean-Square (RMS) energy of a 1D audio signal."""
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2)))


def compute_zcr(signal: np.ndarray) -> float:
    """Computes Zero Crossing Rate (ZCR) of a 1D audio signal."""
    if len(signal) < 2:
        return 0.0
    zero_crosses = np.sum(np.abs(np.diff(np.sign(signal) >= 0)))
    return float(zero_crosses / (len(signal) - 1))


class AcousticAnalyzer:
    """
    Analyzes streaming audio buffers or prerecorded classroom sessions:
    - RMS energy tracking
    - Speech activity vs silence segmentation
    - Prolonged silence interval detection
    """

    def __init__(
        self,
        rms_threshold: float = RMS_ENERGY_THRESHOLD,
        zcr_threshold: float = ZERO_CROSSING_RATE_THRESHOLD,
        prolonged_silence_sec: float = PROLONGED_SILENCE_SECONDS
    ):
        self.rms_threshold = rms_threshold
        self.zcr_threshold = zcr_threshold
        self.prolonged_silence_sec = prolonged_silence_sec

        self.consecutive_silence_seconds: float = 0.0
        self.total_processed_seconds: float = 0.0
        self.total_speech_seconds: float = 0.0
        self.silence_alert_triggered: bool = False

    def process_chunk(self, audio_chunk: np.ndarray, sample_rate: int = AUDIO_SAMPLE_RATE) -> Dict[str, Any]:
        """
        Processes a single contiguous chunk/frame of audio.
        """
        if len(audio_chunk) == 0:
            return {
                "rms": 0.0,
                "zcr": 0.0,
                "is_active_speech": False,
                "energy_level": 0.0,
                "consecutive_silence_sec": round(self.consecutive_silence_seconds, 1),
                "prolonged_silence_alert": False
            }

        chunk_duration = float(len(audio_chunk) / sample_rate)
        rms = compute_rms(audio_chunk)
        zcr = compute_zcr(audio_chunk)

        is_active_speech = (rms >= self.rms_threshold) and (zcr >= self.zcr_threshold)

        self.total_processed_seconds += chunk_duration

        if is_active_speech:
            self.total_speech_seconds += chunk_duration
            self.consecutive_silence_seconds = 0.0
            self.silence_alert_triggered = False
            alert = False
        else:
            self.consecutive_silence_seconds += chunk_duration
            if self.consecutive_silence_seconds >= self.prolonged_silence_sec:
                alert = True
                self.silence_alert_triggered = True
            else:
                alert = False

        # Energy level normalized to 0.0 - 1.0 (typical active classroom voice has rms ~ 0.08)
        energy_level = float(np.clip(rms / 0.10, 0.0, 1.0))

        return {
            "rms": round(rms, 4),
            "zcr": round(zcr, 4),
            "is_active_speech": is_active_speech,
            "energy_level": round(energy_level, 3),
            "consecutive_silence_sec": round(self.consecutive_silence_seconds, 1),
            "prolonged_silence_alert": alert,
            "speech_ratio": round(self.total_speech_seconds / max(0.1, self.total_processed_seconds), 3)
        }

    def analyze_full_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        chunk_size_sec: float = 0.5
    ) -> Dict[str, Any]:
        """
        Analyzes an entire audio track in successive windows.
        Returns timeline metrics and aggregate session summary.
        """
        chunk_samples = int(sample_rate * chunk_size_sec)
        num_chunks = int(np.ceil(len(audio) / chunk_samples))

        timeline = []
        silence_spans = []
        active_silence_start = None

        for i in range(num_chunks):
            start = i * chunk_samples
            end = min(len(audio), start + chunk_samples)
            chunk = audio[start:end]
            if len(chunk) == 0:
                continue

            chunk_time = i * chunk_size_sec
            metrics = self.process_chunk(chunk, sample_rate)
            metrics["time_sec"] = round(chunk_time, 2)
            timeline.append(metrics)

            if metrics["is_active_speech"]:
                if active_silence_start is not None:
                    duration = chunk_time - active_silence_start
                    if duration >= self.prolonged_silence_sec:
                        silence_spans.append({
                            "start_sec": active_silence_start,
                            "end_sec": chunk_time,
                            "duration_sec": round(duration, 1)
                        })
                    active_silence_start = None
            else:
                if active_silence_start is None:
                    active_silence_start = chunk_time

        # Check trailing silence
        total_dur = float(len(audio) / sample_rate)
        if active_silence_start is not None:
            trailing_dur = total_dur - active_silence_start
            if trailing_dur >= self.prolonged_silence_sec:
                silence_spans.append({
                    "start_sec": active_silence_start,
                    "end_sec": total_dur,
                    "duration_sec": round(trailing_dur, 1)
                })

        overall_speech_ratio = float(self.total_speech_seconds / max(0.1, total_dur))
        avg_rms = float(np.mean([t["rms"] for t in timeline])) if timeline else 0.0

        return {
            "total_duration_sec": round(total_dur, 2),
            "total_speech_sec": round(self.total_speech_seconds, 2),
            "speech_ratio": round(overall_speech_ratio, 3),
            "average_rms": round(avg_rms, 4),
            "prolonged_silence_intervals": silence_spans,
            "timeline": timeline
        }

    def reset(self):
        """Resets the state of the analyzer."""
        self.consecutive_silence_seconds = 0.0
        self.total_processed_seconds = 0.0
        self.total_speech_seconds = 0.0
        self.silence_alert_triggered = False


def generate_synthetic_audio(
    duration_sec: float = 10.0,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    scenario: str = "active_lecture"
) -> Tuple[int, np.ndarray]:
    """
    Generates synthetic WAV audio waveforms for automated testing and demonstrations:
    - 'active_lecture': Continuous speech simulation (fundamental harmonic + formants + ambient noise)
    - 'prolonged_silence': Ambient room noise (RMS < 0.005) for testing silence detector
    - 'interactive_qa': Alternating 4s speech and 4s silence
    """
    total_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    noise = np.random.normal(0, 0.003, total_samples)  # Background room noise

    if scenario == "prolonged_silence":
        signal = noise.astype(np.float32)

    elif scenario == "interactive_qa":
        # Square envelope with period 8 seconds (4s speech, 4s silence)
        envelope = 0.5 * (1 + np.sign(np.sin(2 * np.pi * t / 8.0)))
        # Harmonic human voice simulation (180 Hz base pitch + 360 Hz harmonic + 720 Hz)
        voice = (
            0.05 * np.sin(2 * np.pi * 180 * t) +
            0.03 * np.sin(2 * np.pi * 360 * t) +
            0.02 * np.sin(2 * np.pi * 720 * t)
        )
        signal = (voice * envelope + noise).astype(np.float32)

    else:  # 'active_lecture' default
        voice = (
            0.06 * np.sin(2 * np.pi * 150 * t) +
            0.04 * np.sin(2 * np.pi * 300 * t) +
            0.02 * np.sin(2 * np.pi * 600 * t)
        )
        # Random natural pauses
        modulation = 0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
        signal = (voice * modulation + noise).astype(np.float32)

    signal = np.clip(signal, -1.0, 1.0)
    return sample_rate, signal
