"""
Utility script to download official OpenCV Zoo ONNX models for offline use.
Face Detection: YuNet
Face Recognition: SFace
"""

import sys
import urllib.request
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import YUNET_MODEL_PATH, SFACE_MODEL_PATH, MODELS_DIR

MODELS = {
    "YuNet Face Detector": {
        "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "path": YUNET_MODEL_PATH,
        "size_kb": 230
    },
    "SFace Face Recognizer": {
        "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "path": SFACE_MODEL_PATH,
        "size_kb": 37800
    }
}

def download_models():
    """Downloads required ONNX model weights to local models directory."""
    print("=" * 65)
    print("Multimodal Attendance Agent: Local Model Downloader")
    print("=" * 65)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    success = True

    for name, info in MODELS.items():
        dest = Path(info["path"])
        if dest.exists() and dest.stat().st_size > 10000:
            print(f"[OK] {name} already exists locally: {dest.name} ({dest.stat().st_size // 1024} KB)")
            continue

        print(f"\n[DOWNLOADING] {name} (~{info['size_kb']} KB)...")
        print(f" Source URL: {info['url']}")
        try:
            def _progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / (total_size or 1))
                sys.stdout.write(f"\r Progress: {min(percent, 100)}%")
                sys.stdout.flush()

            urllib.request.urlretrieve(info["url"], dest, reporthook=_progress)
            print(f"\n[SUCCESS] Saved {dest.name}")
        except Exception as e:
            print(f"\n[WARNING] Could not download {name}: {e}")
            print(" The system will automatically use the built-in OpenCV Haar / feature fallback.")
            success = False

    print("\n" + "=" * 65)
    if success:
        print("All models verified successfully! Ready for offline execution.")
    else:
        print("Notice: You can run offline using OpenCV built-in fallback modes.")
    print("=" * 65)

if __name__ == "__main__":
    download_models()
