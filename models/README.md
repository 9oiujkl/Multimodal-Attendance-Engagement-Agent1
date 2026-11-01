# Pretrained Model Weights Directory

This folder holds locally cached, offline model weights used by the computer vision pipeline.

## 1. Required Models

The application utilizes official open-source models from the **OpenCV Model Zoo** (Apache 2.0 License):

1. **`face_detection_yunet_2023mar.onnx`**
   - **Task**: Real-time multi-face detection & 5-point facial landmark estimation (eyes, nose, mouth corners).
   - **Size**: ~390 KB.
   - **Source**: [OpenCV Zoo - YuNet](https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet)

2. **`face_recognition_sface_2021dec.onnx`**
   - **Task**: 128-dimensional face embedding extraction for cosine similarity comparison.
   - **Size**: ~37 MB.
   - **Source**: [OpenCV Zoo - SFace](https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface)

---

## 2. Automated Download Script

You can download both models directly into this folder using the helper script:
```powershell
python modules/download_models.py
```

---

## 3. Manual Download Links

If downloading manually, save both files directly inside this `models/` folder:
- **YuNet**: `https://raw.githubusercontent.com/opencv/opencv_zoo/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx`
- **SFace**: `https://raw.githubusercontent.com/opencv/opencv_zoo/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx`

---

## 4. Offline Fallback Mode

If the ONNX models are not yet downloaded or if the environment is strictly air-gapped without external network access, the system automatically falls back to:
- **OpenCV's built-in Haar Cascade classifier** (`haarcascade_frontalface_default.xml`) for face detection.
- **Normalized Grayscale Histogram / LBPH Feature Embeddings** for face recognition and cosine matching.

This guarantees that the application **never crashes** and remains **100% functional and demonstrable** in any offline viva/lab presentation!
