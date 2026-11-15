# Third-party notices

Thumbnail Grabber is proprietary (see `LICENSE`), and depends on third-party
Python packages that carry their own permissive licences.

Licences recorded from the installed distributions:

| Package | Licence | Permissive? |
|---|---|---|
| opencv-python | Apache-2.0 | yes |
| mediapipe | Apache-2.0 | yes |
| numpy | BSD-3-Clause | yes |
| pillow | MIT-CMU (HPND) | yes |
| scikit-learn | BSD-3-Clause | yes |
| fastapi | MIT | yes |
| uvicorn | BSD-3-Clause | yes |
| python-multipart | Apache-2.0 | yes |
| pydantic | MIT | yes |
| anthropic | MIT | yes |
| onnxruntime | MIT | yes |
| faster-whisper | MIT | yes |

Optional neural model weights downloaded on demand into `TG_DATA_DIR` (never required for offline/fallback operation):

| Model / Runtime | Source / Licence | Permissive? |
|---|---|---|
| YuNet & SFace (`face_recognition_sface_2021dec.onnx`) | OpenCV Zoo — Apache-2.0 | yes |
| MediaPipe Face / Hand / Pose Landmarkers | Google MediaPipe — Apache-2.0 | yes |
| U2-Net-P (`u2netp.onnx`) | Xuebin Qin et al. (`U-2-Net`) — Apache-2.0 | yes |
| MobileNetV2 ONNX (`vision_backbone_mnv2.onnx`) | ONNX Model Zoo — Apache-2.0 | yes |
| Whisper CTranslate2 weights (via `faster-whisper`) | OpenAI / Systran — MIT | yes |
| NDI Runtime (`libndi`, optional user-supplied binary in `TG_DATA_DIR/ndi`) | Vizrt / NDI SDK End-User Licence (not bundled or redistributed) | user-supplied |

*(Note: the legacy PyQt6 desktop frontend was retired and removed so the codebase and published container image rely exclusively on permissive runtime libraries.)*

This is a summary written for orientation, not legal advice.
