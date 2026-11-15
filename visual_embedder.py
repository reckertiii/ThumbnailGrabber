"""Compact visual taste embedder for candidate captures (ML-1).

Extracts a dense L2-normalised visual feature vector directly from the speaker crop
so `UserPreferenceModel` is not blind to pixel-level issues that 18 landmark scalars
miss — eyeglasses glare, teeth/tongue mid-syllable, handheld microphone covering the
chin, awkward eyebrow squints, or harsh stage LED washes.

Architecture:
1. Always computes a fast (<0.5ms), zero-dependency 64-D perceptual region descriptor
   covering the eye band, mouth cavity, chin/mic zone, skin chroma, and HOG-lite
   silhouette gradients.
2. When an ONNX vision backbone (`vision_embedder.onnx`, e.g. MobileNetV2 / SqueezeNet
   / MobileCLIP image tower) is available in `TG_DATA_DIR` or `TG_MODELS_DIR` (or
   downloaded on demand), runs CPU inference via `onnxruntime` or `cv2.dnn` — kept
   pinned warm in RAM — and concatenates a projected 64-D neural embedding.
"""

from __future__ import annotations

import os
import threading
import urllib.request
from typing import Any, Dict, Optional

import cv2
import numpy as np

VISION_ONNX_FILENAME = "vision_backbone_mnv2.onnx"
# Compact (~13.5 MB) ImageNet MobileNetV2 ONNX from the official ONNX Model Zoo
VISION_ONNX_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/classification/"
    "mobilenet/model/mobilenetv2-12.onnx"
)
PERCEPTUAL_DIM = 64
NEURAL_PROJ_DIM = 64


def _data_dir() -> str:
    return (
        os.environ.get("TG_DATA_DIR", "").strip()
        or os.environ.get("TG_MODELS_DIR", "").strip()
        or os.path.join(os.getcwd(), "models")
    )


class VisualEmbedder:
    """Thread-safe visual crop embedder kept warm in RAM."""

    def __init__(self, enable_onnx: bool = True) -> None:
        self.enable_onnx = bool(enable_onnx)
        self._lock = threading.Lock()
        self._ort_session: Any = None
        self._cv_net: Any = None
        self._ort_input_name: str = ""
        self._proj_matrix: Optional[np.ndarray] = None
        self._model_loaded: bool = False
        self._backend: str = "perceptual"
        self._load_error: str = ""

        if self.enable_onnx:
            self._try_load_existing_onnx()

    @property
    def model_path(self) -> str:
        custom = os.environ.get("TG_VISION_ONNX_PATH", "").strip()
        if custom:
            return custom
        for root in (
            os.environ.get("TG_DATA_DIR", "").strip(),
            os.environ.get("TG_MODELS_DIR", "").strip(),
            os.path.join(os.getcwd(), "models"),
        ):
            if root:
                candidate = os.path.join(root, VISION_ONNX_FILENAME)
                if os.path.isfile(candidate) and os.path.getsize(candidate) > 1024:
                    return candidate
        return os.path.join(_data_dir(), VISION_ONNX_FILENAME)

    def _get_projection(self, raw_dim: int) -> np.ndarray:
        """Deterministic orthogonal-like Johnson-Lindenstrauss projection to 64-D."""
        if self._proj_matrix is None or self._proj_matrix.shape[0] != raw_dim:
            rng = np.random.default_rng(20261115)
            mat = rng.standard_normal((raw_dim, NEURAL_PROJ_DIM)).astype(np.float32)
            mat /= np.maximum(1e-6, np.linalg.norm(mat, axis=0, keepdims=True))
            self._proj_matrix = mat
        return self._proj_matrix

    def _try_load_existing_onnx(self) -> bool:
        path = self.model_path
        if not os.path.isfile(path) or os.path.getsize(path) < 1024:
            return False
        return self._load_onnx_file(path)

    def _load_onnx_file(self, path: str) -> bool:
        with self._lock:
            if self._model_loaded:
                return True
            # Prefer onnxruntime when installed; fall back to OpenCV DNN
            try:
                import onnxruntime as ort  # type: ignore
                sess_opts = ort.SessionOptions()
                sess_opts.intra_op_num_threads = 2
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
                self._ort_input_name = sess.get_inputs()[0].name
                self._ort_session = sess
                self._model_loaded = True
                self._backend = "onnxruntime"
                self._load_error = ""
                return True
            except Exception as ort_err:
                self._load_error = str(ort_err)

            try:
                net = cv2.dnn.readNetFromONNX(path)
                self._cv_net = net
                self._model_loaded = True
                self._backend = "opencv_dnn"
                self._load_error = ""
                return True
            except Exception as cv_err:
                self._load_error = f"{self._load_error}; cv2.dnn: {cv_err}".strip("; ")
                return False

    def download_backbone(self, timeout_sec: float = 45.0) -> Dict[str, Any]:
        """Downloads the compact MobileNetV2 ONNX backbone into TG_DATA_DIR and pins it in RAM."""
        dest = self.model_path
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not (os.path.isfile(dest) and os.path.getsize(dest) > 1024):
            tmp = f"{dest}.{os.getpid()}.tmp"
            try:
                req = urllib.request.Request(VISION_ONNX_URL, headers={"User-Agent": "ThumbnailGrabber/2.0"})
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp, open(tmp, "wb") as out:
                    out.write(resp.read())
                os.replace(tmp, dest)
            except Exception as e:
                try:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                except OSError:
                    pass
                self._load_error = str(e)
                return {"ok": False, "error": str(e), "backend": self._backend}
        ok = self._load_onnx_file(dest)
        return {"ok": ok, "backend": self._backend, "path": dest, "error": self._load_error}

    def status(self) -> Dict[str, Any]:
        return {
            "backend": self._backend,
            "neural_loaded": self._model_loaded,
            "dim": PERCEPTUAL_DIM + (NEURAL_PROJ_DIM if self._model_loaded else 0),
            "model_path": self.model_path,
            "error": self._load_error,
        }

    @staticmethod
    def extract_perceptual_64d(crop_bgr: np.ndarray) -> np.ndarray:
        """Fast 64-D perceptual descriptor tailored to portrait/cutout flaws."""
        vec = np.zeros(PERCEPTUAL_DIM, dtype=np.float32)
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
            return vec

        img = cv2.resize(crop_bgr, (96, 128), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)

        # Band slices: crown/hair (0..24), eyes/glasses (24..52), nose/cheeks (52..76),
        # mouth/teeth (76..98), chin/mic/neck (98..128)
        bands = [
            gray[0:24, 16:80],
            gray[24:52, 16:80],
            gray[52:76, 16:80],
            gray[76:98, 20:76],
            gray[98:128, 16:80],
        ]
        idx = 0
        # 0..19: per-band mean, std, 95th glare percentile, 10th shadow/cavity percentile (5 * 4 = 20)
        for b in bands:
            vec[idx] = float(np.mean(b))
            vec[idx + 1] = float(np.std(b))
            vec[idx + 2] = float(np.percentile(b, 95))
            vec[idx + 3] = float(np.percentile(b, 10))
            idx += 4

        # 20..25: Eye-band symmetry & glasses specular highlight fraction, mouth dark-cavity & bright-teeth fraction
        eye_band = gray[24:52, 16:80]
        left_eye, right_eye = eye_band[:, :32], np.fliplr(eye_band[:, 32:])
        vec[20] = float(np.mean(np.abs(left_eye - right_eye)))
        vec[21] = float(np.mean(eye_band > 0.92))  # glasses glare
        mouth_band = gray[76:98, 24:72]
        vec[22] = float(np.mean(mouth_band < 0.22))  # open dark oral cavity
        vec[23] = float(np.mean(mouth_band > 0.82))  # bright teeth flash
        chin_band = gray[98:128, 24:72]
        vec[24] = float(np.mean(chin_band < 0.18))   # dark mic capsule / stand under chin
        vec[25] = float(np.std(chin_band))

        # 26..35: Skin & stage LED colour statistics on central face region
        face_bgr = img[28:92, 22:74].astype(np.float32) / 255.0
        face_ycrcb = ycrcb[28:92, 22:74] / 255.0
        face_hsv = hsv[28:92, 22:74]
        vec[26:29] = np.mean(face_bgr, axis=(0, 1))
        vec[29:32] = np.std(face_bgr, axis=(0, 1))
        vec[32] = float(np.mean(face_ycrcb[:, :, 1]))  # Cr (warmth)
        vec[33] = float(np.mean(face_ycrcb[:, :, 2]))  # Cb (blue LED spill)
        vec[34] = float(np.mean(face_hsv[:, :, 1] / 255.0))
        vec[35] = float(np.mean(np.max(face_bgr, axis=2) > 0.97))  # clipped highlight ratio

        # 36..51: 2x2 spatial grid 4-bin gradient orientation histogram (16 dims)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi) * (2.0 / np.pi)  # 0..4
        q_idx = 36
        for r0, r1 in ((0, 64), (64, 128)):
            for c0, c1 in ((0, 48), (48, 96)):
                m_sub = mag[r0:r1, c0:c1]
                a_sub = ang[r0:r1, c0:c1]
                total_m = float(np.sum(m_sub)) + 1e-5
                for b_i in range(4):
                    mask = (a_sub >= b_i) & (a_sub < b_i + 1)
                    vec[q_idx] = float(np.sum(m_sub[mask])) / total_m
                    q_idx += 1

        # 52..63: Outer border vs center contrast + sharpness & horizontal obstruction lines (12 dims)
        lap = cv2.Laplacian(gray[20:100, 18:78], cv2.CV_32F)
        vec[52] = min(1.0, float(np.var(lap)) * 8.0)
        abs_gy_lower = np.abs(gy[80:128, 12:84])
        row_peaks = np.mean(abs_gy_lower > 0.25, axis=1)
        vec[53] = float(np.max(row_peaks)) if row_peaks.size else 0.0
        vec[54] = float(np.mean(row_peaks)) if row_peaks.size else 0.0
        # Hue 9-bin coarse distribution on entire crop
        hue_bins, _ = np.histogram(hsv[:, :, 0], bins=9, range=(0.0, 180.0), density=False)
        vec[55:64] = hue_bins.astype(np.float32) / float(img.shape[0] * img.shape[1])

        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec /= norm
        return vec

    def _extract_neural_64d(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        if not self._model_loaded or crop_bgr is None or crop_bgr.size == 0:
            return None
        try:
            rgb = cv2.cvtColor(cv2.resize(crop_bgr, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            inp = rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            inp = ((inp - mean) / std).transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 224, 224)

            with self._lock:
                if self._ort_session is not None:
                    out = self._ort_session.run(None, {self._ort_input_name: inp})[0]
                elif self._cv_net is not None:
                    self._cv_net.setInput(inp)
                    out = self._cv_net.forward()
                else:
                    return None
            flat = np.asarray(out, dtype=np.float32).reshape(-1)
            if flat.size == 0:
                return None
            proj = flat @ self._get_projection(flat.size)
            n = float(np.linalg.norm(proj))
            if n > 1e-6:
                proj /= n
            return proj.astype(np.float32)
        except Exception:
            return None

    def embed(self, crop_bgr: Optional[np.ndarray], perceptual_only: bool = False) -> np.ndarray:
        """Returns a unit-norm visual embedding (64-D perceptual, or 128-D when ONNX is active)."""
        p_vec = self.extract_perceptual_64d(crop_bgr)
        if perceptual_only or not self._model_loaded:
            return p_vec
        n_vec = self._extract_neural_64d(crop_bgr)
        if n_vec is None:
            return p_vec
        fused = np.concatenate([p_vec * 0.65, n_vec * 0.76]).astype(np.float32)
        norm = float(np.linalg.norm(fused))
        if norm > 1e-6:
            fused /= norm
        return fused


_SHARED_EMBEDDER: Optional[VisualEmbedder] = None
_SHARED_LOCK = threading.Lock()


def get_visual_embedder() -> VisualEmbedder:
    global _SHARED_EMBEDDER
    with _SHARED_LOCK:
        if _SHARED_EMBEDDER is None:
            _SHARED_EMBEDDER = VisualEmbedder(enable_onnx=True)
        return _SHARED_EMBEDDER
