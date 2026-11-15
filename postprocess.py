"""Studio Transparent Cutout Generator (#1) & Broadcast Polish Pipeline (#11).

Designed for TrueNAS servers with abundant RAM:
  - Foreground matting ONNX models (`u2netp.onnx` / `modnet.onnx` / custom `matting.onnx`
    in `TG_DATA_DIR/matting/`) load once and stay pinned warm in RAM for instant
    `<200ms` cutouts.
  - When offline or before any ONNX weight is downloaded, falls back automatically to
    a landmark-seeded GrabCut + guided alpha feathering + edge colour decontamination
    pipeline so `generate_cutout_rgba()` always produces clean 4-channel RGBA PNGs.
  - `polish_broadcast_frame()` removes H.264 RTSP macroblocking, neutralizes stage LED
    colour casts on skin tones, and optionally upscales 2x (1080p -> 4K).
"""

from __future__ import annotations

import os
import threading
import urllib.request
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

MATTING_FILENAME = "u2netp.onnx"
MATTING_URL = (
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
)


def _matting_dir() -> str:
    base = (
        os.environ.get("TG_DATA_DIR", "").strip()
        or os.environ.get("TG_MODELS_DIR", "").strip()
        or os.path.join(os.getcwd(), "models")
    )
    return os.path.join(base, "matting")


class CutoutAndPolishEngine:
    """Thread-safe foreground matting & broadcast image enhancement engine."""

    def __init__(self, auto_load_existing: bool = True) -> None:
        self._lock = threading.Lock()
        self._ort_session: Any = None
        self._cv_net: Any = None
        self._input_name: str = ""
        self._input_size: Tuple[int, int] = (320, 320)
        self._backend: str = "grabcut_guided"
        self._load_error: str = ""

        if auto_load_existing:
            self._try_load_existing()

    @property
    def model_path(self) -> str:
        custom = os.environ.get("TG_MATTING_ONNX_PATH", "").strip()
        if custom:
            return custom
        mdir = _matting_dir()
        for name in ("birefnet_lite.onnx", "isnet-general-use.onnx", "modnet.onnx", MATTING_FILENAME):
            cand = os.path.join(mdir, name)
            if os.path.isfile(cand) and os.path.getsize(cand) > 500_000:
                return cand
        return os.path.join(mdir, MATTING_FILENAME)

    def _try_load_existing(self) -> bool:
        p = self.model_path
        if os.path.isfile(p) and os.path.getsize(p) > 500_000:
            return self._load_onnx(p)
        return False

    def _load_onnx(self, path: str) -> bool:
        with self._lock:
            if self._ort_session is not None or self._cv_net is not None:
                return True
            try:
                import onnxruntime as ort  # type: ignore
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
                inp = sess.get_inputs()[0]
                self._input_name = inp.name
                shape = inp.shape
                if isinstance(shape, (list, tuple)) and len(shape) == 4:
                    ih = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else 320
                    iw = int(shape[3]) if isinstance(shape[3], int) and shape[3] > 0 else 320
                    self._input_size = (iw, ih)
                self._ort_session = sess
                self._backend = f"onnxruntime:{os.path.basename(path)}"
                self._load_error = ""
                return True
            except Exception as e:
                self._load_error = str(e)

            try:
                net = cv2.dnn.readNetFromONNX(path)
                self._cv_net = net
                self._backend = f"opencv_dnn:{os.path.basename(path)}"
                self._load_error = ""
                return True
            except Exception as e2:
                self._load_error = f"{self._load_error}; {e2}".strip("; ")
                return False

    def ensure_matting_model(self, timeout_sec: float = 60.0) -> Dict[str, Any]:
        """Downloads `u2netp.onnx` (~4.7 MB) into TG_DATA_DIR/matting/ and pins it in RAM."""
        if self._ort_session is not None or self._cv_net is not None:
            return {"ok": True, **self.status()}
        dest = self.model_path
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not (os.path.isfile(dest) and os.path.getsize(dest) > 500_000):
            tmp = f"{dest}.{os.getpid()}.tmp"
            try:
                print(f"[PostProcess] Downloading ONNX matting model to '{dest}'...")
                req = urllib.request.Request(MATTING_URL, headers={"User-Agent": "ThumbnailGrabber/2.0"})
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp, open(tmp, "wb") as out:
                    out.write(resp.read())
                os.replace(tmp, dest)
            except Exception as e:
                self._load_error = str(e)
                try:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                except OSError:
                    pass
                return {"ok": False, **self.status()}
        ok = self._load_onnx(dest)
        return {"ok": ok, **self.status()}

    def status(self) -> Dict[str, Any]:
        return {
            "backend": self._backend,
            "onnx_loaded": (self._ort_session is not None or self._cv_net is not None),
            "model_path": self.model_path,
            "error": self._load_error,
        }

    def _predict_onnx_alpha(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        if self._ort_session is None and self._cv_net is None:
            return None
        try:
            h, w = bgr.shape[:2]
            iw, ih = self._input_size
            rgb = cv2.cvtColor(cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            inp = rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            blob = ((inp - mean) / std).transpose(2, 0, 1)[np.newaxis, ...]

            with self._lock:
                if self._ort_session is not None:
                    raw = self._ort_session.run(None, {self._input_name: blob})[0]
                elif self._cv_net is not None:
                    self._cv_net.setInput(blob)
                    raw = self._cv_net.forward()
                else:
                    return None

            mask = np.squeeze(np.asarray(raw, dtype=np.float32))
            if mask.ndim > 2:
                mask = mask[0]
            mn, mx = float(mask.min()), float(mask.max())
            if mx - mn > 1e-6:
                mask = (mask - mn) / (mx - mn)
            else:
                return None
            alpha = cv2.resize(mask, (w, h), interpolation=cv2.INTER_CUBIC)
            return np.clip(alpha, 0.0, 1.0)
        except Exception:
            return None

    @staticmethod
    def _predict_guided_grabcut_alpha(bgr: np.ndarray, analysis: Any = None) -> np.ndarray:
        """Offline landmark-seeded GrabCut + soft edge feathering fallback."""
        h, w = bgr.shape[:2]
        # Work at ~480p max dimension for speed (<90ms), then refine at full res
        max_dim = 480
        scale = min(1.0, max_dim / float(max(h, w)))
        sw, sh = max(24, int(round(w * scale))), max(24, int(round(h * scale)))
        small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)

        gc_mask = np.full((sh, sw), cv2.GC_PR_BGD, dtype=np.uint8)

        # Subject & face hints from AnalysisResult when available
        subj_rect = None
        face_rect = None
        if analysis is not None:
            faces = getattr(analysis, "faces", None) or []
            primary = next((f for f in faces if getattr(f, "is_selected", True)), faces[0]) if faces else None
            subj_rect = getattr(primary, "subject_bbox", None) if primary else None
            face_rect = (getattr(primary, "bbox", None) if primary else None) or getattr(analysis, "face_bbox", None)

        if subj_rect and subj_rect[2] > 8 and subj_rect[3] > 8:
            sx, sy, s_w, s_h = [int(round(v * scale)) for v in subj_rect]
        elif face_rect and face_rect[2] > 8 and face_rect[3] > 8:
            fx, fy, fw, fh = [int(round(v * scale)) for v in face_rect]
            sx = max(0, fx - fw)
            sy = max(0, fy - int(fh * 0.35))
            s_w = min(sw - sx, fw * 3)
            s_h = min(sh - sy, int(fh * 7.5))
        else:
            sx, sy = int(sw * 0.14), int(sh * 0.06)
            s_w, s_h = int(sw * 0.72), int(sh * 0.92)

        sx0, sy0 = max(1, sx), max(1, sy)
        sx1, sy1 = min(sw - 1, sx + s_w), min(sh - 1, sy + s_h)

        # Probable foreground inside subject box; definite background on outer frame rim
        gc_mask[:, :] = cv2.GC_BGD
        pad_x = max(2, int((sx1 - sx0) * 0.12))
        pad_y = max(2, int((sy1 - sy0) * 0.08))
        gc_mask[max(0, sy0 - pad_y):min(sh, sy1 + pad_y),
                max(0, sx0 - pad_x):min(sw, sx1 + pad_x)] = cv2.GC_PR_BGD
        gc_mask[sy0:sy1, sx0:sx1] = cv2.GC_PR_FGD

        # Definite foreground spine along face + central torso
        if face_rect and face_rect[2] > 6 and face_rect[3] > 6:
            fx, fy, fw, fh = [int(round(v * scale)) for v in face_rect]
            fx0 = max(0, fx + int(fw * 0.15))
            fx1 = min(sw, fx + int(fw * 0.85))
            fy0 = max(0, fy + int(fh * 0.10))
            fy1 = min(sh, fy + int(fh * 0.92))
            if fx1 > fx0 and fy1 > fy0:
                gc_mask[fy0:fy1, fx0:fx1] = cv2.GC_FGD
            # Central torso column below face
            tcx0 = max(0, fx - int(fw * 0.25))
            tcx1 = min(sw, fx + fw + int(fw * 0.25))
            tcy0 = min(sh, fy + fh)
            tcy1 = min(sh, sy1)
            if tcx1 > tcx0 and tcy1 > tcy0:
                gc_mask[tcy0:tcy1, tcx0:tcx1] = cv2.GC_FGD
        else:
            cx0 = sx0 + int((sx1 - sx0) * 0.30)
            cx1 = sx0 + int((sx1 - sx0) * 0.70)
            cy0 = sy0 + int((sy1 - sy0) * 0.12)
            cy1 = sy1
            if cx1 > cx0 and cy1 > cy0:
                gc_mask[cy0:cy1, cx0:cx1] = cv2.GC_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(small, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
            fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
        except Exception:
            fg = np.zeros((sh, sw), dtype=np.float32)
            fg[sy0:sy1, sx0:sx1] = 1.0

        # Clean small islands and smooth silhouette
        fg_u8 = (fg * 255).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, k)
        fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, k)
        alpha = cv2.resize(fg_u8.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.6)
        return np.clip((alpha - 0.12) / 0.78, 0.0, 1.0)

    @staticmethod
    def _decontaminate_edges(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Suppresses background colour spill / dark fringes along semi-transparent hair edges."""
        out = bgr.astype(np.float32)
        # Estimate clean interior foreground colour by eroding alpha slightly and diffusing outward
        solid = (alpha > 0.88).astype(np.uint8)
        if int(np.count_nonzero(solid)) < 32:
            return bgr
        eroded = cv2.erode(solid, np.ones((5, 5), np.uint8), iterations=1)
        fringe = (alpha > 0.04) & (alpha < 0.92) & (eroded == 0)
        if not np.any(fringe):
            return bgr
        blurred_fg = cv2.GaussianBlur(out, (15, 15), sigmaX=5.0)
        blend_w = ((1.0 - alpha) * 0.55)[..., np.newaxis]
        corrected = out * (1.0 - blend_w) + blurred_fg * blend_w
        out[fringe] = corrected[fringe]
        return np.clip(out, 0, 255).astype(np.uint8)

    def generate_cutout_rgba(
        self,
        frame_bgr: np.ndarray,
        analysis: Any = None,
        crop_to_subject: bool = False,
        padding: float = 0.06,
        try_download_onnx: bool = False,
    ) -> np.ndarray:
        """Produces a 4-channel BGRA image with clean alpha matte and de-fringed edges."""
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros((1, 1, 4), dtype=np.uint8)

        if try_download_onnx and self._ort_session is None and self._cv_net is None:
            self.ensure_matting_model(timeout_sec=25.0)

        work_bgr = frame_bgr
        work_analysis = analysis
        if crop_to_subject and analysis is not None:
            from stream_processor import subject_crop_rect
            h, w = frame_bgr.shape[:2]
            rect = subject_crop_rect(analysis, w, h, padding=padding)
            if rect:
                rx, ry, rw, rh = rect
                sub = frame_bgr[ry:ry + rh, rx:rx + rw]
                if sub.size > 0:
                    work_bgr = sub
                    work_analysis = None  # cropped tightly around the subject already

        alpha = self._predict_onnx_alpha(work_bgr)
        if alpha is None:
            alpha = self._predict_guided_grabcut_alpha(work_bgr, work_analysis)

        # Subtle morphological cleanup on near-zero background haze
        alpha = np.where(alpha < 0.03, 0.0, np.where(alpha > 0.98, 1.0, alpha)).astype(np.float32)
        clean_bgr = self._decontaminate_edges(work_bgr, alpha)
        alpha_u8 = np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)
        return np.dstack([clean_bgr, alpha_u8])

    @staticmethod
    def polish_broadcast_frame(
        frame_bgr: np.ndarray,
        analysis: Any = None,
        deblock: bool = True,
        neutralize_led: bool = True,
        upscale_2x: bool = False,
    ) -> np.ndarray:
        """Applies natural broadcast finishing: RTSP H.264 de-blocking, stage LED skin
        cast neutralization, and optional crisp 2x (1080p -> 4K) super-resolution."""
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr
        out = frame_bgr.copy()

        # 1. Subtle H.264 macroblock / mosquito-noise smoothing in chroma + flat areas
        # while preserving sharp eye/beard luminance edges.
        if deblock:
            ycrcb = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
            y_ch, cr_ch, cb_ch = cv2.split(ycrcb)
            # Chroma carries most 4:2:0 RTSP blockiness around hair edges
            cr_smooth = cv2.bilateralFilter(cr_ch, d=5, sigmaColor=14, sigmaSpace=14)
            cb_smooth = cv2.bilateralFilter(cb_ch, d=5, sigmaColor=14, sigmaSpace=14)
            y_smooth = cv2.bilateralFilter(y_ch, d=5, sigmaColor=9, sigmaSpace=9)
            # Keep 100% original luma on crisp detail edges, smooth flat macroblocks
            gx = cv2.Sobel(y_ch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(y_ch, cv2.CV_32F, 0, 1, ksize=3)
            edge_w = np.clip(np.hypot(gx, gy) / 48.0, 0.0, 1.0)
            y_out = np.clip(y_ch.astype(np.float32) * edge_w + y_smooth.astype(np.float32) * (1.0 - edge_w), 0, 255).astype(np.uint8)
            out = cv2.cvtColor(cv2.merge([y_out, cr_smooth, cb_smooth]), cv2.COLOR_YCrCb2BGR)

        # 2. Face-masked stage LED colour-cast neutralization
        if neutralize_led:
            h, w = out.shape[:2]
            face_bbox = None
            if analysis is not None:
                faces = getattr(analysis, "faces", None) or []
                primary = next((f for f in faces if getattr(f, "is_selected", True)), faces[0]) if faces else None
                face_bbox = (getattr(primary, "bbox", None) if primary else None) or getattr(analysis, "face_bbox", None)
            if face_bbox and face_bbox[2] > 16 and face_bbox[3] > 16:
                fx, fy, fw, fh = face_bbox
                rx0, ry0 = max(0, fx + int(fw * 0.18)), max(0, fy + int(fh * 0.18))
                rx1, ry1 = min(w, fx + int(fw * 0.82)), min(h, fy + int(fh * 0.82))
            else:
                rx0, ry0 = int(w * 0.35), int(h * 0.18)
                rx1, ry1 = int(w * 0.65), int(h * 0.55)

            roi = out[ry0:ry1, rx0:rx1]
            if roi.size >= 64:
                lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
                l_vals = lab[:, :, 0]
                mid_mask = (l_vals > 45.0) & (l_vals < 230.0)
                if int(np.count_nonzero(mid_mask)) >= 32:
                    mean_a = float(np.mean(lab[:, :, 1][mid_mask]))  # 128 neutral; healthy skin ~136-145
                    mean_b = float(np.mean(lab[:, :, 2][mid_mask]))  # 128 neutral; healthy skin ~136-148
                    # Only correct unnatural blue/violet (mean_b < 126), green (mean_a < 126),
                    # or extreme magenta/red stage LED spill (mean_a > 154)
                    shift_a = 0.0
                    shift_b = 0.0
                    if mean_a < 128.0:
                        shift_a = min(12.0, (134.0 - mean_a) * 0.65)
                    elif mean_a > 152.0:
                        shift_a = max(-14.0, (145.0 - mean_a) * 0.65)
                    if mean_b < 128.0:
                        shift_b = min(15.0, (136.0 - mean_b) * 0.70)
                    elif mean_b > 156.0:
                        shift_b = max(-10.0, (147.0 - mean_b) * 0.55)

                    if abs(shift_a) >= 1.0 or abs(shift_b) >= 1.0:
                        full_lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
                        # Weight correction strongest on midtones so deep black shadows stay black
                        l_norm = full_lab[:, :, 0] / 255.0
                        mid_w = np.clip(4.0 * l_norm * (1.0 - l_norm), 0.15, 1.0)
                        full_lab[:, :, 1] = np.clip(full_lab[:, :, 1] + shift_a * mid_w, 0, 255)
                        full_lab[:, :, 2] = np.clip(full_lab[:, :, 2] + shift_b * mid_w, 0, 255)
                        out = cv2.cvtColor(full_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # 3. Crisp 2x Upscale (e.g. 1920x1080 -> 3840x2160) with halo-free detail enhancement
        if upscale_2x:
            uh, uw = out.shape[0] * 2, out.shape[1] * 2
            up = cv2.resize(out, (uw, uh), interpolation=cv2.INTER_LANCZOS4)
            blur = cv2.GaussianBlur(up, (0, 0), sigmaX=1.15)
            diff = up.astype(np.float32) - blur.astype(np.float32)
            # Clamp sharpening delta to +-14 levels so high-contrast hair edges never halo
            diff = np.clip(diff * 0.55, -14.0, 14.0)
            out = np.clip(up.astype(np.float32) + diff, 0, 255).astype(np.uint8)

        return out


_SHARED_ENGINE: Optional[CutoutAndPolishEngine] = None
_ENGINE_LOCK = threading.Lock()


def get_postprocess_engine() -> CutoutAndPolishEngine:
    global _SHARED_ENGINE
    with _ENGINE_LOCK:
        if _SHARED_ENGINE is None:
            _SHARED_ENGINE = CutoutAndPolishEngine(auto_load_existing=True)
        return _SHARED_ENGINE


def cutout_path_for(filepath: str) -> str:
    stem, _ = os.path.splitext(filepath)
    return f"{stem}_cutout.png"


def save_cutout_png(
    frame_bgr: np.ndarray,
    analysis: Any,
    dest_path: str,
    crop_to_subject: bool = False,
    polish_first: bool = True,
) -> Optional[str]:
    """Generates and writes a lossless transparent RGBA PNG to `dest_path`."""
    try:
        engine = get_postprocess_engine()
        src = (
            engine.polish_broadcast_frame(frame_bgr, analysis, deblock=True, neutralize_led=True, upscale_2x=False)
            if polish_first
            else frame_bgr
        )
        rgba = engine.generate_cutout_rgba(src, analysis, crop_to_subject=crop_to_subject)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        ok = cv2.imwrite(dest_path, rgba, [int(cv2.IMWRITE_PNG_COMPRESSION), 2])
        return dest_path if ok else None
    except Exception as e:
        print(f"[PostProcess] Failed to write cutout PNG '{dest_path}': {e}")
        return None
