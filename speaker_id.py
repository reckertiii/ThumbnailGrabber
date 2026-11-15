"""Target Speaker Face Lock (#3) & Per-Speaker Adaptive Baselines (ML-5).

Uses OpenCV's `FaceRecognizerSF` (`face_recognition_sface_2021dec.onnx`, ~36.9 MB)
when present or downloaded to `TG_DATA_DIR`, with a deterministic aligned-face
2D-DCT + chroma fallback so tests and offline deployments work seamlessly.

Capabilities:
  - Click-to-lock onto a detected face on the live preview (ignores worship team,
    hosts, and stage walk-ons).
  - Save named Speaker Profiles (`Lead Pastor`, `Guest Speaker`, etc.) persisted
    in `TG_DATA_DIR/speakers.json` with 64x64 avatar thumbnails.
  - Track rolling Welford per-speaker expression baselines (resting eye openness,
    typical smile width, mouth openness) so narrow-eyed or subtle-smiling speakers
    are judged relative to their own face rather than a generic average.
"""

from __future__ import annotations

import base64
import json
import math
import os
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
SPEAKERS_FILE = "speakers.json"
DEFAULT_SFACE_THRESHOLD = 0.363
DEFAULT_FALLBACK_THRESHOLD = 0.78


def _data_dir() -> str:
    return (
        os.environ.get("TG_DATA_DIR", "").strip()
        or os.path.join(os.getcwd(), "models")
    )


def _models_dir() -> str:
    return (
        os.environ.get("TG_MODELS_DIR", "").strip()
        or os.path.join(os.getcwd(), "models")
    )


@dataclass
class SpeakerBaseline:
    """Online Welford statistics for one speaker's resting/engaged facial metrics."""
    count: int = 0
    ear_mean: float = 0.32
    ear_m2: float = 0.0
    smile_mean: float = 0.22
    smile_m2: float = 0.0
    mar_mean: float = 0.14
    mar_m2: float = 0.0

    def update(self, ear: float, smile: float, mar: float) -> None:
        self.count += 1
        n = float(self.count)
        for attr_m, attr_m2, val in (
            ("ear_mean", "ear_m2", float(ear)),
            ("smile_mean", "smile_m2", float(smile)),
            ("mar_mean", "mar_m2", float(mar)),
        ):
            mean = getattr(self, attr_m)
            delta = val - mean
            new_mean = mean + delta / n
            delta2 = val - new_mean
            setattr(self, attr_m, new_mean)
            setattr(self, attr_m2, getattr(self, attr_m2) + delta * delta2)

    def ear_std(self) -> float:
        return math.sqrt(self.ear_m2 / max(1, self.count - 1)) if self.count >= 6 else 0.04

    def smile_std(self) -> float:
        return math.sqrt(self.smile_m2 / max(1, self.count - 1)) if self.count >= 6 else 0.10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": int(self.count),
            "ear_mean": round(self.ear_mean, 4),
            "ear_m2": round(self.ear_m2, 5),
            "smile_mean": round(self.smile_mean, 4),
            "smile_m2": round(self.smile_m2, 5),
            "mar_mean": round(self.mar_mean, 4),
            "mar_m2": round(self.mar_m2, 5),
        }

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "SpeakerBaseline":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            count=max(0, int(raw.get("count", 0) or 0)),
            ear_mean=float(raw.get("ear_mean", 0.32) or 0.32),
            ear_m2=max(0.0, float(raw.get("ear_m2", 0.0) or 0.0)),
            smile_mean=float(raw.get("smile_mean", 0.22) or 0.22),
            smile_m2=max(0.0, float(raw.get("smile_m2", 0.0) or 0.0)),
            mar_mean=float(raw.get("mar_mean", 0.14) or 0.14),
            mar_m2=max(0.0, float(raw.get("mar_m2", 0.0) or 0.0)),
        )


@dataclass
class SpeakerProfile:
    id: str
    name: str
    embedding: List[float]
    backend: str = "sface"
    thumb_b64: str = ""
    created_at: float = field(default_factory=time.time)
    baseline: SpeakerBaseline = field(default_factory=SpeakerBaseline)

    def to_public(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "backend": self.backend,
            "thumb_b64": self.thumb_b64,
            "created_at": round(self.created_at, 1),
            "baseline_samples": self.baseline.count,
            "ear_mean": round(self.baseline.ear_mean, 3),
            "smile_mean": round(self.baseline.smile_mean, 3),
        }


class SpeakerIdentifier:
    """Manages SFace face recognition, active speaker lock, and per-speaker baselines."""

    def __init__(self, auto_download_sface: bool = False) -> None:
        self._lock = threading.RLock()
        self._recognizer: Any = None
        self._backend: str = "dct_fallback"
        self._load_error: str = ""

        # Active lock state
        self.locked: bool = False
        self.active_profile_id: Optional[str] = None
        self.active_name: str = ""
        self.target_embedding: Optional[np.ndarray] = None
        self.target_backend: str = "dct_fallback"
        self.target_thumb_b64: str = ""
        self.similarity_threshold: float = DEFAULT_SFACE_THRESHOLD
        self._adhoc_baseline = SpeakerBaseline()

        # Saved profiles keyed by id
        self.profiles: Dict[str, SpeakerProfile] = {}

        self._try_init_sface(allow_download=auto_download_sface)
        self.load_profiles()

    @property
    def sface_path(self) -> str:
        for d in (_data_dir(), _models_dir()):
            p = os.path.join(d, SFACE_FILENAME)
            if os.path.isfile(p) and os.path.getsize(p) > 1_000_000:
                return p
        return os.path.join(_data_dir(), SFACE_FILENAME)

    def _speakers_json_path(self) -> str:
        return os.path.join(_data_dir(), SPEAKERS_FILE)

    def _try_init_sface(self, allow_download: bool = False) -> bool:
        with self._lock:
            if self._recognizer is not None:
                return True
            if not hasattr(cv2, "FaceRecognizerSF"):
                self._load_error = "cv2.FaceRecognizerSF unavailable"
                return False
            path = self.sface_path
            if not (os.path.isfile(path) and os.path.getsize(path) > 1_000_000):
                if not allow_download:
                    return False
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = f"{path}.{os.getpid()}.tmp"
                try:
                    print(f"[SpeakerID] Downloading SFace ONNX model to '{path}'...")
                    req = urllib.request.Request(SFACE_URL, headers={"User-Agent": "ThumbnailGrabber/2.0"})
                    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
                        out.write(resp.read())
                    os.replace(tmp, path)
                except Exception as e:
                    self._load_error = str(e)
                    try:
                        if os.path.exists(tmp):
                            os.unlink(tmp)
                    except OSError:
                        pass
                    return False
            try:
                self._recognizer = cv2.FaceRecognizerSF.create(path, "")
                self._backend = "sface"
                self._load_error = ""
                return True
            except Exception as e:
                self._load_error = str(e)
                return False

    def ensure_sface_ready(self) -> Dict[str, Any]:
        ok = self._try_init_sface(allow_download=True)
        return {"ok": ok, "backend": self._backend, "path": self.sface_path, "error": self._load_error}

    @staticmethod
    def _fallback_embed(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Deterministic 128-D aligned face appearance descriptor (offline fallback)."""
        h, w = frame_bgr.shape[:2]
        x, y, bw, bh = bbox
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(w, int(x + bw)), min(h, int(y + bh))
        if x1 - x0 < 12 or y1 - y0 < 12:
            return None
        crop = cv2.resize(frame_bgr[y0:y1, x0:x1], (64, 64), interpolation=cv2.INTER_AREA)
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
        # Illumination-normalised luma DCT (low-frequency 8x8 = 64 dims, drop DC [0,0])
        luma = cv2.equalizeHist(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 255.0
        dct = cv2.dct(luma)[:8, :8].flatten()
        dct[0] = 0.0
        # Spatial 4x4 Cr/Cb colour + hair/beard chroma layout (4*4*4 = 64 dims)
        small = cv2.resize(ycrcb, (4, 4), interpolation=cv2.INTER_AREA)
        hsv_small = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0, (4, 4))
        chroma = np.concatenate([
            small[:, :, 1].flatten(),
            small[:, :, 2].flatten(),
            hsv_small[:, :, 0].flatten(),
            hsv_small[:, :, 1].flatten(),
        ])
        vec = np.concatenate([dct, chroma]).astype(np.float32)
        n = float(np.linalg.norm(vec))
        if n > 1e-6:
            vec /= n
        return vec

    def extract_embedding(
        self,
        frame_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        raw_yunet_row: Optional[np.ndarray] = None,
        prefer_backend: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], str]:
        """Extracts a unit-norm face identity vector and the backend tag used."""
        if frame_bgr is None or frame_bgr.size == 0:
            return None, "none"
        use_sface = (prefer_backend or self._backend) == "sface" and self._recognizer is not None
        if use_sface:
            try:
                if raw_yunet_row is not None and len(raw_yunet_row) >= 14:
                    face_mat = np.asarray(raw_yunet_row, dtype=np.float32).reshape(1, -1)
                    aligned = self._recognizer.alignCrop(frame_bgr, face_mat)
                else:
                    x, y, bw, bh = bbox
                    h, w = frame_bgr.shape[:2]
                    crop = frame_bgr[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
                    if crop.size == 0:
                        return None, "none"
                    aligned = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
                feat = self._recognizer.feature(aligned)
                vec = np.asarray(feat, dtype=np.float32).reshape(-1)
                n = float(np.linalg.norm(vec))
                if n > 1e-6:
                    vec /= n
                return vec, "sface"
            except Exception:
                pass
        vec = self._fallback_embed(frame_bgr, bbox)
        return vec, "dct_fallback"

    @staticmethod
    def _encode_avatar(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        h, w = frame_bgr.shape[:2]
        x, y, bw, bh = bbox
        pad = int(max(bw, bh) * 0.22)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return ""
        thumb = cv2.resize(crop, (72, 72), interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return base64.b64encode(enc.tobytes()).decode("ascii") if ok else ""

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 1e-6 or nb <= 1e-6:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def lock_from_frame(
        self,
        frame_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        raw_yunet_row: Optional[np.ndarray] = None,
        save_as_name: str = "",
        download_sface: bool = True,
    ) -> Dict[str, Any]:
        """Locks onto the specified face in `frame_bgr`, optionally saving a named profile."""
        if download_sface and self._recognizer is None:
            self._try_init_sface(allow_download=True)
        emb, backend = self.extract_embedding(frame_bgr, bbox, raw_yunet_row=raw_yunet_row)
        if emb is None:
            return {"ok": False, "error": "Could not extract face embedding from crop."}

        thumb = self._encode_avatar(frame_bgr, bbox)
        with self._lock:
            self.locked = True
            self.target_embedding = emb
            self.target_backend = backend
            self.target_thumb_b64 = thumb
            self.similarity_threshold = (
                DEFAULT_SFACE_THRESHOLD if backend == "sface" else DEFAULT_FALLBACK_THRESHOLD
            )
            self._adhoc_baseline = SpeakerBaseline()

            clean_name = (save_as_name or "").strip()
            if clean_name:
                prof_id = uuid.uuid4().hex[:10]
                prof = SpeakerProfile(
                    id=prof_id,
                    name=clean_name,
                    embedding=emb.tolist(),
                    backend=backend,
                    thumb_b64=thumb,
                )
                self.profiles[prof_id] = prof
                self.active_profile_id = prof_id
                self.active_name = clean_name
                self.save_profiles()
            else:
                self.active_profile_id = None
                self.active_name = "Locked Speaker"

        return {"ok": True, **self.status()}

    def activate_profile(self, profile_id: str) -> Dict[str, Any]:
        with self._lock:
            prof = self.profiles.get(profile_id)
            if prof is None:
                return {"ok": False, "error": f"Unknown speaker profile '{profile_id}'"}
            if prof.backend == "sface" and self._recognizer is None:
                self._try_init_sface(allow_download=True)
            self.locked = True
            self.active_profile_id = prof.id
            self.active_name = prof.name
            self.target_embedding = np.asarray(prof.embedding, dtype=np.float32)
            self.target_backend = prof.backend
            self.target_thumb_b64 = prof.thumb_b64
            self.similarity_threshold = (
                DEFAULT_SFACE_THRESHOLD if prof.backend == "sface" else DEFAULT_FALLBACK_THRESHOLD
            )
            return {"ok": True, **self.status()}

    def unlock(self) -> Dict[str, Any]:
        with self._lock:
            self.locked = False
            self.active_profile_id = None
            self.active_name = ""
            self.target_embedding = None
            self.target_thumb_b64 = ""
            return {"ok": True, **self.status()}

    def delete_profile(self, profile_id: str) -> bool:
        with self._lock:
            removed = self.profiles.pop(profile_id, None) is not None
            if removed:
                if self.active_profile_id == profile_id:
                    self.unlock()
                self.save_profiles()
            return removed

    def evaluate_and_filter_faces(
        self,
        frame_bgr: np.ndarray,
        analyzed_faces: List[Any],
        raw_yunet_rows: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        """Matches detected faces against the active target lock & applies speaker baselines."""
        with self._lock:
            is_locked = self.locked and self.target_embedding is not None
            target_emb = self.target_embedding
            target_backend = self.target_backend
            thresh = self.similarity_threshold
            active_prof = self.profiles.get(self.active_profile_id) if self.active_profile_id else None
            speaker_label = self.active_name or "Target Speaker"

        if not analyzed_faces:
            return

        if not is_locked or target_emb is None:
            # Even when unlocked, if only 1 person is tracked, update ad-hoc baseline gently
            if len(analyzed_faces) == 1:
                f0 = analyzed_faces[0]
                if getattr(f0, "is_selected", True):
                    self._apply_baseline_adjustment(self._adhoc_baseline, f0, persist_prof=None)
            return

        best_face = None
        best_sim = -1.0
        sims: List[Tuple[Any, float]] = []
        for f_item in analyzed_faces:
            bbox = getattr(f_item, "raw_bbox", None) or getattr(f_item, "bbox", None)
            if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
                sims.append((f_item, 0.0))
                continue
            raw_row = (raw_yunet_rows or {}).get(f_item.person_id)
            emb, used_backend = self.extract_embedding(
                frame_bgr, bbox, raw_yunet_row=raw_row, prefer_backend=target_backend
            )
            if emb is None or used_backend != target_backend:
                sim = 0.0
            else:
                sim = self.cosine_similarity(target_emb, emb)
            setattr(f_item, "speaker_similarity", round(sim, 3))
            sims.append((f_item, sim))
            if sim > best_sim:
                best_sim = sim
                best_face = f_item

        for f_item, sim in sims:
            is_match = (f_item is best_face) and (sim >= thresh)
            if is_match:
                f_item.is_selected = True
                setattr(f_item, "speaker_name", speaker_label)
                baseline = active_prof.baseline if active_prof is not None else self._adhoc_baseline
                self._apply_baseline_adjustment(baseline, f_item, persist_prof=active_prof)
            else:
                f_item.is_selected = False
                setattr(f_item, "speaker_name", "")
                reasons = getattr(f_item, "disqualified_reasons", None)
                if isinstance(reasons, list) and "not locked speaker" not in reasons:
                    reasons.append("not locked speaker")

    def _apply_baseline_adjustment(
        self,
        baseline: SpeakerBaseline,
        f_item: Any,
        persist_prof: Optional[SpeakerProfile],
    ) -> None:
        """Updates rolling Welford speaker stats and applies personal Z-score adjustments."""
        if not getattr(f_item, "is_blinking", False) and not getattr(f_item, "is_looking_down", False):
            baseline.update(
                ear=float(getattr(f_item, "avg_ear", 0.32) or 0.32),
                smile=float(getattr(f_item, "smile_ratio", 0.22) or 0.22),
                mar=float(getattr(f_item, "mar", 0.14) or 0.14),
            )
            if persist_prof is not None and baseline.count % 25 == 0:
                self.save_profiles()

        if baseline.count < 10:
            return

        # 1. Narrow-eye protection: if this speaker's normal wide-open EAR is naturally low
        # (e.g. 0.23) and current EAR is at or above their personal mean, soften false squint drops.
        curr_ear = float(getattr(f_item, "avg_ear", 0.32) or 0.32)
        if not getattr(f_item, "is_blinking", False) and curr_ear >= baseline.ear_mean - 0.015:
            if getattr(f_item, "eye_openness_score", 1.0) < 0.92:
                f_item.eye_openness_score = round(min(1.0, (f_item.eye_openness_score + 0.95) * 0.5), 2)

        # 2. Personal top-decile smile bonus: reward shots where this speaker is smiling
        # noticeably warmer (+1 SD) than their own baseline.
        curr_smile = float(getattr(f_item, "smile_ratio", 0.0) or 0.0)
        smile_z = (curr_smile - baseline.smile_mean) / max(0.04, baseline.smile_std())
        setattr(f_item, "speaker_smile_z", round(smile_z, 2))
        if smile_z >= 1.0 and not getattr(f_item, "is_awkward_expression", False):
            f_item.expression_score = round(min(1.0, f_item.expression_score + min(0.06, 0.03 * smile_z)), 2)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            active_prof = self.profiles.get(self.active_profile_id) if self.active_profile_id else None
            bl = active_prof.baseline if active_prof is not None else self._adhoc_baseline
            return {
                "locked": self.locked,
                "active_profile_id": self.active_profile_id,
                "active_name": self.active_name,
                "backend": self._backend,
                "target_backend": self.target_backend,
                "similarity_threshold": round(self.similarity_threshold, 3),
                "thumb_b64": self.target_thumb_b64,
                "baseline_samples": bl.count,
                "profiles": [p.to_public() for p in self.profiles.values()],
            }

    def save_profiles(self) -> None:
        with self._lock:
            path = self._speakers_json_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {
                "version": 1,
                "active_profile_id": self.active_profile_id if self.locked else None,
                "profiles": [
                    {
                        "id": p.id,
                        "name": p.name,
                        "embedding": [round(float(v), 6) for v in p.embedding],
                        "backend": p.backend,
                        "thumb_b64": p.thumb_b64,
                        "created_at": p.created_at,
                        "baseline": p.baseline.to_dict(),
                    }
                    for p in self.profiles.values()
                ],
            }
            tmp = f"{path}.{os.getpid()}.tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp, path)
            except OSError as e:
                print(f"[SpeakerID] Could not save {SPEAKERS_FILE}: {e}")

    def load_profiles(self) -> None:
        path = self._speakers_json_path()
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            with self._lock:
                self.profiles.clear()
                for item in (raw.get("profiles") or []):
                    pid = str(item.get("id") or "").strip()
                    emb = item.get("embedding") or []
                    if not pid or not emb:
                        continue
                    self.profiles[pid] = SpeakerProfile(
                        id=pid,
                        name=str(item.get("name") or "Speaker"),
                        embedding=[float(x) for x in emb],
                        backend=str(item.get("backend") or "sface"),
                        thumb_b64=str(item.get("thumb_b64") or ""),
                        created_at=float(item.get("created_at") or time.time()),
                        baseline=SpeakerBaseline.from_dict(item.get("baseline")),
                    )
                active_id = raw.get("active_profile_id")
                if active_id and active_id in self.profiles:
                    self.activate_profile(active_id)
        except Exception as e:
            print(f"[SpeakerID] Ignoring unreadable {SPEAKERS_FILE}: {e}")
