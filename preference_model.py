"""Next-Gen User Preference & Visual Taste Learning Model (ML-1 .. ML-6).

Upgrades over v4:
  - `FEATURE_VERSION = 5` (18 normalized geometric/gaze/velocity/blur/colour features).
  - Non-linear ensemble (`LogisticRegression` + `HistGradientBoostingClassifier`) so
    U-shaped sweet spots (e.g. open-mouth laugh vs. open-mouth mid-sentence) are learned.
  - Visual Embedding Taste Memory (`ML-1` via `visual_embedder.py`): learns pixel-level
    visual flaws and aesthetic preferences directly from speaker crops.
  - Pairwise Session Ranking & Implicit Feedback (`ML-2`): learns automatically when an
    operator downloads a single capture, pins a winner, swaps a burst frame, or deletes a card.
  - "Bootstrap from Past Winners" folder trainer (`ML-3`).
  - AI Judge -> Fast ML Teacher-Student Distillation (`ML-4`) with soft weights (`0.35x`).
  - Transparent "ML Brain" Inspector (`ML-6`) with feature drivers, 2,000-sample history,
    and 1-click Undo / Flip rating controls.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import QualityThresholds
from visual_embedder import get_visual_embedder


class UserPreferenceModel:
    """Hybrid Geometric + Visual + Pairwise Preference Learner."""

    _DEFAULT_JSON_DATA_PATH = os.path.join(
        os.environ.get("TG_DATA_DIR", "").strip() or os.path.join(os.getcwd(), "models"),
        "user_preference_model.json",
    )
    JSON_DATA_PATH = _DEFAULT_JSON_DATA_PATH

    def _get_json_path(self) -> str:
        if self.JSON_DATA_PATH != self._DEFAULT_JSON_DATA_PATH:
            return self.JSON_DATA_PATH
        env_dir = os.environ.get("TG_DATA_DIR", "").strip()
        if env_dir:
            return os.path.join(env_dir, "user_preference_model.json")
        return self.JSON_DATA_PATH

    # v5: 18 geometric features (adding head_roll_norm, eye_gaze_away, mouth_velocity,
    # hand_sharpness_ratio, color_cast_clean) + visual taste prototypes + pairwise diffs.
    FEATURE_VERSION = 5

    MIN_SAMPLES_TO_TRAIN = 12
    MIN_PER_CLASS = 3
    MAX_STORED_SAMPLES = 2000

    FEATURE_NAMES = [
        "head_yaw_norm",         # 0: Head Turn (0=facing camera)
        "head_pitch_norm",       # 1: Head Pitch (0=level)
        "head_roll_norm",        # 2: Head Roll (0=upright)
        "eye_openness",          # 3: Eye Openness (EAR)
        "eye_gaze_away",         # 4: Conjugate horizontal gaze away (0=lens contact)
        "smile_ratio",           # 5: Smile Ratio
        "mouth_openness_mar",    # 6: Mouth Openness (MAR)
        "mouth_velocity",        # 7: Frame-to-frame mouth motion (0=paused)
        "shoulder_tilt_norm",    # 8: Shoulder Tilt (0=level)
        "sharpness_norm",        # 9: Face Laplacian Sharpness
        "hand_sharpness_ratio",  # 10: Hand vs Face sharpness (1=crisp hands)
        "exposure_score",        # 11: Exposure/Lighting
        "color_cast_clean",      # 12: Clean skin tone vs LED colour wash
        "cutout_score",          # 13: Cutout suitability
        "torso_clear",           # 14: 1.0 = nothing crossing the body
        "hands_clear",           # 15: 1.0 = hands clear of chest
        "background_score",      # 16: Plain backdrop score
        "subject_height",        # 17: Subject height fraction
    ]

    FEATURE_LABELS = {
        "head_yaw_norm": "Head Turn Away",
        "head_pitch_norm": "Head Tilt / Bow",
        "head_roll_norm": "Head Roll",
        "eye_openness": "Wide-Open Eyes",
        "eye_gaze_away": "Side-Eye Gaze",
        "smile_ratio": "Warm Smile",
        "mouth_openness_mar": "Open Mouth",
        "mouth_velocity": "Mid-Syllable Motion",
        "shoulder_tilt_norm": "Shoulder Slant",
        "sharpness_norm": "Crisp Face Focus",
        "hand_sharpness_ratio": "Sharp Gesturing Hands",
        "exposure_score": "Balanced Exposure",
        "color_cast_clean": "Natural Skin Tone",
        "cutout_score": "Clean Cutout Framing",
        "torso_clear": "Unobstructed Torso",
        "hands_clear": "Open Hand Posture",
        "background_score": "Clean Stage Backdrop",
        "subject_height": "Subject Fill Size",
    }

    DEFAULT_PENALTIES: Dict[str, float] = {
        "head_down": 1.0,
        "eyes_closed": 1.0,
        "mouth_open": 1.0,
        "hand_obstruction": 1.0,
        "prop_strictness": 1.0,
        "blur": 1.0,
    }

    def __init__(self) -> None:
        self.feature_count = len(self.FEATURE_NAMES)
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced")
        self.gb_clf: Optional[HistGradientBoostingClassifier] = None
        self.pair_clf: Optional[LogisticRegression] = None

        # Backward-compatible training dataset: list of (feature_list, label) or (feature_list, label, weight)
        self.samples: List[Tuple[Any, ...]] = []
        self.sample_meta: List[Dict[str, Any]] = []
        self.pairwise_diffs: List[Tuple[List[float], float]] = []
        self.visual_samples: List[Tuple[List[float], int, float]] = []
        self.feedback_notes: List[Dict[str, Any]] = []
        self.learned_penalties: Dict[str, float] = dict(self.DEFAULT_PENALTIES)

        self.is_trained: bool = False
        self.confidence: float = 0.0
        self._pos_visual_proto: Optional[np.ndarray] = None
        self._neg_visual_proto: Optional[np.ndarray] = None

        self.load()

    @staticmethod
    def extract_features(analysis: Any, thresholds: Optional[QualityThresholds] = None) -> np.ndarray:
        """Extracts a normalized 18-dimensional v5 feature vector from an AnalysisResult or SingleFaceAnalysis."""
        t = thresholds or QualityThresholds()
        yaw_norm = min(1.0, abs(float(getattr(analysis, "yaw_deg", 0.0) or 0.0)) / max(1.0, t.max_yaw_deg))
        pitch_norm = min(1.0, abs(float(getattr(analysis, "pitch_deg", 0.0) or 0.0)) / max(1.0, t.max_pitch_deg))
        roll_norm = min(1.0, abs(float(getattr(analysis, "roll_deg", 0.0) or 0.0)) / max(1.0, t.max_roll_deg))
        eye_open = min(1.0, max(0.0, float(getattr(analysis, "avg_ear", 0.35) or 0.35) / 0.40))
        gaze_away = min(1.0, max(0.0, float(getattr(analysis, "eye_gaze_away_score", 0.0) or 0.0)))
        smile = min(1.0, max(0.0, float(getattr(analysis, "smile_ratio", 0.5) or 0.0) / 1.0))
        mar = min(1.0, max(0.0, float(getattr(analysis, "mar", 0.2) or 0.0)))
        mouth_vel = min(1.0, max(0.0, float(getattr(analysis, "mouth_velocity", 0.0) or 0.0) / 0.35))
        shoulder_tilt_norm = min(1.0, max(0.0, float(getattr(analysis, "shoulder_tilt_deg", 0.0) or 0.0) / 15.0))

        floor, good = t.sharpness_floor, max(t.sharpness_floor + 1.0, t.sharpness_good)
        lap = float(getattr(analysis, "laplacian_variance", floor) or floor)
        sharp = min(1.0, max(0.0, (lap - floor) / (good - floor)))

        hand_lap = float(getattr(analysis, "hand_laplacian_variance", 0.0) or 0.0)
        if hand_lap <= 0.0 or lap <= 1.0:
            hand_sharp_ratio = 0.25 if getattr(analysis, "is_hand_blurry", False) else 1.0
        else:
            hand_sharp_ratio = min(1.0, max(0.0, hand_lap / max(1.0, lap)))

        exposure = min(1.0, max(0.0, float(getattr(analysis, "exposure_score", 1.0) or 0.0)))
        color_clean = float(getattr(analysis, "color_cast_score", exposure) or exposure)
        color_clean = min(1.0, max(0.0, color_clean))
        cutout = min(1.0, max(0.0, float(getattr(analysis, "cutout_score", 0.5) or 0.0)))
        torso_clear = min(1.0, max(0.0, float(getattr(analysis, "torso_clear_score", 1.0) or 0.0)))
        hands_clear = 0.0 if getattr(analysis, "hands_on_torso", False) else 1.0
        background = min(1.0, max(0.0, float(getattr(analysis, "background_score", 1.0) or 0.0)))
        subject_h = min(1.0, max(0.0, float(getattr(analysis, "subject_height_ratio", 0.0) or 0.0)))

        return np.array([
            yaw_norm, pitch_norm, roll_norm, eye_open, gaze_away,
            smile, mar, mouth_vel, shoulder_tilt_norm, sharp,
            hand_sharp_ratio, exposure, color_clean, cutout,
            torso_clear, hands_clear, background, subject_h,
        ], dtype=np.float64)

    def _unpack_samples(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        feats: List[List[float]] = []
        labels: List[int] = []
        weights: List[float] = []
        for item in self.samples:
            if len(item) >= 3:
                f, lbl, w = item[0], item[1], float(item[2])
            else:
                f, lbl, w = item[0], item[1], 1.0
            if len(f) == self.feature_count:
                feats.append(list(f))
                labels.append(int(lbl))
                weights.append(max(0.05, w))
        if not feats:
            return np.empty((0, self.feature_count)), np.empty((0,), dtype=np.int32), np.empty((0,))
        return (
            np.array(feats, dtype=np.float64),
            np.array(labels, dtype=np.int32),
            np.array(weights, dtype=np.float64),
        )

    def _predict_visual(self, crop_bgr: Optional[np.ndarray]) -> Optional[float]:
        if crop_bgr is None or self._pos_visual_proto is None or self._neg_visual_proto is None:
            return None
        try:
            vec = get_visual_embedder().embed(crop_bgr)
            if vec.shape != self._pos_visual_proto.shape or float(np.linalg.norm(vec)) <= 1e-6:
                return None
            sim_pos = float(np.dot(vec, self._pos_visual_proto))
            sim_neg = float(np.dot(vec, self._neg_visual_proto))
            # Map cosine margin (-0.4 .. +0.4) smoothly onto (0.05 .. 0.95)
            margin = (sim_pos - sim_neg) * 4.5
            prob = 1.0 / (1.0 + float(np.exp(-margin)))
            return float(np.clip(prob, 0.05, 0.95))
        except Exception:
            return None

    def predict(self, features: np.ndarray, crop_bgr: Optional[np.ndarray] = None) -> float:
        """Predicts preference probability score (0.0 to 1.0) fusing geometry, pairwise, & visual taste."""
        if not self.is_trained:
            return 0.5

        try:
            x_row = np.asarray(features, dtype=np.float64).reshape(1, -1)
            X_scaled = self.scaler.transform(x_row)
            probs = self.clf.predict_proba(X_scaled)[0]
            p_linear = float(probs[1]) if len(probs) > 1 else float(probs[0])

            if self.gb_clf is not None:
                gb_probs = self.gb_clf.predict_proba(x_row)[0]
                p_gb = float(gb_probs[1]) if len(gb_probs) > 1 else p_linear
                p_geom = 0.50 * p_linear + 0.50 * p_gb
            else:
                p_geom = p_linear

            if self.pair_clf is not None and len(self.pairwise_diffs) >= 4:
                p_pair = float(self.pair_clf.predict_proba(X_scaled)[0][1])
                p_geom = 0.78 * p_geom + 0.22 * p_pair

            p_vis = self._predict_visual(crop_bgr)
            if p_vis is not None and len(self.visual_samples) >= 6:
                vis_w = min(0.30, len(self.visual_samples) / 80.0)
                blended = (1.0 - vis_w) * p_geom + vis_w * p_vis
            else:
                blended = p_geom

            return float(round(max(0.0, min(1.0, blended)), 3))
        except Exception:
            return 0.5

    def add_sample(
        self,
        analysis: Any,
        is_favorite: bool,
        comment: str = "",
        reasons: Optional[List[str]] = None,
        crop_bgr: Optional[np.ndarray] = None,
        source: str = "human",
        weight: float = 1.0,
        filename: str = "",
    ) -> str:
        """Adds a labelled preference sample, optionally with crop pixels & structured reasons."""
        features = self.extract_features(analysis).tolist()
        label = 1 if is_favorite else 0
        eff_weight = round(max(0.05, min(2.0, float(weight))), 3)
        sample_id = uuid.uuid4().hex[:10]

        self.samples.append((features, label, eff_weight))
        if len(self.samples) > self.MAX_STORED_SAMPLES:
            self.samples = self.samples[-self.MAX_STORED_SAMPLES:]

        clean_comment = (comment or "").strip()
        clean_reasons = [str(r).strip() for r in (reasons or []) if str(r).strip()]

        self.sample_meta.append({
            "id": sample_id,
            "timestamp": round(time.time(), 1),
            "favorite": bool(is_favorite),
            "label": label,
            "source": source,
            "weight": eff_weight,
            "filename": os.path.basename(filename) if filename else "",
            "reasons": clean_reasons,
            "comment": clean_comment,
            "features": [round(float(v), 4) for v in features],
        })
        self.sample_meta = self.sample_meta[-self.MAX_STORED_SAMPLES:]

        if crop_bgr is not None and crop_bgr.size > 0:
            try:
                vis_vec = get_visual_embedder().embed(crop_bgr)
                if float(np.linalg.norm(vis_vec)) > 1e-6:
                    self.visual_samples.append((vis_vec.tolist(), label, eff_weight))
                    self.visual_samples = self.visual_samples[-600:]
            except Exception:
                pass

        if clean_comment or clean_reasons:
            self.feedback_notes.append({
                "timestamp": round(time.time(), 1),
                "favorite": bool(is_favorite),
                "reasons": clean_reasons,
                "comment": clean_comment,
                "metrics": {
                    "yaw_deg": round(float(getattr(analysis, "yaw_deg", 0.0)), 1),
                    "pitch_deg": round(float(getattr(analysis, "pitch_deg", 0.0)), 1),
                    "avg_ear": round(float(getattr(analysis, "avg_ear", 0.35)), 2),
                    "mar": round(float(getattr(analysis, "mar", 0.2)), 2),
                    "is_prop_moment": bool(getattr(analysis, "is_prop_moment", False)),
                },
            })
            self.feedback_notes = self.feedback_notes[-80:]
            self._apply_reason_learning(features, is_favorite, clean_comment, clean_reasons)

        self.fit()
        self.save()
        return sample_id

    def add_pairwise_preference(
        self,
        winner_analysis: Any,
        loser_analysis: Any,
        weight: float = 0.7,
    ) -> None:
        """Records that `winner` was preferred over `loser` in the same session (ML-2)."""
        w_feat = self.extract_features(winner_analysis)
        l_feat = self.extract_features(loser_analysis)
        diff = (w_feat - l_feat).tolist()
        if float(np.linalg.norm(diff)) <= 1e-5:
            return
        self.pairwise_diffs.append(([round(float(v), 4) for v in diff], float(weight)))
        self.pairwise_diffs = self.pairwise_diffs[-800:]

    def record_implicit_feedback(
        self,
        target_capture: Any,
        peer_captures: Optional[List[Any]] = None,
        action: str = "download",
        crop_bgr: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Implicit preference signal from normal UI usage (download, pin, burst_swap, delete)."""
        analysis = getattr(target_capture, "analysis", None)
        if analysis is None and hasattr(target_capture, "overall_score"):
            analysis = target_capture
        if analysis is None:
            return None
        fname = getattr(target_capture, "filepath", "") or ""
        act = (action or "").strip().lower()

        if act == "delete":
            return self.add_sample(
                analysis, is_favorite=False, source="implicit_delete",
                weight=0.55, crop_bgr=crop_bgr, filename=fname,
            )

        # Positive implicit signal ("download", "pin", "burst_swap")
        w_map = {"pin": 0.80, "download": 0.65, "burst_swap": 0.85}
        eff_w = w_map.get(act, 0.60)
        for peer in (peer_captures or [])[:6]:
            if peer is target_capture:
                continue
            p_analysis = getattr(peer, "analysis", None)
            if p_analysis is not None and not getattr(peer, "is_pinned", False):
                self.add_pairwise_preference(analysis, p_analysis, weight=eff_w * 0.8)

        return self.add_sample(
            analysis, is_favorite=True, source=f"implicit_{act}",
            weight=eff_w, crop_bgr=crop_bgr, filename=fname,
        )

    def record_judge_distillation(
        self,
        analysis: Any,
        verdict: Any,
        crop_bgr: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Teacher-Student Distillation (ML-4): feeds high-confidence AI Judge verdicts
        into the fast 8-FPS local ML scorer with soft weight (`0.35x`)."""
        if analysis is None or verdict is None:
            return None
        score = float(getattr(verdict, "score", 0.5) or 0.5)
        usable = bool(getattr(verdict, "usable", True))
        if usable and score >= 0.85:
            is_fav = True
        elif (not usable) or score <= 0.32:
            is_fav = False
        else:
            return None  # Middling verdicts are not informative enough to distill

        # Cap distilled samples so human feedback always dominates
        distilled_ids = [m["id"] for m in self.sample_meta if m.get("source") == "ai_judge"]
        if len(distilled_ids) >= 120:
            self.undo_rating(distilled_ids[0], refit=False)

        return self.add_sample(
            analysis,
            is_favorite=is_fav,
            source="ai_judge",
            weight=0.35,
            crop_bgr=crop_bgr,
        )

    def bootstrap_from_folder(
        self,
        folder_path: str,
        analyzer: Any,
        max_images: int = 80,
    ) -> Dict[str, Any]:
        """Pre-trains the model from an existing folder of past winning thumbnails (ML-3)."""
        if not folder_path or not os.path.isdir(folder_path):
            return {"ok": False, "error": f"Folder not found: {folder_path}"}

        exts = (".png", ".jpg", ".jpeg", ".webp")
        files = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path))
            if f.lower().endswith(exts) and not f.lower().endswith("_cutout.png")
        ][:max_images]
        if not files:
            return {"ok": False, "error": "No image files (.png/.jpg) found in that folder."}

        added_pos = 0
        pos_vectors: List[List[float]] = []
        for fpath in files:
            img = cv2.imread(fpath)
            if img is None or img.size == 0:
                continue
            res = analyzer.analyze_frame(img)
            if not getattr(res, "face_detected", False):
                continue
            faces = getattr(res, "faces", None) or []
            primary = faces[0] if faces else res
            feat = self.extract_features(primary, getattr(analyzer, "thresholds", None)).tolist()
            pos_vectors.append(feat)

            # Extract subject/face crop for visual embedding memory
            bbox = getattr(primary, "subject_bbox", None) or getattr(primary, "bbox", None) or getattr(res, "face_bbox", None)
            crop = img
            if bbox and bbox[2] > 16 and bbox[3] > 16:
                x, y, w, h = bbox
                sub = img[max(0, y):min(img.shape[0], y + h), max(0, x):min(img.shape[1], x + w)]
                if sub.size > 0:
                    crop = sub

            self.samples.append((feat, 1, 0.90))
            self.sample_meta.append({
                "id": uuid.uuid4().hex[:10],
                "timestamp": round(time.time(), 1),
                "favorite": True,
                "label": 1,
                "source": "bootstrap",
                "weight": 0.90,
                "filename": os.path.basename(fpath),
                "reasons": [],
                "comment": "Bootstrapped from folder",
                "features": [round(float(v), 4) for v in feat],
            })
            try:
                vis_vec = get_visual_embedder().embed(crop)
                if float(np.linalg.norm(vis_vec)) > 1e-6:
                    self.visual_samples.append((vis_vec.tolist(), 1, 0.90))
            except Exception:
                pass
            added_pos += 1

        if added_pos == 0:
            return {"ok": False, "error": "Found images, but none contained a detectable speaker face."}

        # Ensure sufficient hard-negative contrast prototypes exist so a fresh install can fit immediately
        _, existing_y, _ = self._unpack_samples()
        neg_count = int(np.sum(existing_y == 0)) if existing_y.size else 0
        needed_neg = max(0, max(self.MIN_PER_CLASS + 3, added_pos // 2) - neg_count)
        synth_neg = 0
        if needed_neg > 0 and pos_vectors:
            archetypes = [
                # Blinking + bowed head
                {1: 0.90, 3: 0.12, 5: 0.15},
                # Wide open mid-syllable talking mouth + high velocity
                {5: 0.10, 6: 0.78, 7: 0.85},
                # Heavy motion blur + smeared hand
                {9: 0.08, 10: 0.18},
                # Turned away + side-eye + obstructed torso
                {0: 0.92, 4: 0.85, 14: 0.25, 15: 0.0},
            ]
            for i in range(needed_neg):
                base = list(pos_vectors[i % len(pos_vectors)])
                patch = archetypes[i % len(archetypes)]
                for idx_k, val_v in patch.items():
                    base[idx_k] = val_v
                self.samples.append((base, 0, 0.65))
                self.sample_meta.append({
                    "id": uuid.uuid4().hex[:10],
                    "timestamp": round(time.time(), 1),
                    "favorite": False,
                    "label": 0,
                    "source": "bootstrap_contrast",
                    "weight": 0.65,
                    "filename": "",
                    "reasons": ["auto hard-negative"],
                    "comment": "",
                    "features": [round(float(v), 4) for v in base],
                })
                synth_neg += 1

        self.fit()
        self.save()
        return {
            "ok": True,
            "added_positives": added_pos,
            "synthetic_negatives": synth_neg,
            "is_trained": self.is_trained,
            "confidence": self.confidence,
            "trust": round(self.trust, 3),
            "sample_count": len(self.samples),
        }

    def undo_rating(self, sample_id: str, refit: bool = True) -> bool:
        """Removes a specific rating by `id` from `sample_meta` & `samples` (ML-6)."""
        idx = next((i for i, m in enumerate(self.sample_meta) if m.get("id") == sample_id), None)
        if idx is None:
            return False
        meta = self.sample_meta.pop(idx)
        target_feat = meta.get("features")
        target_lbl = meta.get("label")
        for s_i in range(len(self.samples) - 1, -1, -1):
            s = self.samples[s_i]
            if int(s[1]) == target_lbl:
                if target_feat and len(s[0]) == len(target_feat):
                    if np.allclose(s[0], target_feat, atol=1e-3):
                        self.samples.pop(s_i)
                        break
        if refit:
            self.fit()
            self.save()
        return True

    def flip_rating(self, sample_id: str) -> bool:
        """Flips a misclicked rating (Favorite <-> Reject) in place and refits (ML-6)."""
        meta = next((m for m in self.sample_meta if m.get("id") == sample_id), None)
        if meta is None:
            return False
        old_lbl = int(meta.get("label", 0))
        new_lbl = 0 if old_lbl == 1 else 1
        target_feat = meta.get("features")
        for s_i in range(len(self.samples) - 1, -1, -1):
            s = self.samples[s_i]
            if int(s[1]) == old_lbl and target_feat and len(s[0]) == len(target_feat):
                if np.allclose(s[0], target_feat, atol=1e-3):
                    w = float(s[2]) if len(s) >= 3 else 1.0
                    self.samples[s_i] = (list(s[0]), new_lbl, w)
                    break
        meta["label"] = new_lbl
        meta["favorite"] = bool(new_lbl == 1)
        self.fit()
        self.save()
        return True

    def reset_all(self) -> None:
        self.samples.clear()
        self.sample_meta.clear()
        self.pairwise_diffs.clear()
        self.visual_samples.clear()
        self.feedback_notes.clear()
        self.learned_penalties = dict(self.DEFAULT_PENALTIES)
        self.is_trained = False
        self.confidence = 0.0
        self._pos_visual_proto = None
        self._neg_visual_proto = None
        self.save()

    def _apply_reason_learning(
        self,
        base_features: List[float],
        is_favorite: bool,
        comment: str,
        reasons: List[str],
    ) -> None:
        combined = (" ".join(reasons) + " " + comment).lower()
        if not is_favorite:
            synth = list(base_features)
            matched_any = False

            if any(k in combined for k in ("head", "bow", "down", "chin", "gaze", "looking", "floor", "notes")):
                self.learned_penalties["head_down"] = min(1.35, round(self.learned_penalties.get("head_down", 1.0) + 0.08, 2))
                synth[1] = max(synth[1], 0.85)   # head_pitch_norm
                synth[3] = min(synth[3], 0.60)   # eye_openness
                matched_any = True

            if any(k in combined for k in ("eye", "blink", "closed", "squint", "glasses")):
                self.learned_penalties["eyes_closed"] = min(1.35, round(self.learned_penalties.get("eyes_closed", 1.0) + 0.08, 2))
                synth[3] = min(synth[3], 0.50)   # eye_openness
                matched_any = True

            if any(k in combined for k in ("mouth", "talk", "word", "open", "grimace", "awkward", "expression")):
                self.learned_penalties["mouth_open"] = min(1.35, round(self.learned_penalties.get("mouth_open", 1.0) + 0.08, 2))
                synth[6] = max(synth[6], 0.65)   # mouth_openness_mar
                synth[5] = min(synth[5], 0.35)   # smile_ratio
                matched_any = True

            if any(k in combined for k in ("mic", "hand", "block", "cover", "obstruct", "chest", "stand", "ipad")):
                self.learned_penalties["hand_obstruction"] = min(1.35, round(self.learned_penalties.get("hand_obstruction", 1.0) + 0.08, 2))
                synth[14] = min(synth[14], 0.35)  # torso_clear
                synth[15] = 0.0                   # hands_clear
                matched_any = True

            if any(k in combined for k in ("prop", "box", "holding", "cut off", "clipped")):
                self.learned_penalties["prop_strictness"] = min(1.35, round(self.learned_penalties.get("prop_strictness", 1.0) + 0.10, 2))
                synth[13] = min(synth[13], 0.45)  # cutout_score
                matched_any = True

            if any(k in combined for k in ("blur", "soft", "focus", "smear", "motion")):
                self.learned_penalties["blur"] = min(1.35, round(self.learned_penalties.get("blur", 1.0) + 0.08, 2))
                synth[9] = min(synth[9], 0.25)    # sharpness_norm
                synth[10] = min(synth[10], 0.30)  # hand_sharpness_ratio
                matched_any = True

            if matched_any:
                self.samples.append((synth, 0, 1.0))
        else:
            for k in list(self.learned_penalties.keys()):
                if self.learned_penalties[k] > 1.0:
                    self.learned_penalties[k] = max(1.0, round(self.learned_penalties[k] - 0.05, 2))

    def get_judge_feedback_prompt(self, max_notes: int = 8) -> str:
        if not self.feedback_notes:
            return ""
        bad_lines: List[str] = []
        good_lines: List[str] = []
        for entry in reversed(self.feedback_notes):
            parts = []
            if entry.get("reasons"):
                parts.append(", ".join(entry["reasons"]))
            if entry.get("comment"):
                parts.append(entry["comment"])
            text = " — ".join(parts).strip()
            if not text:
                continue
            if entry.get("favorite"):
                if text not in good_lines and len(good_lines) < 3:
                    good_lines.append(text)
            else:
                if text not in bad_lines and len(bad_lines) < max_notes:
                    bad_lines.append(text)

        if not bad_lines and not good_lines:
            return ""

        out = ["\nOPERATOR FEEDBACK FROM PREVIOUS CAPTURES (follow these strictly):"]
        if bad_lines:
            out.append("The operator explicitly REJECTED photos for these reasons — penalize heavily (score <= 0.35, or usable=false if severe) if you see any of these:")
            for b in bad_lines:
                out.append(f"  - REJECT IF: {b}")
        if good_lines:
            out.append("The operator praised photos with these qualities:")
            for g in good_lines:
                out.append(f"  - PREFER: {g}")
        return "\n".join(out)

    def _fit_visual_prototypes(self) -> None:
        self._pos_visual_proto = None
        self._neg_visual_proto = None
        if len(self.visual_samples) < 4:
            return
        target_dim = len(self.visual_samples[-1][0])
        pos_vecs = [np.asarray(v, dtype=np.float32) * float(w) for v, lbl, w in self.visual_samples if lbl == 1 and len(v) == target_dim]
        neg_vecs = [np.asarray(v, dtype=np.float32) * float(w) for v, lbl, w in self.visual_samples if lbl == 0 and len(v) == target_dim]
        if len(pos_vecs) >= 2 and len(neg_vecs) >= 2:
            p_sum = np.sum(pos_vecs, axis=0)
            n_sum = np.sum(neg_vecs, axis=0)
            pn, nn = float(np.linalg.norm(p_sum)), float(np.linalg.norm(n_sum))
            if pn > 1e-6 and nn > 1e-6:
                self._pos_visual_proto = (p_sum / pn).astype(np.float32)
                self._neg_visual_proto = (n_sum / nn).astype(np.float32)

    def _fit_pairwise(self) -> None:
        self.pair_clf = None
        valid = [
            (np.asarray(d, dtype=np.float64), float(w))
            for d, w in self.pairwise_diffs
            if len(d) == self.feature_count
        ]
        if len(valid) < 4 or not hasattr(self.scaler, "scale_"):
            return
        try:
            scale = np.maximum(1e-6, self.scaler.scale_)
            X_p, y_p, w_p = [], [], []
            for diff, w in valid:
                sd = diff / scale
                X_p.append(sd)
                y_p.append(1)
                w_p.append(w)
                X_p.append(-sd)
                y_p.append(0)
                w_p.append(w)
            p_clf = LogisticRegression(C=0.8, fit_intercept=False, solver="liblinear")
            p_clf.fit(np.array(X_p), np.array(y_p), sample_weight=np.array(w_p))
            self.pair_clf = p_clf
        except Exception:
            self.pair_clf = None

    def fit(self) -> None:
        """Fits the geometric linear + gradient-boosted ensemble, pairwise ranker, and visual prototypes."""
        self.is_trained = False
        self.confidence = 0.0
        self.gb_clf = None
        self.pair_clf = None

        X, y, sample_weights = self._unpack_samples()
        if len(X) < self.MIN_SAMPLES_TO_TRAIN:
            return

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or counts.min() < self.MIN_PER_CLASS:
            return

        try:
            self.scaler = StandardScaler()
            self.clf = LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced")
            X_scaled = self.scaler.fit_transform(X)
            self.clf.fit(X_scaled, y, sample_weight=sample_weights)

            # Fit non-linear HistGradientBoosting when we have enough rows for leaf splits
            if len(X) >= 24 and counts.min() >= 5:
                try:
                    gb = HistGradientBoostingClassifier(
                        max_depth=3,
                        max_iter=45,
                        min_samples_leaf=max(3, min(8, int(counts.min() // 2))),
                        l2_regularization=2.0,
                        class_weight="balanced",
                        random_state=0,
                    )
                    gb.fit(X, y, sample_weight=sample_weights)
                    self.gb_clf = gb
                except Exception:
                    self.gb_clf = None

            self._fit_pairwise()
            self._fit_visual_prototypes()
            self.is_trained = True
            self.confidence = self._cross_val_accuracy(X, y)
        except Exception as e:
            print(f"[UserPreferenceModel] Scikit-Learn fit error: {e}")

    @property
    def trust(self) -> float:
        """How far past guessing the model actually is, on 0..1.

        Requires clearing a 0.55 balanced-accuracy noise margin so pure label skew or
        small-sample noise earns 0.0 influence.
        """
        if not self.is_trained or self.confidence <= 0.55:
            return 0.0
        return max(0.0, min(1.0, (self.confidence - 0.55) / 0.45))

    def _cross_val_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        try:
            n_splits = int(min(5, np.bincount(y).min()))
            if n_splits < 2:
                return 0.0
            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced"),
            )
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy")
            return float(round(scores.mean(), 4))
        except Exception:
            return 0.0

    def get_brain_summary(self) -> Dict[str, Any]:
        """Structured transparency payload for the Web UI 'ML Brain' Inspector (ML-6)."""
        X, y, _ = self._unpack_samples()
        likes = int(np.sum(y == 1)) if y.size else 0
        rejects = int(np.sum(y == 0)) if y.size else 0

        drivers: List[Dict[str, Any]] = []
        if self.is_trained and hasattr(self.clf, "coef_") and self.clf.coef_ is not None:
            coefs = self.clf.coef_[0]
            max_abs = float(np.max(np.abs(coefs))) if coefs.size else 1.0
            max_abs = max(1e-6, max_abs)
            for idx_f, name in enumerate(self.FEATURE_NAMES):
                if idx_f >= len(coefs):
                    continue
                raw_w = float(coefs[idx_f])
                drivers.append({
                    "feature": name,
                    "label": self.FEATURE_LABELS.get(name, name),
                    "weight": round(raw_w, 3),
                    "strength_pct": int(round((abs(raw_w) / max_abs) * 100)),
                    "rel_strength": round(abs(raw_w) / max_abs, 3),
                    "direction": "prefers_higher" if raw_w >= 0 else "prefers_lower",
                })
            drivers.sort(key=lambda d: abs(d["weight"]), reverse=True)

        by_source: Dict[str, int] = {}
        for m in self.sample_meta:
            src = str(m.get("source") or "human")
            by_source[src] = by_source.get(src, 0) + 1

        recent_history = [
            {
                "id": m.get("id", ""),
                "timestamp": m.get("timestamp", 0),
                "favorite": bool(m.get("favorite", False)),
                "source": m.get("source", "human"),
                "filename": m.get("filename", ""),
                "reasons": m.get("reasons", []),
                "comment": m.get("comment", ""),
            }
            for m in reversed(self.sample_meta[-40:])
        ]

        return {
            "feature_version": self.FEATURE_VERSION,
            "is_trained": self.is_trained,
            "nonlinear_active": self.gb_clf is not None,
            "ensemble_active": self.gb_clf is not None,
            "pairwise_active": self.pair_clf is not None,
            "visual_active": self._pos_visual_proto is not None,
            "confidence": round(self.confidence, 4),
            "trust": round(self.trust, 3),
            "sample_count": int(len(X)),
            "likes_count": likes,
            "rejects_count": rejects,
            "likes": likes,
            "rejects": rejects,
            "pairwise_count": len(self.pairwise_diffs),
            "visual_sample_count": len(self.visual_samples),
            "sources": by_source,
            "learned_penalties": dict(self.learned_penalties),
            "top_drivers": drivers[:10],
            "recent_history": recent_history,
            "recent_ratings": recent_history,
            "visual_embedder": get_visual_embedder().status(),
            "visual_backbone": get_visual_embedder().status(),
        }

    def save(self) -> None:
        json_path = self._get_json_path()
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        try:
            data = {
                "feature_version": self.FEATURE_VERSION,
                "is_trained": self.is_trained,
                "confidence": self.confidence,
                "confidence_metric": "cross_val_balanced_accuracy",
                "trust": self.trust,
                "sample_count": len(self.samples),
                "samples": self.samples[-self.MAX_STORED_SAMPLES:],
                "sample_meta": self.sample_meta[-self.MAX_STORED_SAMPLES:],
                "pairwise_diffs": self.pairwise_diffs[-800:],
                "visual_samples": self.visual_samples[-600:],
                "feedback_notes": self.feedback_notes[-80:],
                "learned_penalties": self.learned_penalties,
            }
            tmp = f"{json_path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, json_path)
        except Exception as e:
            print(f"[UserPreferenceModel] Error saving preference data: {e}")

    def load(self) -> None:
        json_path = self._get_json_path()
        if not os.path.exists(json_path):
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[UserPreferenceModel] JSON data load error: {e}")
            return

        if isinstance(data.get("feedback_notes"), list):
            self.feedback_notes = data["feedback_notes"][-80:]
        if isinstance(data.get("learned_penalties"), dict):
            for k in self.DEFAULT_PENALTIES:
                if k in data["learned_penalties"]:
                    try:
                        val = float(data["learned_penalties"][k])
                        self.learned_penalties[k] = round(max(0.8, min(1.35, val)), 2)
                    except (TypeError, ValueError):
                        pass

        stored_version = int(data.get("feature_version", 1))
        raw_samples = data.get("samples", [])

        if stored_version != self.FEATURE_VERSION:
            if raw_samples:
                print(
                    f"[UserPreferenceModel] Discarding {len(raw_samples)} rating(s) recorded "
                    f"under feature set v{stored_version}; upgraded to v{self.FEATURE_VERSION}."
                )
            self.samples = []
            self.sample_meta = []
            self.pairwise_diffs = []
            self.visual_samples = list(data.get("visual_samples") or [])[-600:]
            self.is_trained = False
            self.confidence = 0.0
            self.save()
            return

        parsed_samples: List[Tuple[Any, ...]] = []
        for s in raw_samples:
            if isinstance(s, (list, tuple)) and len(s) >= 2 and len(s[0]) == self.feature_count:
                w = float(s[2]) if len(s) >= 3 else 1.0
                parsed_samples.append((list(s[0]), int(s[1]), w))
        self.samples = parsed_samples
        self.sample_meta = list(data.get("sample_meta") or [])[-self.MAX_STORED_SAMPLES:]
        self.pairwise_diffs = [
            (list(p[0]), float(p[1]))
            for p in (data.get("pairwise_diffs") or [])
            if isinstance(p, (list, tuple)) and len(p) >= 2 and len(p[0]) == self.feature_count
        ][-800:]
        self.visual_samples = [
            (list(v[0]), int(v[1]), float(v[2]) if len(v) >= 3 else 1.0)
            for v in (data.get("visual_samples") or [])
            if isinstance(v, (list, tuple)) and len(v) >= 2
        ][-600:]

        self.fit()
