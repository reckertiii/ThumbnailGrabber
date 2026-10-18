import os
import json
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

from config import QualityThresholds

class UserPreferenceModel:
    """Scikit-Learn ML Model (Logistic Regression with StandardScaler)
    that learns user visual preference for thumbnail selection.
    
    Extracts normalized 13-D feature vectors from AnalysisResult objects and
    trains on user feedback (⭐ Favorite = 1, 👎 Rejected = 0).
    Labelled samples persist to models/user_preference_model.json; the estimator is
    refit from them on load.
    """
    
    # User ratings, unlike the downloaded weights, must persist across container
    # restarts — so this stays on the mounted data directory.
    _DEFAULT_JSON_DATA_PATH = os.path.join(
        os.environ.get("TG_DATA_DIR", "").strip() or os.path.join(os.getcwd(), "models"),
        "user_preference_model.json"
    )
    JSON_DATA_PATH = _DEFAULT_JSON_DATA_PATH

    def _get_json_path(self) -> str:
        # Respect test overrides of JSON_DATA_PATH while still honoring runtime TG_DATA_DIR changes.
        if self.JSON_DATA_PATH != self._DEFAULT_JSON_DATA_PATH:
            return self.JSON_DATA_PATH
        env_dir = os.environ.get("TG_DATA_DIR", "").strip()
        if env_dir:
            return os.path.join(env_dir, "user_preference_model.json")
        return self.JSON_DATA_PATH

    # Bump whenever the meaning or scale of any feature changes. Samples recorded under
    # an older version describe different quantities, so mixing them in would train the
    # model on nonsense; they are dropped on load instead.
    #   v2: sharpness renormalised to a fixed face size, blendshape-derived eye/mouth
    #       metrics, and cutout suitability replacing face centering.
    #   v3: sharpness normalised against the configured thresholds instead of a constant
    #       that no longer matched them (it saturated at a Laplacian of 70 while real
    #       captures measure 100-166, so the feature was pinned at 1.0 for most rows),
    #       plus the signals the scorer had gained since: whether anything crosses the
    #       body, hand position, background cleanliness and subject size.
    FEATURE_VERSION = 3

    # Below this many labelled samples the model isn't worth blending into the score.
    MIN_SAMPLES_TO_TRAIN = 12
    # Each class needs enough examples to say anything. Two of a class fits noise.
    MIN_PER_CLASS = 3

    FEATURE_NAMES = [
        "head_yaw_norm",        # 0: Head Turn (0=facing camera)
        "head_pitch_norm",      # 1: Head Pitch (0=level)
        "eye_openness",         # 2: Eye Openness (EAR)
        "smile_ratio",          # 3: Smile Ratio
        "mouth_openness_mar",   # 4: Mouth Openness (MAR)
        "shoulder_tilt_norm",   # 5: Shoulder Tilt (0=level)
        "sharpness_norm",       # 6: Laplacian Sharpness
        "exposure_score",       # 7: Exposure/Lighting
        "cutout_score",         # 8: Cutout suitability (framing, size, clean background)
        "torso_clear",          # 9: 1.0 = nothing crossing the body
        "hands_clear",          # 10: 1.0 = hands off the torso
        "background_score",     # 11: How plain the surroundings are
        "subject_height",       # 12: Subject height as a fraction of the frame
    ]

    DEFAULT_PENALTIES: Dict[str, float] = {
        "head_down": 1.0,
        "eyes_closed": 1.0,
        "mouth_open": 1.0,
        "hand_obstruction": 1.0,
        "prop_strictness": 1.0,
        "blur": 1.0,
    }

    def __init__(self):
        self.feature_count = len(self.FEATURE_NAMES)
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=1.0, solver='liblinear')
        
        # Training dataset: list of (feature_vector, label)
        self.samples: List[Tuple[List[float], int]] = []
        self.feedback_notes: List[Dict[str, Any]] = []
        self.learned_penalties: Dict[str, float] = dict(self.DEFAULT_PENALTIES)
        self.is_trained: bool = False
        self.confidence: float = 0.0
        
        self.load()

    @staticmethod
    def extract_features(analysis, thresholds: Optional[QualityThresholds] = None) -> np.ndarray:
        """Extracts a normalized 13-dimensional feature vector from an AnalysisResult."""
        t = thresholds or QualityThresholds()
        yaw_norm = min(1.0, abs(getattr(analysis, 'yaw_deg', 0.0)) / max(1.0, t.max_yaw_deg))
        pitch_norm = min(1.0, abs(getattr(analysis, 'pitch_deg', 0.0)) / max(1.0, t.max_pitch_deg))
        eye_open = min(1.0, max(0.0, getattr(analysis, 'avg_ear', 0.35) / 0.40))
        smile = min(1.0, max(0.0, getattr(analysis, 'smile_ratio', 0.5) / 1.0))
        mar = min(1.0, max(0.0, getattr(analysis, 'mar', 0.2)))
        shoulder_tilt_norm = min(1.0, max(0.0, getattr(analysis, 'shoulder_tilt_deg', 0.0) / 15.0))
        # Normalised against the live thresholds rather than a hardcoded span, so a
        # recalibration of the scorer cannot silently flatten this feature.
        floor, good = t.sharpness_floor, max(t.sharpness_floor + 1.0, t.sharpness_good)
        lap = getattr(analysis, 'laplacian_variance', floor)
        sharp = min(1.0, max(0.0, (lap - floor) / (good - floor)))

        exposure = min(1.0, max(0.0, getattr(analysis, 'exposure_score', 1.0)))
        cutout = min(1.0, max(0.0, getattr(analysis, 'cutout_score', 0.5)))
        torso_clear = min(1.0, max(0.0, getattr(analysis, 'torso_clear_score', 1.0)))
        hands_clear = 0.0 if getattr(analysis, 'hands_on_torso', False) else 1.0
        background = min(1.0, max(0.0, getattr(analysis, 'background_score', 1.0)))
        subject_h = min(1.0, max(0.0, getattr(analysis, 'subject_height_ratio', 0.0)))

        return np.array([
            yaw_norm, pitch_norm, eye_open, smile,
            mar, shoulder_tilt_norm, sharp, exposure, cutout,
            torso_clear, hands_clear, background, subject_h
        ], dtype=np.float64)

    def predict(self, features: np.ndarray) -> float:
        """Predicts preference probability score (0.0 to 1.0) for a feature vector."""
        if not self.is_trained:
            return 0.5  # Neutral default prior to training
            
        try:
            X_scaled = self.scaler.transform(features.reshape(1, -1))
            probs = self.clf.predict_proba(X_scaled)[0]
            # Prob of class 1 (Favorite)
            prob_favorite = float(probs[1]) if len(probs) > 1 else float(probs[0])
            return float(round(prob_favorite, 3))
        except Exception:
            return 0.5

    def add_sample(self, analysis, is_favorite: bool, comment: str = "",
                   reasons: Optional[List[str]] = None):
        """Adds a human feedback sample (1 for Favorite, 0 for Reject), optionally with
        structured reason tags and a free-text comment explaining why, and retrains."""
        features = self.extract_features(analysis).tolist()
        label = 1 if is_favorite else 0
        self.samples.append((features, label))

        clean_comment = (comment or "").strip()
        clean_reasons = [str(r).strip() for r in (reasons or []) if str(r).strip()]
        if clean_comment or clean_reasons:
            import time as _time
            self.feedback_notes.append({
                "timestamp": round(_time.time(), 1),
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
            self.feedback_notes = self.feedback_notes[-50:]
            self._apply_reason_learning(features, is_favorite, clean_comment, clean_reasons)

        self.fit()
        self.save()

    def _apply_reason_learning(self, base_features: List[float], is_favorite: bool,
                               comment: str, reasons: List[str]) -> None:
        """Translates human feedback comments/tags into immediate penalty adjustments
        and targeted feature-axis training examples so the model learns *why* a shot
        failed without needing dozens of blind binary clicks."""
        combined = (" ".join(reasons) + " " + comment).lower()
        if not is_favorite:
            synth = list(base_features)
            matched_any = False

            # 1. Bowed head / looking down / chin down / reading notes or box
            if any(k in combined for k in ("head", "bow", "down", "chin", "gaze", "looking", "floor", "notes")):
                self.learned_penalties["head_down"] = min(2.5, round(self.learned_penalties.get("head_down", 1.0) + 0.25, 2))
                synth[1] = max(synth[1], 0.85)   # high head_pitch_norm
                synth[2] = min(synth[2], 0.60)   # lower eye_openness
                matched_any = True

            # 2. Eyes closed / blinking / hidden behind glasses
            if any(k in combined for k in ("eye", "blink", "closed", "squint", "glasses")):
                self.learned_penalties["eyes_closed"] = min(2.5, round(self.learned_penalties.get("eyes_closed", 1.0) + 0.25, 2))
                synth[2] = min(synth[2], 0.50)   # low eye_openness
                matched_any = True

            # 3. Mouth open mid-word / talking / awkward expression
            if any(k in combined for k in ("mouth", "talk", "word", "open", "grimace", "awkward", "expression")):
                self.learned_penalties["mouth_open"] = min(2.5, round(self.learned_penalties.get("mouth_open", 1.0) + 0.25, 2))
                synth[4] = max(synth[4], 0.65)   # high mouth_openness_mar
                synth[3] = min(synth[3], 0.35)   # low smile_ratio
                matched_any = True

            # 4. Microphone / hand blocking face or chest / obstruction
            if any(k in combined for k in ("mic", "hand", "block", "cover", "obstruct", "chest", "stand", "ipad")):
                self.learned_penalties["hand_obstruction"] = min(2.5, round(self.learned_penalties.get("hand_obstruction", 1.0) + 0.25, 2))
                synth[9] = min(synth[9], 0.35)   # low torso_clear
                synth[10] = 0.0                  # hands_clear = 0
                matched_any = True

            # 5. False prop / box at bottom / not presenting prop
            if any(k in combined for k in ("prop", "box", "holding", "cut off", "clipped")):
                self.learned_penalties["prop_strictness"] = min(2.5, round(self.learned_penalties.get("prop_strictness", 1.0) + 0.30, 2))
                synth[8] = min(synth[8], 0.45)   # lower cutout_score
                matched_any = True

            # 6. Motion blur / soft focus
            if any(k in combined for k in ("blur", "soft", "focus", "smear", "motion")):
                self.learned_penalties["blur"] = min(2.5, round(self.learned_penalties.get("blur", 1.0) + 0.25, 2))
                synth[6] = min(synth[6], 0.25)   # low sharpness_norm
                matched_any = True

            if matched_any:
                # Reinforce the targeted negative pattern in the classifier dataset
                self.samples.append((synth, 0))

    def get_judge_feedback_prompt(self, max_notes: int = 8) -> str:
        """Formats the operator's recent rejection/approval comments into a directive
        block for the Vision LLM (AIJudge) so it learns from human explanations."""
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

    def fit(self):
        """Fits the scaler + classifier and estimates how well it actually generalises."""
        self.is_trained = False
        self.confidence = 0.0

        if len(self.samples) < self.MIN_SAMPLES_TO_TRAIN:
            return

        X = np.array([s[0] for s in self.samples], dtype=np.float64)
        y = np.array([s[1] for s in self.samples], dtype=np.int32)

        # Both a liked and a disliked example are needed; one class alone says nothing
        # about what distinguishes them. Waiting is better than fitting a synthetic
        # placeholder row and calling the result a trained model.
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or counts.min() < self.MIN_PER_CLASS:
            return

        try:
            self.scaler = StandardScaler()
            # Ratings skew heavily towards rejects, and without balancing the fit mostly
            # learns to say "reject", which scores well on raw accuracy while telling
            # you nothing.
            self.clf = LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced')
            X_scaled = self.scaler.fit_transform(X)
            self.clf.fit(X_scaled, y)
            self.is_trained = True
            self.confidence = self._cross_val_accuracy(X, y)
        except Exception as e:
            print(f"[UserPreferenceModel] Scikit-Learn fit error: {e}")

    @property
    def trust(self) -> float:
        """How far past guessing the model actually is, on 0..1.

        Balanced accuracy is 0.5 for a model that has learned nothing, so the useful
        quantity is the distance above that. With ratings running 4 favourites to 18
        rejects, raw accuracy of 0.92 sounded authoritative while 0.82 of it came free
        from always answering "reject" - this reports 0 for such a model.
        """
        if not self.is_trained:
            return 0.0
        return max(0.0, min(1.0, (self.confidence - 0.5) / 0.5))

    def _cross_val_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Held-out *balanced* accuracy via stratified k-fold.

        Two separate traps, both of which this avoids. Scoring on the rows it was fitted
        to measures memorisation, not prediction. And plain accuracy on skewed labels
        rewards always answering with the majority class - balanced accuracy averages
        the per-class rates, so that strategy scores 0.5 and is seen for what it is.
        """
        try:
            n_splits = int(min(5, np.bincount(y).min()))
            if n_splits < 2:
                return 0.0
            pipe = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced'))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="balanced_accuracy")
            return float(round(scores.mean(), 4))
        except Exception:
            return 0.0

    def save(self):
        """Persists the labelled samples. The estimator itself is refit from these on
        load, so there is no pickled model to fall out of step with the ratings."""
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
                "samples": self.samples[-100:],
                "feedback_notes": self.feedback_notes[-50:],
                "learned_penalties": self.learned_penalties,
            }
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[UserPreferenceModel] Error saving preference data: {e}")

    def load(self):
        """Loads stored feedback, discarding it if it predates the current feature set."""
        json_path = self._get_json_path()
        if not os.path.exists(json_path):
            return

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[UserPreferenceModel] JSON data load error: {e}")
            return

        # Always preserve textual feedback notes and learned penalty multipliers
        if isinstance(data.get("feedback_notes"), list):
            self.feedback_notes = data["feedback_notes"][-50:]
        if isinstance(data.get("learned_penalties"), dict):
            for k in self.DEFAULT_PENALTIES:
                if k in data["learned_penalties"]:
                    try:
                        self.learned_penalties[k] = float(data["learned_penalties"][k])
                    except (TypeError, ValueError):
                        pass

        stored_version = int(data.get("feature_version", 1))
        raw_samples = data.get("samples", [])

        if stored_version != self.FEATURE_VERSION:
            # The features these samples describe no longer mean the same thing, so the
            # labels can't be reused. Keeping them would quietly bias every score.
            if raw_samples:
                print(f"[UserPreferenceModel] Discarding {len(raw_samples)} rating(s) recorded "
                      f"under feature set v{stored_version}; the scoring metrics have changed "
                      f"(now v{self.FEATURE_VERSION}). Re-rate a few captures to retrain.")
            self.samples = []
            self.is_trained = False
            self.confidence = 0.0
            self.save()
            return

        self.samples = [
            (feat, label) for feat, label in
            ((s[0], s[1]) for s in raw_samples)
            if len(feat) == self.feature_count
        ]

        # Refit from the samples rather than trusting a pickled estimator: it is cheap at
        # this size and keeps the fitted model and the stored ratings from drifting apart.
        self.fit()
