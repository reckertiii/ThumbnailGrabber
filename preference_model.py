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
    
    Extracts normalized 9-D feature vectors from AnalysisResult objects and
    trains on user feedback (⭐ Favorite = 1, 👎 Rejected = 0).
    Labelled samples persist to models/user_preference_model.json; the estimator is
    refit from them on load.
    """
    
    # User ratings, unlike the downloaded weights, must persist across container
    # restarts — so this stays on the mounted data directory.
    JSON_DATA_PATH = os.path.join(
        os.environ.get("TG_DATA_DIR", "").strip() or os.path.join(os.getcwd(), "models"),
        "user_preference_model.json"
    )

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

    def __init__(self):
        self.feature_count = len(self.FEATURE_NAMES)
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=1.0, solver='liblinear')
        
        # Training dataset: list of (feature_vector, label)
        self.samples: List[Tuple[List[float], int]] = []
        self.is_trained: bool = False
        self.confidence: float = 0.0
        
        self.load()

    @staticmethod
    def extract_features(analysis) -> np.ndarray:
        """Extracts a normalized 9-dimensional feature vector from an AnalysisResult."""
        yaw_norm = min(1.0, abs(getattr(analysis, 'yaw_deg', 0.0)) / 30.0)
        pitch_norm = min(1.0, abs(getattr(analysis, 'pitch_deg', 0.0)) / 25.0)
        eye_open = min(1.0, max(0.0, getattr(analysis, 'avg_ear', 0.35) / 0.40))
        smile = min(1.0, max(0.0, getattr(analysis, 'smile_ratio', 0.5) / 1.0))
        mar = min(1.0, max(0.0, getattr(analysis, 'mar', 0.2)))
        shoulder_tilt_norm = min(1.0, max(0.0, getattr(analysis, 'shoulder_tilt_deg', 0.0) / 15.0))
        # Normalised against the live thresholds rather than a hardcoded span, so a
        # recalibration of the scorer cannot silently flatten this feature.
        t = QualityThresholds()
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

    def add_sample(self, analysis, is_favorite: bool):
        """Adds a human feedback sample (1 for Favorite, 0 for Reject) and retrains scikit-learn model."""
        features = self.extract_features(analysis).tolist()
        label = 1 if is_favorite else 0
        
        self.samples.append((features, label))
        self.fit()
        self.save()

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
        os.makedirs(os.path.dirname(self.JSON_DATA_PATH), exist_ok=True)
        try:
            data = {
                "feature_version": self.FEATURE_VERSION,
                "is_trained": self.is_trained,
                "confidence": self.confidence,
                "confidence_metric": "cross_val_balanced_accuracy",
            "trust": self.trust,
                "sample_count": len(self.samples),
                "samples": self.samples[-100:]
            }
            with open(self.JSON_DATA_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[UserPreferenceModel] Error saving preference data: {e}")

    def load(self):
        """Loads stored feedback, discarding it if it predates the current feature set."""
        if not os.path.exists(self.JSON_DATA_PATH):
            return

        try:
            with open(self.JSON_DATA_PATH, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[UserPreferenceModel] JSON data load error: {e}")
            return

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
            return

        self.samples = [
            (feat, label) for feat, label in
            ((s[0], s[1]) for s in raw_samples)
            if len(feat) == self.feature_count
        ]

        # Refit from the samples rather than trusting a pickled estimator: it is cheap at
        # this size and keeps the fitted model and the stored ratings from drifting apart.
        self.fit()
