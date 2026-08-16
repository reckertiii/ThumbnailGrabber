import os
import json
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class UserPreferenceModel:
    """Scikit-Learn ML Model (Logistic Regression with StandardScaler)
    that learns user visual preference for thumbnail selection.
    
    Extracts normalized 8-D feature vectors from AnalysisResult objects and
    trains on user feedback (⭐ Favorite = 1, 👎 Rejected = 0).
    Persists model state and samples to models/scikit_preference_model.joblib.
    """
    
    MODEL_PATH = os.path.join(os.getcwd(), "models", "scikit_preference_model.joblib")
    JSON_DATA_PATH = os.path.join(os.getcwd(), "models", "user_preference_model.json")
    
    FEATURE_NAMES = [
        "head_yaw_norm",        # 0: Head Turn (0=facing camera)
        "head_pitch_norm",      # 1: Head Pitch (0=level)
        "eye_openness",         # 2: Eye Openness (EAR)
        "smile_ratio",          # 3: Smile Ratio
        "mouth_openness_mar",   # 4: Mouth Openness (MAR)
        "shoulder_tilt_norm",   # 5: Shoulder Tilt (0=level)
        "sharpness_norm",       # 6: Laplacian Sharpness
        "exposure_score",       # 7: Exposure/Lighting
        "centering_score"       # 8: Centering/Framing
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
        sharp = min(1.0, max(0.0, getattr(analysis, 'laplacian_variance', 50.0) / 150.0))
        exposure = min(1.0, max(0.0, getattr(analysis, 'exposure_score', 1.0)))
        centering = min(1.0, max(0.0, getattr(analysis, 'centering_score', 0.5)))
        
        return np.array([
            yaw_norm, pitch_norm, eye_open, smile,
            mar, shoulder_tilt_norm, sharp, exposure, centering
        ], dtype=np.float64)

    def predict(self, features: np.ndarray) -> float:
        """Predicts preference probability score (0.0 to 1.0) for a feature vector."""
        if not self.is_trained or len(self.samples) < 2:
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
        """Fits Scikit-Learn StandardScaler and LogisticRegression classifier."""
        if len(self.samples) < 2:
            self.is_trained = False
            return

        X = np.array([s[0] for s in self.samples], dtype=np.float64)
        y = np.array([s[1] for s in self.samples], dtype=np.int32)
        
        # Check if we have at least 2 distinct classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            # Synthetic balancing sample to allow fitting before 2 distinct classes
            dummy_label = 1 if unique_classes[0] == 0 else 0
            dummy_x = np.ones((1, self.feature_count), dtype=np.float64) * 0.5
            X = np.vstack([X, dummy_x])
            y = np.append(y, dummy_label)
        
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.clf.fit(X_scaled, y)
            self.is_trained = True
            
            # Confidence metric: mean accuracy score
            self.confidence = float(self.clf.score(X_scaled, y))
        except Exception as e:
            print(f"[UserPreferenceModel] Scikit-Learn fit error: {e}")

    def save(self):
        """Saves scikit-learn model using joblib and metadata using JSON."""
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        try:
            joblib.dump({"scaler": self.scaler, "clf": self.clf}, self.MODEL_PATH)
            
            data = {
                "is_trained": self.is_trained,
                "confidence": self.confidence,
                "sample_count": len(self.samples),
                "samples": self.samples[-100:]
            }
            with open(self.JSON_DATA_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[UserPreferenceModel] Error saving scikit-learn model: {e}")

    def load(self):
        """Loads scikit-learn model from joblib and JSON data with dimension migration support."""
        if os.path.exists(self.MODEL_PATH):
            try:
                saved_state = joblib.load(self.MODEL_PATH)
                # Verify scaler feature dimension
                if hasattr(saved_state.get("scaler"), "n_features_in_") and saved_state["scaler"].n_features_in_ == self.feature_count:
                    self.scaler = saved_state["scaler"]
                    self.clf = saved_state["clf"]
            except Exception as e:
                print(f"[UserPreferenceModel] Joblib load error: {e}")

        if os.path.exists(self.JSON_DATA_PATH):
            try:
                with open(self.JSON_DATA_PATH, "r") as f:
                    data = json.load(f)
                self.is_trained = bool(data.get("is_trained", False))
                self.confidence = float(data.get("confidence", 0.0))
                raw_samples = data.get("samples", [])
                migrated = []
                for s in raw_samples:
                    feat, label = s[0], s[1]
                    if len(feat) == 8:
                        # Insert default 0.0 for shoulder_tilt_norm
                        feat = feat[:5] + [0.0] + feat[5:]
                    if len(feat) == self.feature_count:
                        migrated.append((feat, label))
                self.samples = migrated
                if self.samples and (not hasattr(self.scaler, "n_features_in_") or self.scaler.n_features_in_ != self.feature_count):
                    self.fit()
            except Exception as e:
                print(f"[UserPreferenceModel] JSON data load error: {e}")
