"""
Canary Early Warning AI — FastAPI Backend
=========================================
Redesigned monitoring pipeline:
  1. InstabilityTracker  — multi-component instability score fed into BOCPD
  2. Weighted Risk Score — 45% Bayesian + 25% instability + 15% RAM + 15% variance
  3. Multi-condition alerts
  4. CSV upload preprocessing audit
  5. Enriched API response (instability_score, variance_score, warning_level, etc.)
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import sys
import os
import psutil

# Add parent dir so models package is found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bcpd import BayesianChangePointDetection

app = FastAPI(title="Canary Early Warning AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# InstabilityTracker
# =============================================================================

class InstabilityTracker:
    """
    Computes a multi-component instability score in [0, 1] that is fed into BOCPD.

    Components:
      EMA deviation   35%   |val - EMA|
      Rolling std     25%   std(last N)
      First deriv     20%   |val - prev_val|
      Second deriv    20%   |first_deriv - prev_first_deriv|

    The score is normalized against a running baseline so it stays near 0
    for stable data and rises toward 1 (and beyond, capped for weighting) when
    the signal becomes erratic.
    """

    def __init__(self, window: int = 20, ema_alpha: float = 0.3):
        self.window = window
        self.ema_alpha = ema_alpha
        self._buf: list[float] = []
        self._ema: Optional[float] = None
        self._prev_val: Optional[float] = None
        self._prev_d1: float = 0.0
        self._baseline: float = 1.0          # normalization anchor
        self._baseline_locked: bool = False

    def update(self, val: float) -> tuple[float, float, float]:
        """
        Ingest one observation.
        Returns (instability_score, rolling_std, variance_score).
        """
        # EMA
        if self._ema is None:
            self._ema = val
        else:
            self._ema = self.ema_alpha * val + (1 - self.ema_alpha) * self._ema

        ema_dev = abs(val - self._ema)

        # Rolling window
        self._buf.append(val)
        if len(self._buf) > self.window:
            self._buf.pop(0)

        rolling_std = float(np.std(self._buf)) if len(self._buf) > 1 else 0.0
        variance_score = rolling_std ** 2

        # First derivative
        d1 = abs(val - self._prev_val) if self._prev_val is not None else 0.0
        # Second derivative
        d2 = abs(d1 - self._prev_d1)

        self._prev_val = val
        self._prev_d1 = d1

        # Composite raw score
        raw = 0.35 * ema_dev + 0.25 * rolling_std + 0.20 * d1 + 0.20 * d2

        # Establish baseline from first `window` stable points
        if not self._baseline_locked and len(self._buf) >= self.window:
            # Use mean of first window as normalization factor
            self._baseline = max(raw, 1e-6)
            self._baseline_locked = True

        norm_score = raw / max(self._baseline, 1e-6)
        instability = float(np.clip(norm_score, 0.0, 3.0))   # allow >1 for collapse detection

        return instability, rolling_std, float(np.clip(variance_score / max(self._baseline**2, 1e-6), 0.0, 1.0))

    def reset(self):
        self.__init__(self.window, self.ema_alpha)


# =============================================================================
# Risk / Warning helpers
# =============================================================================

def compute_risk_score(
    bayesian_prob: float,
    instability_score: float,
    ram_percent: float,
    variance_score: float,
) -> float:
    """
    Weighted risk score 0–100.
      45% Bayesian changepoint probability (0–1)
      25% Instability score (capped 0–1)
      15% RAM percentage (0–100 → 0–1)
      15% Variance score (capped 0–1)
    """
    bp = np.clip(bayesian_prob, 0.0, 1.0)
    ins = np.clip(instability_score / 3.0, 0.0, 1.0)   # normalise capped range
    ram = np.clip(ram_percent / 100.0, 0.0, 1.0)
    var = np.clip(variance_score, 0.0, 1.0)

    risk = (0.45 * bp + 0.25 * ins + 0.15 * ram + 0.15 * var) * 100.0
    return float(np.clip(risk, 0.0, 100.0))


def get_warning_level(risk_score: float) -> tuple[str, str]:
    """Return (warning_level, recommended_action) from risk_score."""
    if risk_score < 20:
        return "LOW", "Training is stable."
    elif risk_score < 40:
        return "LOW_MEDIUM", "Monitor training closely."
    elif risk_score < 60:
        return "MEDIUM", "Possible data drift detected. Inspect dataset."
    elif risk_score < 80:
        return "HIGH", "Model becoming unstable. Inspect dataset immediately."
    else:
        return "CRITICAL", "Training should be halted. High probability of failure."


def is_alert_triggered(
    bayesian_prob: float,
    risk_score: float,
    ram_percent: float,
    metric_val: float,
    prev_metric_val: Optional[float],
    variance_score: float,
    alert_threshold: float,
) -> bool:
    """Multi-condition alert logic."""
    if bayesian_prob > alert_threshold:
        return True
    if risk_score > 40.0:
        return True
    if ram_percent > 90.0:
        return True
    if prev_metric_val is not None and abs(metric_val - prev_metric_val) > 30.0:
        return True                                     # metric collapse
    if variance_score > 2.0:
        return True                                     # variance explosion
    return False


# =============================================================================
# SimulationState
# =============================================================================

class SimulationState:
    def __init__(self):
        self.alert_threshold = 0.05
        self.reset("sudden_shift")

    def reset(self, dataset_type="sudden_shift", opt_batch_size=None, opt_epochs=None,
              ml_task_type=None, ml_model_name=None):

        self.dataset_type = dataset_type
        self.time_step = 0
        self.data: list[float] = []
        self.alerts: list[int] = []
        self.probabilities: list[float] = []

        # New tracking lists
        self.instability_scores: list[float] = []
        self.risk_scores: list[float] = []

        self.bcpd = BayesianChangePointDetection(hazard_rate=1 / 100)
        self.instability_tracker = InstabilityTracker()

        np.random.seed(42)
        if dataset_type == "sudden_shift":
            normal = np.random.normal(0, 1, 100)
            drift = np.random.normal(0, 1, 100) + np.linspace(0, 3, 100)
            shift = np.random.normal(5, 1.5, 100)
            self.synthetic_data = np.concatenate([normal, drift, shift])
        elif dataset_type == "sudden_spike":
            normal1 = np.random.normal(0, 1, 140)
            spike = np.random.normal(15, 2, 20)
            normal2 = np.random.normal(0, 1, 140)
            self.synthetic_data = np.concatenate([normal1, spike, normal2])
        elif dataset_type == "variance_shift":
            normal1 = np.random.normal(0, 0.5, 150)
            normal2 = np.random.normal(0, 4.0, 150)
            self.synthetic_data = np.concatenate([normal1, normal2])
        elif dataset_type == "gradual_drift":
            drift = np.random.normal(0, 1, 300) + np.linspace(0, 10, 300)
            self.synthetic_data = drift
        elif dataset_type == "mobilenet_training":
            self.synthetic_data = []
            self.bcpd = BayesianChangePointDetection(hazard_rate=1 / 50)
            self.instability_tracker = InstabilityTracker()

            from train_mobilenet import (NumPyMobileNetV2, DATA_ROOT, NUM_CLASSES,
                                         get_system_vram_or_ram_gb, predict_batch_size, predict_epochs)
            self.model = NumPyMobileNetV2(num_classes=NUM_CLASSES)
            self.batch_size = opt_batch_size if opt_batch_size else predict_batch_size(get_system_vram_or_ram_gb())
            self.X_accumulated: list = []
            self.Y_accumulated: list = []

            self.image_paths: list = []
            for d in range(NUM_CLASSES):
                folder = os.path.join(DATA_ROOT, str(d))
                if os.path.isdir(folder):
                    for f in os.listdir(folder):
                        if f.lower().endswith(('.jpg', '.png')):
                            self.image_paths.append((os.path.join(folder, f), d))

            dataset_size = len(self.image_paths)
            epochs = opt_epochs if opt_epochs else predict_epochs(dataset_size)
            steps_per_epoch = max(1, dataset_size // self.batch_size)
            self.total_steps = steps_per_epoch * epochs
            self._proc = psutil.Process(os.getpid())
            self.global_batch = 0

        elif dataset_type == "custom_model_training":
            self.bcpd = BayesianChangePointDetection(hazard_rate=1 / 50)
            self.instability_tracker = InstabilityTracker()

            if hasattr(self, "custom_df"):
                df = self.custom_df
                self.dataset_size = len(df)
                if self.dataset_size == 0:
                    self.dataset_type = "sudden_shift"
                    return
                if len(df.columns) > 1:
                    self.custom_X = df.iloc[:, :-1].values
                    self.custom_y = df.iloc[:, -1].values
                else:
                    self.custom_X = np.arange(self.dataset_size).reshape(-1, 1)
                    self.custom_y = df.iloc[:, 0].values

                self.ml_task_type = ml_task_type if ml_task_type else "Regression"
                self.ml_model_name = ml_model_name if ml_model_name else "Linear Regression"

                # ---- sklearn model selection (unchanged) ----
                self.sklearn_model = _select_sklearn_model(self.ml_task_type, self.ml_model_name)

                self.batch_size = opt_batch_size if opt_batch_size else 32
                epochs = opt_epochs if opt_epochs else 1
                steps_per_epoch = max(1, self.dataset_size // self.batch_size)
                self.total_steps = steps_per_epoch * epochs
                self.global_batch = 0
                self._proc = psutil.Process(os.getpid())
                self.X_accumulated = []

                # Rolling metric history for instability computation
                self._metric_history: list[float] = []
            else:
                self.dataset_type = "sudden_shift"
        else:  # default fallback
            self.synthetic_data = np.random.normal(0, 1, 300)

        if dataset_type not in ["mobilenet_training", "custom_model_training"]:
            self.total_steps = len(self.synthetic_data)


def _select_sklearn_model(task_type: str, model_name: str):
    """Return an sklearn estimator instance for the requested task/model pair."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task_type == "Classification":
            if model_name == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(max_iter=1000)
            elif model_name == "Decision Tree":
                from sklearn.tree import DecisionTreeClassifier
                return DecisionTreeClassifier()
            elif model_name == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier()
            elif model_name == "Support Vector Machine":
                from sklearn.svm import SVC
                return SVC(probability=True)
            elif model_name == "XGBoost ⭐":
                from xgboost import XGBClassifier
                return XGBClassifier(eval_metric='logloss')
            elif model_name == "K-Nearest Neighbors":
                from sklearn.neighbors import KNeighborsClassifier
                return KNeighborsClassifier()
            elif model_name == "Naive Bayes":
                from sklearn.naive_bayes import GaussianNB
                return GaussianNB()
            else:
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(max_iter=1000)
        elif task_type == "Regression":
            if model_name == "Linear Regression":
                from sklearn.linear_model import LinearRegression
                return LinearRegression()
            elif model_name == "Polynomial Regression":
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.linear_model import LinearRegression
                return make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
            elif model_name == "Ridge Regression":
                from sklearn.linear_model import Ridge
                return Ridge()
            elif model_name == "Lasso Regression":
                from sklearn.linear_model import Lasso
                return Lasso()
            elif model_name == "Random Forest Regressor":
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor()
            elif model_name == "Gradient Boosting":
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor()
            elif model_name == "XGBoost ⭐":
                from xgboost import XGBRegressor
                return XGBRegressor()
            elif model_name == "Support Vector Regression":
                from sklearn.svm import SVR
                return SVR()
            else:
                from sklearn.linear_model import LinearRegression
                return LinearRegression()
        elif task_type == "Clustering":
            if model_name == "K-Means ⭐":
                from sklearn.cluster import KMeans
                return KMeans(n_clusters=3, n_init='auto')
            elif model_name == "Hierarchical Clustering":
                from sklearn.cluster import AgglomerativeClustering
                return AgglomerativeClustering(n_clusters=3)
            elif model_name == "DBSCAN":
                from sklearn.cluster import DBSCAN
                return DBSCAN()
            elif model_name == "Gaussian Mixture Model":
                from sklearn.mixture import GaussianMixture
                return GaussianMixture(n_components=3)
            elif model_name == "Mean Shift":
                from sklearn.cluster import MeanShift
                return MeanShift()
            else:
                from sklearn.cluster import KMeans
                return KMeans(n_clusters=3, n_init='auto')
        else:
            from sklearn.linear_model import LinearRegression
            return LinearRegression()


state = SimulationState()


# =============================================================================
# Pydantic models
# =============================================================================

class StepResponse(BaseModel):
    time: int
    value: float
    probability: float
    is_alert: bool
    risk_score: float
    # --- Enriched fields ---
    bayesian_probability: float = 0.0
    instability_score: float = 0.0
    variance_score: float = 0.0
    ram_score: float = 0.0
    warning_level: str = "LOW"
    recommended_action: str = "Training is stable."


class ResetRequest(BaseModel):
    dataset_type: str = "sudden_shift"
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    ml_task_type: Optional[str] = None
    ml_model_name: Optional[str] = None


class SettingsRequest(BaseModel):
    alert_threshold: float


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
def read_root():
    return {"message": "Canary API is running"}


@app.post("/api/reset")
def reset_simulation(req: ResetRequest = None):
    dataset_type = req.dataset_type if req else "sudden_shift"
    batch_size = getattr(req, "batch_size", None) if req else None
    epochs = getattr(req, "epochs", None) if req else None
    ml_task_type = getattr(req, "ml_task_type", None) if req else None
    ml_model_name = getattr(req, "ml_model_name", None) if req else None
    state.reset(dataset_type, batch_size, epochs, ml_task_type, ml_model_name)
    return {"status": "Simulation reset", "dataset": dataset_type}


@app.get("/api/training_info")
def get_training_info(dataset: str = "mobilenet_training"):
    from train_mobilenet import get_system_vram_or_ram_gb, predict_batch_size, predict_epochs, DATA_ROOT, NUM_CLASSES

    if dataset == "custom_model_training" and hasattr(state, "custom_df") and state.custom_df is not None:
        dataset_size = len(state.custom_df)
    else:
        image_paths = []
        for d in range(NUM_CLASSES):
            folder = os.path.join(DATA_ROOT, str(d))
            if os.path.isdir(folder):
                for f in os.listdir(folder):
                    if f.lower().endswith(('.jpg', '.png')):
                        image_paths.append(f)
        dataset_size = len(image_paths)

    memory_gb = get_system_vram_or_ram_gb()
    predict_bs = predict_batch_size(memory_gb)
    predict_ep = predict_epochs(dataset_size)
    return {
        "dataset_size": dataset_size,
        "memory_gb": round(memory_gb, 2),
        "predicted_batch_size": predict_bs,
        "predicted_epochs": predict_ep,
    }


# =============================================================================
# /api/step — core monitoring loop
# =============================================================================

@app.get("/api/step", response_model=StepResponse)
def compute_next_step():
    """Simulates real-time data arriving and returns enriched monitoring payload."""

    mem_info = psutil.virtual_memory()
    sys_ram_percent = float(mem_info.percent)

    # -------------------------------------------------------------------------
    # MobileNet training path
    # -------------------------------------------------------------------------
    if getattr(state, "dataset_type", "sudden_shift") == "mobilenet_training":
        if len(state.alerts) > 0:
            last_risk = state.risk_scores[-1] if state.risk_scores else 100.0
            last_prob = state.probabilities[-1] if state.probabilities else 1.0
            wl, ra = get_warning_level(last_risk)
            return StepResponse(
                time=state.time_step - 1, value=state.data[-1],
                probability=last_prob, is_alert=True,
                risk_score=last_risk, bayesian_probability=last_prob,
                instability_score=state.instability_scores[-1] if state.instability_scores else 1.0,
                variance_score=0.0, ram_score=sys_ram_percent / 100.0,
                warning_level="CRITICAL", recommended_action="Training should be halted. High probability of failure.",
            )

        from train_mobilenet import IMG_SIZE
        from PIL import Image

        batch_size = getattr(state, "batch_size", 32)
        i = state.global_batch * batch_size
        if i >= len(state.image_paths):
            i = 0
            state.global_batch = 0

        chunk = state.image_paths[i:i + batch_size]
        state.global_batch += 1

        X_batch_list, Y_batch_list = [], []
        for path, label in chunk:
            try:
                img = Image.open(path).convert("RGB").resize(IMG_SIZE)
                arr = np.array(img, dtype=np.float32) / 255.0
                X_batch_list.append(arr)
                Y_batch_list.append(label)
            except:
                pass

        if X_batch_list:
            X_batch = np.stack(X_batch_list, axis=0)
            Y_batch = np.array(Y_batch_list, dtype=np.int32)
            _ = state.model.process_batch(X_batch)
            state.X_accumulated.append(X_batch)
            state.Y_accumulated.append(Y_batch)

        # Simulate memory growth
        if state.global_batch > 15:
            surge_size = int(1.2 ** (state.global_batch - 15) * 5 * 1024 * 1024)
            surge_size = min(surge_size, 300 * 1024 * 1024)
            try:
                state.X_accumulated.append(np.ones(surge_size, dtype=np.float32))
            except MemoryError:
                pass

        val = float(state._proc.memory_info().rss / (1024 ** 2))
        state.data.append(val)

        instability, rolling_std, var_score = state.instability_tracker.update(val)
        prob, run_length = state.bcpd.update(instability)
        state.probabilities.append(prob)
        state.instability_scores.append(instability)

        risk = compute_risk_score(prob, instability, sys_ram_percent, var_score)
        state.risk_scores.append(risk)

        prev_val = state.data[-2] if len(state.data) > 1 else None
        is_alert = is_alert_triggered(prob, risk, sys_ram_percent, val, prev_val, var_score, state.alert_threshold)
        if is_alert:
            state.alerts.append(state.time_step)

        wl, ra = get_warning_level(risk)
        current_time = state.time_step
        state.time_step += 1

        return StepResponse(
            time=current_time, value=val,
            probability=prob, is_alert=is_alert,
            risk_score=risk, bayesian_probability=prob,
            instability_score=instability, variance_score=var_score,
            ram_score=float(sys_ram_percent / 100.0),
            warning_level=wl, recommended_action=ra,
        )

    # -------------------------------------------------------------------------
    # Custom sklearn model training path
    # -------------------------------------------------------------------------
    elif getattr(state, "dataset_type", "sudden_shift") == "custom_model_training":
        if len(state.alerts) > 0:
            last_risk = state.risk_scores[-1] if state.risk_scores else 100.0
            last_prob = state.probabilities[-1] if state.probabilities else 1.0
            wl, ra = get_warning_level(last_risk)
            return StepResponse(
                time=state.time_step - 1, value=state.data[-1],
                probability=last_prob, is_alert=True,
                risk_score=last_risk, bayesian_probability=last_prob,
                instability_score=state.instability_scores[-1] if state.instability_scores else 1.0,
                variance_score=0.0, ram_score=sys_ram_percent / 100.0,
                warning_level=wl, recommended_action=ra,
            )

        batch_size = getattr(state, "batch_size", 32)
        i = (state.global_batch + 1) * batch_size
        if i >= state.dataset_size:
            i = state.dataset_size

        X_sub = state.custom_X[:i]
        y_sub = state.custom_y[:i]
        state.global_batch += 1

        metric_val = 0.0
        confidence = 0.5
        pred_entropy = 1.0

        import warnings as _warn
        with _warn.catch_warnings():
            _warn.simplefilter("ignore")
            if len(X_sub) > 5:
                try:
                    if state.ml_task_type == "Classification":
                        from sklearn.preprocessing import LabelEncoder
                        from sklearn.metrics import accuracy_score
                        le = LabelEncoder()
                        y_enc = le.fit_transform(y_sub)
                        if len(np.unique(y_enc)) > 1:
                            state.sklearn_model.fit(X_sub, y_enc)
                            preds = state.sklearn_model.predict(X_sub)
                            metric_val = float(accuracy_score(y_enc, preds) * 100.0)

                            # Confidence & entropy from predict_proba if available
                            if hasattr(state.sklearn_model, "predict_proba"):
                                proba = state.sklearn_model.predict_proba(X_sub)
                                confidence = float(np.mean(np.max(proba, axis=1)))
                                # Mean entropy across samples
                                eps = 1e-12
                                h = -np.sum(proba * np.log(proba + eps), axis=1)
                                pred_entropy = float(np.mean(h))
                        else:
                            metric_val = 50.0
                    elif state.ml_task_type == "Regression":
                        from sklearn.metrics import r2_score
                        state.sklearn_model.fit(X_sub, y_sub)
                        preds = state.sklearn_model.predict(X_sub)
                        r2 = float(r2_score(y_sub, preds) * 100.0)
                        metric_val = max(-100.0, r2)
                        # Variance of residuals as confidence proxy
                        residuals = y_sub - preds
                        pred_entropy = float(np.var(residuals))
                        confidence = float(np.clip(1.0 - abs(metric_val - 100.0) / 200.0, 0.0, 1.0))
                    elif state.ml_task_type == "Clustering":
                        from sklearn.metrics import silhouette_score
                        if hasattr(state.sklearn_model, "fit_predict"):
                            preds = state.sklearn_model.fit_predict(X_sub)
                        else:
                            state.sklearn_model.fit(X_sub)
                            preds = state.sklearn_model.predict(X_sub)
                        if len(np.unique(preds)) > 1:
                            metric_val = float(silhouette_score(X_sub, preds) * 100.0)
                        else:
                            metric_val = 0.0
                except Exception as e:
                    print(f"Metric calculation failed: {e}")
                    metric_val = 0.0

        # Track rolling metric for volatility
        state._metric_history.append(metric_val)
        rolling_acc_vol = float(np.std(state._metric_history[-5:])) if len(state._metric_history) >= 2 else 0.0

        # Build a composite "training signal" for the instability tracker
        # Use normalized accuracy + confidence deviation + entropy
        training_signal = metric_val / 100.0 * (1.0 + rolling_acc_vol / 100.0)

        val = metric_val
        state.data.append(val)

        instability, rolling_std, var_score = state.instability_tracker.update(training_signal)

        # Elevate instability based on entropy (high entropy → model uncertain)
        instability = float(np.clip(instability + min(pred_entropy / 5.0, 0.5), 0.0, 3.0))

        prob, run_length = state.bcpd.update(instability)
        state.probabilities.append(prob)
        state.instability_scores.append(instability)

        risk = compute_risk_score(prob, instability, sys_ram_percent, var_score)
        state.risk_scores.append(risk)

        prev_val = state.data[-2] if len(state.data) > 1 else None
        is_alert = is_alert_triggered(prob, risk, sys_ram_percent, val, prev_val, var_score, state.alert_threshold)
        if is_alert:
            state.alerts.append(state.time_step)

        current_time = state.time_step
        state.time_step += 1

        if i >= state.dataset_size and state.time_step < state.total_steps:
            state.global_batch = 0

        wl, ra = get_warning_level(risk)
        return StepResponse(
            time=current_time, value=val,
            probability=prob, is_alert=is_alert,
            risk_score=risk, bayesian_probability=prob,
            instability_score=instability, variance_score=var_score,
            ram_score=float(sys_ram_percent / 100.0),
            warning_level=wl, recommended_action=ra,
        )

    # -------------------------------------------------------------------------
    # Synthetic dataset path (sudden_shift, gradual_drift, spike, variance, etc.)
    # -------------------------------------------------------------------------
    if state.time_step >= state.total_steps:
        state.time_step = 0

    val = float(state.synthetic_data[state.time_step])
    state.data.append(val)

    instability, rolling_std, var_score = state.instability_tracker.update(val)
    prob, run_length = state.bcpd.update(instability)
    state.probabilities.append(prob)
    state.instability_scores.append(instability)

    risk = compute_risk_score(prob, instability, sys_ram_percent, var_score)
    state.risk_scores.append(risk)

    prev_val = state.data[-2] if len(state.data) > 1 else None
    is_alert = is_alert_triggered(prob, risk, sys_ram_percent, val, prev_val, var_score, state.alert_threshold)
    if is_alert:
        state.alerts.append(state.time_step)

    current_time = state.time_step
    state.time_step += 1

    wl, ra = get_warning_level(risk)
    return StepResponse(
        time=current_time, value=val,
        probability=prob, is_alert=is_alert,
        risk_score=risk, bayesian_probability=prob,
        instability_score=instability, variance_score=var_score,
        ram_score=float(sys_ram_percent / 100.0),
        warning_level=wl, recommended_action=ra,
    )


# =============================================================================
# /api/history
# =============================================================================

@app.get("/api/history")
def get_history():
    return {
        "times": list(range(len(state.data))),
        "values": state.data,
        "probabilities": state.probabilities,
        "alerts": state.alerts,
        "alert_threshold": state.alert_threshold,
        "risk_scores": state.risk_scores,
        "instability_scores": state.instability_scores,
    }


# =============================================================================
# /api/settings
# =============================================================================

@app.post("/api/settings")
def update_settings(req: SettingsRequest):
    state.alert_threshold = req.alert_threshold
    return {"status": "Settings updated", "alert_threshold": state.alert_threshold}


# =============================================================================
# /api/inject_anomaly
# =============================================================================

@app.post("/api/inject_anomaly")
def inject_anomaly():
    if state.time_step < state.total_steps:
        remaining = state.total_steps - state.time_step
        shift = np.random.normal(15, 2.0, remaining)
        state.synthetic_data[state.time_step:] += shift
    return {"status": "Anomaly injected"}


# =============================================================================
# /api/stop_training
# =============================================================================

@app.post("/api/stop_training")
def stop_training():
    if getattr(state, "dataset_type", "") == "mobilenet_training":
        state.dataset_type = "stopped"
        state.X_accumulated = []
        state.Y_accumulated = []
        import gc
        gc.collect()
    return {"status": "Training stopped"}


# =============================================================================
# /api/upload_csv  — with preprocessing audit
# =============================================================================

@app.post("/api/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    import pandas as pd
    import io

    contents = await file.read()
    filename = file.filename.lower()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"error": "Unsupported file format. Please upload a CSV or Excel file."}
    except Exception as e:
        return {"error": f"Failed to parse file: {str(e)}"}

    # -----------------------------------------------------------------
    # Phase 1 — Preprocessing audit (return warnings, don't silently fix)
    # -----------------------------------------------------------------
    warnings_list: list[str] = []

    # Missing values
    total_missing = int(df.isnull().sum().sum())
    if total_missing > 0:
        warnings_list.append(f"Missing values detected ({total_missing} cells across {int((df.isnull().sum() > 0).sum())} columns)")

    # Infinite values
    numeric_df = df.select_dtypes(include=["number"])
    n_inf = int(np.isinf(numeric_df.values).sum())
    if n_inf > 0:
        warnings_list.append(f"Infinite values detected ({n_inf} cells)")

    # Constant columns
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if const_cols:
        warnings_list.append(f"Constant column(s) detected: {', '.join(str(c) for c in const_cols)}")

    # Duplicate rows
    n_dup = int(df.duplicated().sum())
    if n_dup > 0:
        warnings_list.append(f"Duplicate rows detected ({n_dup} rows)")

    # Mixed dtypes (object columns containing both numbers and strings)
    for col in df.columns:
        if df[col].dtype == object:
            parsed_num = pd.to_numeric(df[col], errors='coerce').notna().sum()
            if 0 < parsed_num < len(df):
                warnings_list.append(f"Mixed data types in column '{col}'")

    # Extreme class imbalance (last column treated as label for classification hint)
    last_col = df.columns[-1]
    if df[last_col].nunique() > 1 and df[last_col].nunique() <= 20:
        vc = df[last_col].value_counts()
        if len(vc) >= 2 and vc.iloc[0] / max(vc.iloc[-1], 1) > 10:
            warnings_list.append(f"High class imbalance in '{last_col}' (ratio {vc.iloc[0] / max(vc.iloc[-1], 1):.1f}:1)")

    # High dimensionality
    if len(df.columns) > 50:
        warnings_list.append(f"High dimensionality ({len(df.columns)} columns) — consider feature selection")

    # Near-zero variance columns
    nzv_cols = [c for c in numeric_df.columns if numeric_df[c].std() < 0.01 and numeric_df[c].std() >= 0.0]
    nzv_cols = [c for c in nzv_cols if c not in const_cols]
    if nzv_cols:
        warnings_list.append(f"Near-zero variance column(s): {', '.join(str(c) for c in nzv_cols)}")

    # Outliers (IQR method)
    outlier_cols = []
    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna().replace([np.inf, -np.inf], np.nan).dropna()
        if len(col_data) < 4:
            continue
        q1, q3 = np.percentile(col_data, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            n_out = int(((col_data < q1 - 3 * iqr) | (col_data > q3 + 3 * iqr)).sum())
            if n_out > 0:
                outlier_cols.append(f"{col}({n_out})")
    if outlier_cols:
        warnings_list.append(f"Outliers detected in: {', '.join(outlier_cols)}")

    # -----------------------------------------------------------------
    # Phase 2 — Encode and store (same logic as before)
    # -----------------------------------------------------------------
    from sklearn.preprocessing import LabelEncoder
    df_clean = df.copy()

    for col in df_clean.columns:
        if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'category':
            df_clean[col] = df_clean[col].fillna('Missing')
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    target_col = None
    for col in df_clean.columns:
        if str(col).lower().strip() == 'value':
            target_col = col
            break
    if not target_col:
        ncols = df_clean.select_dtypes(include=['number']).columns
        if len(ncols) > 0:
            target_col = ncols[0]

    if not target_col:
        return {"error": "No valid data found in the file.", "warnings": warnings_list}

    df_clean = df_clean.select_dtypes(include=['number']).dropna()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
    state.custom_df = df_clean

    parsed_data = df_clean[target_col].astype(float).tolist()
    if len(parsed_data) == 0:
        return {"error": "No valid data could be parsed from the file.", "warnings": warnings_list}

    state.synthetic_data = np.array(parsed_data)
    state.dataset_type = "csv_upload"
    state.time_step = 0
    state.data = []
    state.alerts = []
    state.probabilities = []
    state.instability_scores = []
    state.risk_scores = []
    state.bcpd = BayesianChangePointDetection(hazard_rate=1 / 100)
    state.instability_tracker = InstabilityTracker()
    state.total_steps = len(state.synthetic_data)

    preview = parsed_data[:1000]

    return {
        "status": "success",
        "length": len(parsed_data),
        "preview": preview,
        "warnings": warnings_list,
    }


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
