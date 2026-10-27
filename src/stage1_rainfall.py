"""Stage 1: Leakage-safe multi-horizon rainfall forecasting for Almatti."""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

class Stage1RainfallForecaster:
    def __init__(self, horizons=None):
        self.horizons = horizons or [1, 6, 12, 24]
        self.models = {}
        self.metrics = {}
        self.feature_names = {}

    def train(self, df, all_feature_cols, target_cols, test_size=0.2):
        df = df.sort_index()
        split_idx = int(len(df) * (1 - test_size))
        if split_idx <= 0 or split_idx >= len(df):
            raise ValueError("Not enough chronological samples to split train/test data.")
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
        for h, target in zip(self.horizons, target_cols):
            features = list(all_feature_cols)
            self.feature_names[h] = features
            model = XGBRegressor(
                n_estimators=250, max_depth=4, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.9,
                objective="reg:squarederror", random_state=42, n_jobs=-1,
            )
            model.fit(train_df[features], train_df[target])
            preds = np.clip(model.predict(test_df[features]), 0, None)
            actual = test_df[target].to_numpy()
            mae = float(mean_absolute_error(actual, preds))
            rmse = float(np.sqrt(mean_squared_error(actual, preds)))
            r2 = float(r2_score(actual, preds)) if len(actual) > 1 else float("nan")
            actual_rain, pred_rain = actual > 0.5, preds > 0.5
            tp = int(np.sum(actual_rain & pred_rain))
            fp = int(np.sum(~actual_rain & pred_rain))
            fn = int(np.sum(actual_rain & ~pred_rain))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            self.models[h] = model
            self.metrics[h] = {
                "horizon_hours": h, "mae_mm": round(mae, 4),
                "rmse_mm": round(rmse, 4), "r2_score": round(r2, 4),
                "precision": round(precision, 3), "recall": round(recall, 3),
                "f1_score": round(f1, 3), "test_samples": int(len(actual)),
            }
            print(f"[Stage 1] {h}h: MAE={mae:.3f} mm | RMSE={rmse:.3f} mm | R2={r2:.3f} | F1={f1:.3f}")
        return self.metrics

    def predict(self, feature_vector, *args, **kwargs):
        # The optional external forecast is intentionally NOT blended here.
        # This returns an independent ML estimate for comparison.
        if isinstance(feature_vector, pd.Series):
            values = feature_vector.to_dict()
        elif isinstance(feature_vector, dict):
            values = dict(feature_vector)
        else:
            values = {}
        forecasts = {}
        for h in self.horizons:
            model = self.models.get(h)
            features = self.feature_names.get(h, [])
            if model is None or not features:
                continue
            row = [[float(values.get(name, np.nan)) for name in features]]
            if not np.isfinite(np.asarray(row, dtype=float)).all():
                # Avoid silently substituting zero for unavailable live inputs.
                forecasts[h] = float("nan")
                continue
            forecasts[h] = round(float(np.clip(model.predict(row)[0], 0, None)), 2)
        return forecasts

    def save(self, filepath=None):
        filepath = filepath or os.path.join(MODEL_DIR, "stage1_rainfall.joblib")
        joblib.dump({"models": self.models, "metrics": self.metrics,
                     "feature_names": self.feature_names, "horizons": self.horizons}, filepath)
        print(f"Stage 1 models saved to {filepath}")

    @classmethod
    def load(cls, filepath=None):
        filepath = filepath or os.path.join(MODEL_DIR, "stage1_rainfall.joblib")
        if not os.path.exists(filepath):
            return None
        payload = joblib.load(filepath)
        obj = cls(horizons=payload["horizons"])
        obj.models, obj.metrics, obj.feature_names = payload["models"], payload["metrics"], payload["feature_names"]
        return obj


def harmonize_with_google_forecast(ai_forecast, live_forecast_hourly):
    """Compare the ML estimate with the weather-provider forecast honestly.

    The live provider forecast is used as the dashboard's operational weather
    input; AI values remain independent comparison values. No artificial
    agreement floor or forced calibration is applied.
    """
    comparison = {}
    for h in [1, 6, 12, 24]:
        if live_forecast_hourly is not None and len(live_forecast_hourly) >= h:
            provider_val = float(np.nansum(np.asarray(live_forecast_hourly[:h], dtype=float)))
        else:
            provider_val = float("nan")
        ai_val = float(ai_forecast.get(h, float("nan")))
        if np.isfinite(provider_val) and np.isfinite(ai_val):
            delta = abs(ai_val - provider_val)
            match_pct = max(0.0, 100.0 * (1.0 - delta / max(provider_val, 0.1)))
        else:
            delta, match_pct = float("nan"), float("nan")
        comparison[h] = {
            "ai_pred_mm": round(ai_val, 2) if np.isfinite(ai_val) else None,
            "google_pred_mm": round(provider_val, 2) if np.isfinite(provider_val) else None,
            "delta_mm": round(delta, 2) if np.isfinite(delta) else None,
            "match_pct": round(match_pct, 1) if np.isfinite(match_pct) else None,
            # Use provider forecast for the downstream risk calculation when available.
            "blended_mm": round(provider_val, 2) if np.isfinite(provider_val) else (round(ai_val, 2) if np.isfinite(ai_val) else 0.0),
        }
    return comparison
