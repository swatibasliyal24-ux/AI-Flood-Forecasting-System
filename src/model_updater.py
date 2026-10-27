"""
Continuous Model Updating & Retraining Engine (Synopsis Step 8)
AI-Based Multi-Stage Flood Forecasting and Early Warning System
Target: Almatti Reservoir (Lat: 16.33 N, Lon: 75.88 E)
GNIT / AKTU Academic Mini Project
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from .data_loader import create_master_hourly_dataset
from .feature_engineering import prepare_stage1_rainfall_features, prepare_stage2_water_level_features
from .stage1_rainfall import Stage1RainfallForecaster
from .stage2_water_level import Stage2WaterLevelForecaster
from .stage4_risk_assessment import FloodRiskClassifier

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "retraining_log.json")
OBS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "new_live_observations.csv")


def record_live_observation(timestamp, water_level_m, rainfall_hourly_mm):
    """
    Appends a new real-time telemetry observation to the incremental storage buffer.
    """
    os.makedirs(os.path.dirname(OBS_PATH), exist_ok=True)
    new_row = pd.DataFrame([{
        "DateTime": timestamp,
        "Water_Level_m": float(water_level_m),
        "Rainfall_Hourly_mm": float(rainfall_hourly_mm)
    }])
    if os.path.exists(OBS_PATH):
        new_row.to_csv(OBS_PATH, mode="a", header=False, index=False)
    else:
        new_row.to_csv(OBS_PATH, mode="w", header=True, index=False)
    return True


def trigger_model_retraining(epochs_or_estimators=100):
    """
    Executes automated model updating as specified in Synopsis Step 8:
    1. Ingests base telemetry + new observations.
    2. Regenerates engineered feature matrices.
    3. Retrains Stage 1, Stage 2, and Stage 4 models.
    4. Computes performance improvements.
    5. Saves updated model artifacts and writes to retraining log.
    """
    start_time = datetime.now()
    hourly_df = create_master_hourly_dataset()

    # Append new observations if available
    if os.path.exists(OBS_PATH):
        try:
            obs = pd.read_csv(OBS_PATH)
            obs["DateTime"] = pd.to_datetime(obs["DateTime"])
            obs = obs.set_index("DateTime")
            hourly_df = pd.concat([hourly_df, obs]).sort_index()
            hourly_df = hourly_df[~hourly_df.index.duplicated(keep="last")]
        except Exception as e:
            print(f"Error appending observations: {e}")

    # Retrain Stage 1 (Rainfall)
    clean_s1, s1_feats, s1_targets = prepare_stage1_rainfall_features(hourly_df)
    s1_model = Stage1RainfallForecaster()
    s1_metrics = s1_model.train(clean_s1, s1_feats, s1_targets)
    s1_model.save()

    # Retrain Stage 2 (Water Level)
    clean_s2, s2_feats, s2_targets = prepare_stage2_water_level_features(hourly_df)
    s2_model = Stage2WaterLevelForecaster()
    s2_metrics = s2_model.train(clean_s2, s2_feats, s2_targets)
    s2_model.save()

    # Retrain Stage 4 (Risk Classifier)
    risk_classifier = FloodRiskClassifier()
    s4_metrics = risk_classifier.train_classifier(clean_s2)
    risk_classifier.save()

    end_time = datetime.now()
    duration_s = (end_time - start_time).total_seconds()

    log_entry = {
        "timestamp": end_time.isoformat(),
        "status": "SUCCESS",
        "training_duration_seconds": round(duration_s, 2),
        "total_training_samples": len(clean_s2),
        "stage1_metrics": s1_metrics,
        "stage2_metrics": s2_metrics,
        "stage4_metrics": {
            "accuracy": s4_metrics["accuracy"],
            "weighted_f1": s4_metrics["weighted_f1"]
        }
    }

    # Save to history log
    logs = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r") as f:
                logs = json.load(f)
        except Exception:
            logs = []

    logs.append(log_entry)
    with open(LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)

    return log_entry


def get_retraining_history():
    """
    Returns history of retraining cycles.
    """
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []
