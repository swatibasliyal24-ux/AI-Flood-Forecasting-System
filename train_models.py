"""
Master Training Script
AI-Based Multi-Stage Flood Forecasting and Early Warning System
Target: Almatti Reservoir (Lat: 16.33 N, Lon: 75.88 E)
GNIT / AKTU Academic Mini Project
"""

import os
import sys
import json
import time

# Ensure workspace root is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import create_master_hourly_dataset, create_master_daily_dataset, ALMATTI_METADATA
from src.feature_engineering import prepare_stage1_rainfall_features, prepare_stage2_water_level_features
from src.stage1_rainfall import Stage1RainfallForecaster
from src.stage2_water_level import Stage2WaterLevelForecaster
from src.stage4_risk_assessment import FloodRiskClassifier


def main():
    print("=" * 70)
    print("AI-BASED MULTI-STAGE FLOOD FORECASTING AND EARLY WARNING SYSTEM")
    print(f"Target: {ALMATTI_METADATA['name']}")
    print(f"Coordinates: Lat {ALMATTI_METADATA['latitude']} N, Lon {ALMATTI_METADATA['longitude']} E")
    print(f"River: {ALMATTI_METADATA['river']} | Basin: {ALMATTI_METADATA['basin']}")
    print("=" * 70)

    start_all = time.time()

    # Step 1: Load and Preprocess Data
    print("\n[Step 1 & 2] Loading and Preprocessing Telemetry Data...")
    hourly_df = create_master_hourly_dataset()
    print(f"Loaded master hourly telemetry. Shape: {hourly_df.shape}")
    print(f"Date range: {hourly_df.index.min()} to {hourly_df.index.max()}")
    print(f"Valid water level records: {hourly_df['Water_Level_m'].notna().sum()}")

    # Step 3: Stage 1 Rainfall Forecasting Model Training
    print("\n" + "-" * 50)
    print("[Stage 1] Training Multi-Horizon Rainfall Forecasting Models (1h, 6h, 12h, 24h)...")
    from src.data_loader import load_almatti_weather_dataset
    weather_df = load_almatti_weather_dataset()
    if weather_df.empty:
        weather_df = hourly_df

    clean_s1, s1_feats, s1_targets = prepare_stage1_rainfall_features(
        weather_df,
        rain_col="Precipitation_mm" if "Precipitation_mm" in weather_df.columns else "Rainfall_Hourly_mm"
    )
    print(f"Stage 1 Clean Training Samples: {len(clean_s1)}, Features: {len(s1_feats)}")

    s1_forecaster = Stage1RainfallForecaster(horizons=[1, 6, 12, 24])
    s1_metrics = s1_forecaster.train(clean_s1, s1_feats, s1_targets)
    s1_forecaster.save()


    # Step 4: Stage 2 Water-Level Prediction Model Training
    print("\n" + "-" * 50)
    print("[Stage 2] Training Multi-Horizon Water-Level Forecasting Models (1h, 6h, 12h, 24h)...")
    clean_s2, s2_feats, s2_targets = prepare_stage2_water_level_features(hourly_df)
    clean_s2["current_wl"] = clean_s2["Water_Level_m"]
    if "current_wl" not in s2_feats:
        s2_feats.append("current_wl")
    print(f"Stage 2 Clean Training Samples: {len(clean_s2)}, Features: {len(s2_feats)}")

    s2_forecaster = Stage2WaterLevelForecaster(horizons=[1, 6, 12, 24])
    s2_metrics = s2_forecaster.train(clean_s2, s2_feats, s2_targets)
    s2_forecaster.save()

    # Step 5: Stage 4 Flood Risk Classifier Training & Confusion Matrix
    print("\n" + "-" * 50)
    print("[Stage 4] Training Flood Risk Classifier & Generating Confusion Matrix...")
    risk_classifier = FloodRiskClassifier()
    s4_metrics = risk_classifier.train_classifier(clean_s2)
    risk_classifier.save()

    total_time = round(time.time() - start_all, 2)
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE IN {total_time} SECONDS!")
    print("Summary of Evaluation Metrics across Stages:")
    print("Stage 1 (Rainfall):")
    for h, m in s1_metrics.items():
        print(f"  {h}h -> MAE: {m['mae_mm']} mm, RMSE: {m['rmse_mm']} mm, R2: {m['r2_score']}, F1: {m['f1_score']}")

    print("\nStage 2 (Water Level):")
    for h, m in s2_metrics.items():
        print(f"  {h}h -> MAE: {m['mae_meters']} m, RMSE: {m['rmse_meters']} m, R2: {m['r2_score']}, MAPE: {m['mape_percent']}%")

    print("\nStage 4 (Risk Classification):")
    print(f"  Accuracy: {s4_metrics['accuracy']*100:.2f}%, Weighted F1: {s4_metrics['weighted_f1']:.4f}")
    print(f"  Confusion Matrix: {s4_metrics['confusion_matrix']}")
    print("=" * 70)

    # Save summary report JSON
    summary_path = os.path.join(ROOT_DIR, "models", "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "target": ALMATTI_METADATA["name"],
            "coordinates": {"lat": ALMATTI_METADATA["latitude"], "lon": ALMATTI_METADATA["longitude"]},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage1_rainfall": s1_metrics,
            "stage2_water_level": s2_metrics,
            "stage4_risk": s4_metrics
        }, f, indent=2)
    print(f"Training summary saved to {summary_path}")


if __name__ == "__main__":
    main()
