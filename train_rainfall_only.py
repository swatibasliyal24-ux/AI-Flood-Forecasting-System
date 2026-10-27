"""Retrain only Stage 1 rainfall models using the hourly Almatti weather dataset."""
from src.data_loader import load_almatti_weather_dataset
from src.feature_engineering import prepare_stage1_rainfall_features
from src.stage1_rainfall import Stage1RainfallForecaster


def main():
    weather = load_almatti_weather_dataset()
    if weather.empty:
        raise SystemExit("Hourly weather data not found: data/almatti_ecmwf_hourly_weather.csv")
    clean, features, targets = prepare_stage1_rainfall_features(
        weather, rain_col="Precipitation_mm", horizons=[1, 6, 12, 24]
    )
    print(f"Training rows: {len(clean):,}; features: {features}")
    model = Stage1RainfallForecaster(horizons=[1, 6, 12, 24])
    metrics = model.train(clean, features, targets, test_size=0.2)
    model.save()
    print("Chronological hold-out metrics:")
    for horizon, values in metrics.items():
        print(horizon, values)

if __name__ == "__main__":
    main()
