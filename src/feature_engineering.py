
"""
Feature Engineering Module for Multi-Stage Flood Forecasting
Targets: 1h, 6h, 12h, 24h prediction horizons
Almatti Reservoir (Lat: 16.33 N, Lon: 75.88 E)
"""

import numpy as np
import pandas as pd

from .data_loader import ALMATTI_METADATA


# =========================================================
# CALENDAR FEATURES
# =========================================================

def add_cyclical_calendar_features(df, time_col=None):
    """Add cyclical time features and a monsoon indicator."""

    data = df.copy()

    if time_col is not None:
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(data[time_col])
        )
    elif isinstance(data.index, pd.DatetimeIndex):
        timestamps = data.index
    else:
        raise ValueError(
            "DataFrame must have a DatetimeIndex or a time_col."
        )

    day_of_year = timestamps.dayofyear
    month = timestamps.month
    hour = timestamps.hour

    data["sin_doy"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )
    data["cos_doy"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    data["sin_month"] = np.sin(
        2 * np.pi * month / 12.0
    )
    data["cos_month"] = np.cos(
        2 * np.pi * month / 12.0
    )

    data["sin_hour"] = np.sin(
        2 * np.pi * hour / 24.0
    )
    data["cos_hour"] = np.cos(
        2 * np.pi * hour / 24.0
    )

    data["is_monsoon"] = (
        month.isin([6, 7, 8, 9, 10]).astype(int)
    )

    return data


# =========================================================
# STAGE 1: HOURLY RAINFALL FORECASTING
# =========================================================

def prepare_stage1_rainfall_features(
    df,
    rain_col="Rainfall_Hourly_mm",
    horizons=[1, 6, 12, 24],
):
    """Prepare leakage-safe rainfall targets from hourly historical weather.

    Features are only information available at timestamp t: calendar, current
    rainfall and current meteorological observations. Targets are accumulated
    rainfall over the next 1/6/12/24 hours. No future rainfall is used as a
    feature and no missing observation is silently converted to zero.
    """
    if rain_col not in df.columns:
        raise ValueError(f"Required rainfall column '{rain_col}' is missing.")
    if not isinstance(df.index, pd.DatetimeIndex):
        if "DateTime" in df.columns:
            df = df.copy()
            df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
            df = df.dropna(subset=["DateTime"]).set_index("DateTime")
        else:
            raise ValueError("Rainfall data must have a DateTime index or DateTime column.")

    data = df.copy().sort_index()
    data = data[~data.index.duplicated(keep="last")]
    data[rain_col] = pd.to_numeric(data[rain_col], errors="coerce")
    data.loc[data[rain_col] < 0, rain_col] = np.nan
    data = add_cyclical_calendar_features(data)

    # Use a consistent feature name between the historical dataset and live API.
    data["Rainfall_Hourly_mm"] = data[rain_col]
    weather_features = [
        "Rainfall_Hourly_mm", "Temperature_C", "Humidity_pct",
        "Pressure_hPa", "WindSpeed_kmh",
    ]
    for col in weather_features:
        if col not in data.columns:
            data[col] = np.nan

    feature_cols = [
        "sin_doy", "cos_doy", "sin_month", "cos_month",
        "sin_hour", "cos_hour", "is_monsoon",
        *weather_features,
    ]

    target_cols = []
    future_rain = data["Rainfall_Hourly_mm"].shift(-1)
    for horizon in horizons:
        target_col = f"target_rain_{horizon}h"
        data[target_col] = (
            future_rain.iloc[::-1]
            .rolling(window=horizon, min_periods=horizon)
            .sum()
            .iloc[::-1]
        )
        target_cols.append(target_col)

    clean_data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=feature_cols + target_cols)
        .copy()
    )
    if clean_data.empty:
        raise ValueError(
            "No valid rainfall samples remain. Check that the hourly weather CSV "
            "contains precipitation and meteorological columns."
        )
    return clean_data, feature_cols, target_cols


# =========================================================
# STAGE 2: WATER-LEVEL FORECASTING
# =========================================================

def prepare_stage2_water_level_features(
    df,
    wl_col="Water_Level_m",
    rain_col="Rainfall_Hourly_mm",
    horizons=[1, 6, 12, 24],
):
    """
    Build water-level forecasting features.

    Missing rainfall is retained as missing. Training rows
    requiring unavailable rainfall are excluded.
    """

    if wl_col not in df.columns:
        raise ValueError(
            f"Required water-level column '{wl_col}' is missing."
        )

    if rain_col not in df.columns:
        raise ValueError(
            f"Required rainfall column '{rain_col}' is missing."
        )

    data = df.copy().sort_index()

    data[wl_col] = pd.to_numeric(
        data[wl_col],
        errors="coerce",
    )

    data[rain_col] = pd.to_numeric(
        data[rain_col],
        errors="coerce",
    )

    data = add_cyclical_calendar_features(data)

    water_level = data[wl_col]
    rainfall = data[rain_col]

    # -----------------------------------------------------
    # Water-level lags
    # -----------------------------------------------------

    for lag in [1, 2, 3, 6, 12, 24, 48]:
        data[f"wl_lag_{lag}h"] = (
            water_level.shift(lag)
        )

    data["rate_of_rise_1h"] = (
        water_level - data["wl_lag_1h"]
    )

    data["rate_of_rise_3h"] = (
        water_level - data["wl_lag_3h"]
    ) / 3.0

    data["rate_of_rise_6h"] = (
        water_level - data["wl_lag_6h"]
    ) / 6.0

    data["rate_of_rise_24h"] = (
        water_level - data["wl_lag_24h"]
    ) / 24.0

    # -----------------------------------------------------
    # Rolling water-level statistics
    # -----------------------------------------------------

    for window in [6, 12, 24, 72]:
        rolling_wl = water_level.rolling(
            window=window,
            min_periods=window,
        )

        data[f"wl_mean_{window}h"] = (
            rolling_wl.mean()
        )

        data[f"wl_std_{window}h"] = (
            rolling_wl.std()
        )

        data[f"wl_min_{window}h"] = (
            rolling_wl.min()
        )

        data[f"wl_max_{window}h"] = (
            rolling_wl.max()
        )

    # -----------------------------------------------------
    # Reservoir threshold features
    # -----------------------------------------------------

    frl = ALMATTI_METADATA["frl_m"]
    warning = ALMATTI_METADATA["warning_level_m"]

    data["buffer_to_frl"] = frl - water_level
    data["buffer_to_warning"] = warning - water_level

    data["is_above_warning"] = (
        water_level >= warning
    ).astype(int)

    # -----------------------------------------------------
    # Rainfall history
    # -----------------------------------------------------

    for window in [1, 3, 6, 12, 24, 48, 72]:
        data[f"rain_accum_{window}h"] = (
            rainfall.rolling(
                window=window,
                min_periods=window,
            ).sum()
        )

    # -----------------------------------------------------
    # Upstream water-level features
    # -----------------------------------------------------

    if "WL_Kurundwad_m" in data.columns:
        upstream = pd.to_numeric(
            data["WL_Kurundwad_m"],
            errors="coerce",
        )

        data["kw_wl"] = upstream
        data["kw_lag_12h"] = upstream.shift(12)
        data["kw_lag_24h"] = upstream.shift(24)

        data["kw_rate_24h"] = (
            data["kw_wl"] - data["kw_lag_24h"]
        ) / 24.0

    # -----------------------------------------------------
    # Future water-level targets
    # -----------------------------------------------------

    target_cols = []

    for horizon in horizons:
        target_col = f"target_wl_{horizon}h"

        data[target_col] = water_level.shift(-horizon)

        target_cols.append(target_col)

    # -----------------------------------------------------
    # Clean sparse station columns & fill weather predictors
    # -----------------------------------------------------
    sparse_cols = [c for c in data.columns if c.startswith("Rain_") or c == "WL_Kurundwad_m" or c.startswith("kw_")]
    data = data.drop(columns=sparse_cols, errors="ignore")

    for c in ["Temperature_C", "Humidity_pct", "Pressure_hPa", "WindSpeed_kmh", "Rainfall_Daily_mm"]:
        if c in data.columns:
            data[c] = data[c].ffill().bfill().fillna(0.0)

    # -----------------------------------------------------
    # Select valid features
    # -----------------------------------------------------
    excluded_cols = set(
        target_cols + [wl_col, "DateTime"]
    )

    feature_cols = [
        col for col in data.columns
        if col not in excluded_cols
        and not col.startswith("target_")
    ]

    # Required valid target and antecedent level
    required_cols = [wl_col, "wl_lag_24h"] + target_cols

    clean_data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required_cols)
        .copy()
    )

    # Fill any remaining NaNs in feature_cols with 0.0
    clean_data[feature_cols] = clean_data[feature_cols].fillna(0.0)

    if clean_data.empty:
        raise ValueError(
            "No valid Stage 2 training samples remain. "
            "Check missing rainfall and water-level data."
        )

    return clean_data, feature_cols, target_cols



# =========================================================
# DAILY FEATURES
# =========================================================

def prepare_daily_features(
    daily_df,
    horizons=[1, 2, 3],
):
    """
    Prepare daily water-level and rainfall features.

    Missing rainfall remains missing and is not automatically
    assumed to mean zero precipitation.
    """

    required = [
        "WL_mean",
        "Rainfall_mm",
    ]

    missing = [
        col for col in required
        if col not in daily_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required daily columns: {missing}"
        )

    data = daily_df.copy().sort_index()

    data["WL_mean"] = pd.to_numeric(
        data["WL_mean"],
        errors="coerce",
    )

    data["Rainfall_mm"] = pd.to_numeric(
        data["Rainfall_mm"],
        errors="coerce",
    )

    data = add_cyclical_calendar_features(data)

    water_level = data["WL_mean"]
    rainfall = data["Rainfall_mm"]

    # Daily lags
    for lag in [1, 2, 3, 7, 14]:
        data[f"wl_lag_{lag}d"] = (
            water_level.shift(lag)
        )

        data[f"rain_lag_{lag}d"] = (
            rainfall.shift(lag)
        )

    # Rolling daily features
    for window in [3, 7, 14]:
        data[f"rain_sum_{window}d"] = (
            rainfall.rolling(
                window=window,
                min_periods=window,
            ).sum()
        )

        data[f"wl_mean_{window}d"] = (
            water_level.rolling(
                window=window,
                min_periods=window,
            ).mean()
        )

        data[f"wl_max_{window}d"] = (
            water_level.rolling(
                window=window,
                min_periods=window,
            ).max()
        )

    data["rate_of_rise_1d"] = (
        water_level - data["wl_lag_1d"]
    )

    data["rate_of_rise_3d"] = (
        water_level - data["wl_lag_3d"]
    ) / 3.0

    # Daily future targets
    wl_targets = []
    rain_targets = []

    for horizon in horizons:
        wl_target = f"target_wl_{horizon}d"
        rain_target = f"target_rain_{horizon}d"

        data[wl_target] = water_level.shift(-horizon)
        data[rain_target] = rainfall.shift(-horizon)

        wl_targets.append(wl_target)
        rain_targets.append(rain_target)

    excluded_cols = set(
        wl_targets
        + rain_targets
        + [
            "WL_mean",
            "WL_min",
            "WL_max",
            "WL_last",
            "Rainfall_mm",
        ]
    )

    feature_cols = [
        col for col in data.columns
        if col not in excluded_cols
        and not col.startswith("target_")
    ]

    required_cols = (
        feature_cols + wl_targets + rain_targets
    )

    clean_data = (
        data.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required_cols)
        .copy()
    )

    return (
        clean_data,
        feature_cols,
        wl_targets,
        rain_targets,
    )


# =========================================================
# TEST FEATURE ENGINEERING
# =========================================================

if __name__ == "__main__":

    from .data_loader import (
        create_master_hourly_dataset,
        create_master_daily_dataset,
    )

    print("=" * 60)
    print("TESTING FEATURE ENGINEERING")
    print("=" * 60)

    # Daily feature test
    try:
        daily = create_master_daily_dataset()

        clean_daily, daily_features, daily_wl_targets, daily_rain_targets = (
            prepare_daily_features(daily)
        )

        print("\nDaily dataset:")
        print("Valid samples:", len(clean_daily))
        print("Feature count:", len(daily_features))
        print("Water-level targets:", daily_wl_targets)
        print("Rainfall targets:", daily_rain_targets)

    except Exception as exc:
        print("\nDaily feature test failed:", exc)

    # Hourly feature test
    try:
        hourly = create_master_hourly_dataset()

        print("\nHourly source dataset:", hourly.shape)

        clean_rain, rain_features, rain_targets = (
            prepare_stage1_rainfall_features(hourly)
        )

        print("\nStage 1 rainfall features:")
        print("Valid samples:", len(clean_rain))
        print("Feature count:", len(rain_features))
        print("Targets:", rain_targets)

    except Exception as exc:
        print("\nStage 1 rainfall feature test:", exc)

    try:
        hourly = create_master_hourly_dataset()

        # Stage 2 can only use rows with valid rainfall features
        # because rainfall is an explicit predictor.
        clean_wl, wl_features, wl_targets = (
            prepare_stage2_water_level_features(hourly)
        )

        print("\nStage 2 water-level features:")
        print("Valid samples:", len(clean_wl))
        print("Feature count:", len(wl_features))
        print("Targets:", wl_targets)

    except Exception as exc:
        print("\nStage 2 water-level feature test:", exc)

    print("\nFeature engineering test finished.")

