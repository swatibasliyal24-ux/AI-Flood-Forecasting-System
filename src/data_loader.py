
"""
Data Loader & Preprocessing Module
AI-Based Multi-Stage Flood Forecasting and Early Warning System
Target: Almatti Reservoir (Krishna Basin, Lat: 16.33 N, Lon: 75.88 E)
"""

import os
import pandas as pd
import numpy as np


# =========================================================
# ALMATTI RESERVOIR METADATA
# =========================================================

ALMATTI_METADATA = {
    "name": "Almatti Reservoir (Lal Bahadur Shastri Dam)",
    "river": "Krishna River",
    "basin": "Krishna Basin",
    "state": "Karnataka",
    "district": "Bagalkot / Vijayapura",
    "latitude": 16.33,
    "longitude": 75.88,
    "bed_level_m": 490.0,
    "mddl_m": 506.87,
    "spillway_crest_m": 509.0,
    "warning_level_m": 517.0,
    "danger_level_m": 519.0,
    "frl_m": 519.60,
    "mwl_m": 520.00,
    "gross_capacity_tmc": 130.3,
    "live_capacity_tmc": 123.08,
    "catchment_area_sqkm": 35925,
}


# =========================================================
# KRISHNA BASIN MONITORING STATIONS
# =========================================================

BASIN_STATIONS = {
    "Almatti Reservoir": {
        "lat": 16.33,
        "lon": 75.88,
        "role": "Target Reservoir",
        "type": "Dam",
    },
    "KURUNDWAD": {
        "lat": 16.6844,
        "lon": 74.6022,
        "role": "Upstream Main Krishna",
        "type": "River Gauge",
    },
    "Cholachagudda": {
        "lat": 15.8722,
        "lon": 75.7233,
        "role": "Upstream Malaprabha Tributary",
        "type": "River Gauge",
    },
    "Koyna": {
        "lat": 17.40,
        "lon": 73.75,
        "role": "Far Upstream Dam (Western Ghats)",
        "type": "Dam",
    },
    "Sadalga (Seasonal)": {
        "lat": 16.5636,
        "lon": 74.5275,
        "role": "Upstream Dudhganga Tributary",
        "type": "River Gauge",
    },
    "Narayanapur Reservoir": {
        "lat": 16.20,
        "lon": 76.36,
        "role": "Immediate Downstream Dam",
        "type": "Dam",
    },
    "Deosugur": {
        "lat": 16.3822,
        "lon": 77.3572,
        "role": "Downstream River Reach",
        "type": "River Gauge",
    },
    "P.D. Jurala": {
        "lat": 16.33,
        "lon": 77.70,
        "role": "Downstream Reservoir (Telangana)",
        "type": "Dam",
    },
    "Srisailam Reservoir": {
        "lat": 16.08,
        "lon": 78.90,
        "role": "Downstream Major Reservoir",
        "type": "Dam",
    },
    "Nagarjuna Sagar Dam": {
        "lat": 16.50,
        "lon": 79.25,
        "role": "Downstream Mega Dam",
        "type": "Dam",
    },
}


# =========================================================
# DATA DIRECTORY
# =========================================================

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)


# =========================================================
# LOAD CWC WATER LEVEL DATA
# =========================================================

def load_cwc_water_level(
    filepath=None,
    station="Almatti Reservoir",
):
    """
    Load and clean CWC water-level telemetry.

    Invalid Almatti sensor readings outside the configured
    physical range are discarded.
    """

    if filepath is None:
        filepath = os.path.join(
            DATA_DIR,
            "4ddede56-5e49-4956-bb6f-df3786bf601f.csv",
        )

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Water level file not found: {filepath}"
        )

    df = pd.read_csv(filepath)

    if "Station" not in df.columns:
        raise ValueError(
            "Water-level CSV does not contain the 'Station' column."
        )

    if station:
        df = df[
            df["Station"].astype(str).str.strip() == station
        ].copy()

    time_col = "Data Acquisition Time"
    level_col = "River Water Level Telemetry Hourly (meter)"

    if time_col not in df.columns or level_col not in df.columns:
        raise ValueError(
            "Water-level CSV is missing required columns."
        )

    df["DateTime"] = pd.to_datetime(
        df[time_col],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )

    df["Water_Level_m"] = pd.to_numeric(
        df[level_col],
        errors="coerce",
    )

    df = df.dropna(
        subset=["DateTime", "Water_Level_m"]
    ).sort_values("DateTime")

    # Apply Almatti-specific physical range validation.
    if station == "Almatti Reservoir":
        valid = (
            (df["Water_Level_m"] >= 490.0)
            & (df["Water_Level_m"] <= 525.0)
        )
        df = df.loc[valid].copy()

    df = df.drop_duplicates(
        subset=["DateTime"],
        keep="last",
    )

    df = df.set_index("DateTime").sort_index()

    return df


# =========================================================
# LOAD NASA POWER DAILY RAINFALL
# =========================================================

def load_nasa_rainfall(filepath=None):
    """
    Load NASA POWER daily rainfall data for Almatti.

    NASA daily rainfall remains daily data. It must not be
    divided by 24 and treated as measured hourly rainfall.
    """

    if filepath is None:
        filepath = os.path.join(
            DATA_DIR,
            "almatti_nasa_rainfall_clean.csv",
        )

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)

    else:
        raw_path = os.path.join(
            DATA_DIR,
            "POWER_Point_Daily_20210101_20241231_016d33N_075d88E_LST.csv",
        )

        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                "NASA rainfall file not found. Expected either "
                f"{filepath} or {raw_path}"
            )

        with open(raw_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        header_indices = [
            i for i, line in enumerate(lines)
            if "YEAR,DOY" in line
        ]

        if not header_indices:
            raise ValueError(
                "Could not locate the YEAR,DOY header in the NASA file."
            )

        df = pd.read_csv(
            raw_path,
            skiprows=header_indices[0],
        )

        required = {"YEAR", "DOY", "PRECTOTCORR"}

        if not required.issubset(df.columns):
            raise ValueError(
                "NASA file does not contain YEAR, DOY and PRECTOTCORR."
            )

        df["Date"] = pd.to_datetime(
            df["YEAR"].astype(str)
            + df["DOY"].astype(str).str.zfill(3),
            format="%Y%j",
            errors="coerce",
        )

        df = df.rename(
            columns={"PRECTOTCORR": "Rainfall_mm"}
        )

        df = df[["Date", "Rainfall_mm"]].copy()

        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(filepath, index=False)

    if "Date" not in df.columns or "Rainfall_mm" not in df.columns:
        raise ValueError(
            "NASA rainfall CSV must contain Date and Rainfall_mm columns."
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    ).dt.normalize()

    df["Rainfall_mm"] = pd.to_numeric(
        df["Rainfall_mm"],
        errors="coerce",
    )

    df = df.dropna(subset=["Date"])
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.set_index("Date").sort_index()

    # Negative rainfall values are invalid.
    df.loc[df["Rainfall_mm"] < 0, "Rainfall_mm"] = np.nan

    return df[["Rainfall_mm"]]


# =========================================================
# LOAD CWC HOURLY RAINFALL
# =========================================================

def load_cwc_rainfall(filepath=None):
    """
    Load CWC rainfall telemetry.

    Missing observations remain missing; they are not
    automatically interpreted as zero rainfall.
    """

    if filepath is None:
        filepath = os.path.join(
            DATA_DIR,
            "1d439909-5a54-4fb5-9bda-869b63b9438d.csv",
        )

    if not os.path.exists(filepath):
        return pd.DataFrame(
            columns=[
                "Station",
                "DateTime",
                "Rainfall_Hourly_mm",
            ]
        )

    df = pd.read_csv(filepath)

    required = {
        "Station",
        "Data Acquisition Time",
        "Telemetry Hourly Rainfall (mm)",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            "CWC rainfall CSV is missing required columns."
        )

    df["DateTime"] = pd.to_datetime(
        df["Data Acquisition Time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )

    df["Rainfall_Hourly_mm"] = pd.to_numeric(
        df["Telemetry Hourly Rainfall (mm)"],
        errors="coerce",
    )

    df = df.dropna(subset=["DateTime"])

    # Negative values are invalid, but zero is a valid observation.
    df.loc[
        df["Rainfall_Hourly_mm"] < 0,
        "Rainfall_Hourly_mm",
    ] = np.nan

    df = df.sort_values("DateTime")

    return df


# =========================================================
# CREATE DAILY MASTER DATASET
# =========================================================

def create_master_daily_dataset():
    """
    Combine daily Almatti water-level statistics with
    NASA POWER daily rainfall.

    Missing water-level dates are interpolated.
    Missing rainfall remains missing rather than being
    automatically converted to zero.
    """

    wl_df = load_cwc_water_level(
        station="Almatti Reservoir"
    )

    nasa_rain = load_nasa_rainfall()

    daily_wl = (
        wl_df[["Water_Level_m"]]
        .resample("1D")
        .agg(
            WL_mean=("Water_Level_m", "mean"),
            WL_min=("Water_Level_m", "min"),
            WL_max=("Water_Level_m", "max"),
            WL_last=("Water_Level_m", "last"),
        )
    )

    if daily_wl.empty:
        raise ValueError(
            "No valid Almatti water-level observations were found."
        )

    full_idx = pd.date_range(
        start=min(
            pd.Timestamp("2021-01-01"),
            daily_wl.index.min(),
        ),
        end=max(
            pd.Timestamp("2024-12-31"),
            daily_wl.index.max(),
        ),
        freq="D",
    )

    daily_wl = daily_wl.reindex(full_idx)

    for col in [
        "WL_mean",
        "WL_min",
        "WL_max",
        "WL_last",
    ]:
        daily_wl[col] = daily_wl[col].interpolate(
            method="time",
            limit=30,
            limit_area="inside",
        )

    merged = daily_wl.join(
        nasa_rain,
        how="left",
    )

    merged.index.name = "Date"

    return merged


# =========================================================
# CREATE HOURLY MASTER DATASET
# =========================================================




def create_master_hourly_dataset():
    """
    Build an hourly dataset using CWC water levels and
    hourly rainfall observations from available stations.

    NASA daily rainfall is kept separate; it is never
    divided by 24 to fabricate hourly observations.
    """

    # 1. Almatti water level
    wl_df = load_cwc_water_level(
        station="Almatti Reservoir"
    )

    if wl_df.empty:
        raise ValueError("No Almatti water-level data found.")

    hourly_wl = (
        wl_df[["Water_Level_m"]]
        .resample("1h")
        .mean()
    )

    # 2. Upstream water level
    try:
        kw_df = load_cwc_water_level(
            station="KURUNDWAD"
        )

        if not kw_df.empty:
            kw_hourly = (
                kw_df[["Water_Level_m"]]
                .resample("1h")
                .mean()
                .rename(
                    columns={
                        "Water_Level_m": "WL_Kurundwad_m"
                    }
                )
            )
            hourly_wl = hourly_wl.join(
                kw_hourly, how="left"
            )
        else:
            hourly_wl["WL_Kurundwad_m"] = np.nan

    except Exception:
        hourly_wl["WL_Kurundwad_m"] = np.nan

    # 3. NASA daily rainfall, kept as a separate variable
    nasa = load_nasa_rainfall().copy()
    nasa.index = pd.to_datetime(nasa.index).normalize()
    nasa = nasa[
        ~nasa.index.duplicated(keep="last")
    ]

    daily_map = nasa["Rainfall_mm"].to_dict()

    hourly_wl["Rainfall_Daily_mm"] = [
        daily_map.get(ts.normalize(), np.nan)
        for ts in hourly_wl.index
    ]

    # 4. Initialize station rainfall columns
    rainfall_stations = [
        "Almatti Reservoir",
        "Cholachagudda",
        "Mudhol",
        "Narayanapur Reservoir",
    ]

    for station in rainfall_stations:
        col = (
            "Rainfall_Hourly_mm"
            if station == "Almatti Reservoir"
            else f"Rain_{station.replace(' ', '_')}_mm"
        )
        hourly_wl[col] = np.nan

    # 5. Load and align CWC hourly rainfall by station
    cwc = load_cwc_rainfall()

    if not cwc.empty:
        cwc["Station"] = (
            cwc["Station"].astype(str).str.strip()
        )

        cwc["Rainfall_Hourly_mm"] = pd.to_numeric(
            cwc["Rainfall_Hourly_mm"],
            errors="coerce",
        )

        cwc.loc[
            cwc["Rainfall_Hourly_mm"] < 0,
            "Rainfall_Hourly_mm",
        ] = np.nan

        for station in rainfall_stations:
            station_data = cwc[
                cwc["Station"] == station
            ].copy()

            if station_data.empty:
                continue

            station_data = station_data.dropna(
                subset=["DateTime"]
            ).drop_duplicates(
                subset=["DateTime"],
                keep="last",
            )

            hourly_rain = (
                station_data
                .set_index("DateTime")["Rainfall_Hourly_mm"]
                .sort_index()
                .resample("1h")
                .mean()
            )

            col = (
                "Rainfall_Hourly_mm"
                if station == "Almatti Reservoir"
                else f"Rain_{station.replace(' ', '_')}_mm"
            )

            hourly_wl[col] = hourly_rain.reindex(
                hourly_wl.index
            )

    # 6. Overlay clean ECMWF hourly meteorological data (matching Google Weather model)
    ecmwf_weather = load_almatti_weather_dataset()
    if not ecmwf_weather.empty:
        for c in ["Temperature_C", "Humidity_pct", "Pressure_hPa", "WindSpeed_kmh"]:
            if c in ecmwf_weather.columns:
                hourly_wl[c] = ecmwf_weather[c].reindex(hourly_wl.index)
        if "Precipitation_mm" in ecmwf_weather.columns:
            # Clean, non-corrupted hourly precipitation
            clean_p = ecmwf_weather["Precipitation_mm"].reindex(hourly_wl.index)
            hourly_wl["Rainfall_Hourly_mm"] = clean_p.combine_first(hourly_wl["Rainfall_Hourly_mm"]).fillna(0.0)

    # 7. Interpolate only short gaps in water level
    hourly_wl["Water_Level_m"] = (
        hourly_wl["Water_Level_m"].interpolate(
            method="time",
            limit=6,
            limit_area="inside",
        )
    )

    hourly_wl.index.name = "DateTime"

    return hourly_wl


def load_almatti_weather_dataset(filepath=None):
    """
    Load clean ECMWF/ERA5 hourly meteorological dataset for Almatti (Lat 16.33, Lon 75.88).
    This dataset aligns with Google Weather's underlying ECMWF IFS forecast models.
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "almatti_ecmwf_hourly_weather.csv")

    if not os.path.exists(filepath):
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"]).drop_duplicates(subset=["DateTime"])
    df = df.set_index("DateTime").sort_index()
    return df


   


# =========================================================
# BASIC DATASET TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING ALMATTI FLOOD FORECAST DATA LOADER")
    print("=" * 60)

    try:
        daily = create_master_daily_dataset()

        print("\nDaily dataset shape:", daily.shape)
        print(daily.head(3))

        hourly = create_master_hourly_dataset()

        print("\nHourly dataset shape:", hourly.shape)

        print(
            "Valid water-level observations:",
            hourly["Water_Level_m"].notna().sum(),
        )

        print(
            "Available hourly rainfall observations:",
            hourly["Rainfall_Hourly_mm"].notna().sum(),
        )

        print(
            "Available NASA daily rainfall values:",
            hourly["Rainfall_Daily_mm"].notna().sum(),
        )

        print("\nHourly rainfall summary:")
        print(
            hourly["Rainfall_Hourly_mm"].describe()
        )

        print("\nFirst five hourly records:")
        print(hourly.head())

        print("\nData loader test completed.")

    except Exception as exc:
        print("\nDATA LOADER TEST FAILED:")
        print(type(exc).__name__, "-", exc)
        raise

