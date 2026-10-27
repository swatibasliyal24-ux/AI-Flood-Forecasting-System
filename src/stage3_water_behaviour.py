"""
Stage 3: Water Behaviour Analysis Module
Hydraulic Depth, Rate of Rise, Velocity, Flow Direction, and Flood Spread Modeling
Target: Almatti Reservoir (Lat: 16.33 N, Lon: 75.88 E)
Krishna River Basin
"""

import numpy as np
import pandas as pd
from .data_loader import ALMATTI_METADATA


class WaterBehaviourAnalyzer:
    """
    Hydraulic and geometric behavior analysis engine for Almatti Reservoir & Krishna River.
    Computes depth, rate of rise, volumetric dynamics, flow velocity, direction,
    and flood spread inundation surface area.
    """

    def __init__(self):
        self.bed_level = ALMATTI_METADATA["bed_level_m"]
        self.mddl = ALMATTI_METADATA["mddl_m"]
        self.warning_level = ALMATTI_METADATA["warning_level_m"]
        self.danger_level = ALMATTI_METADATA["danger_level_m"]
        self.frl = ALMATTI_METADATA["frl_m"]
        self.live_capacity_tmc = ALMATTI_METADATA["live_capacity_tmc"]
        self.gross_capacity_tmc = ALMATTI_METADATA["gross_capacity_tmc"]

    def compute_depth(self, water_level_m):
        """
        Total hydraulic water depth above riverbed (meters).
        """
        return max(0.0, round(float(water_level_m) - self.bed_level, 3))

    def compute_rate_of_rise(self, current_wl, previous_wl, dt_hours=1.0):
        """
        Rate of water-level change in centimeters per hour (cm/h).
        Positive = Rising (inflow dominant), Negative = Falling (receding/releasing).
        """
        if dt_hours <= 0:
            dt_hours = 1.0
        delta_m = current_wl - previous_wl
        rate_cm_per_h = (delta_m / dt_hours) * 100.0
        return round(float(rate_cm_per_h), 2)

    def estimate_storage_volume(self, water_level_m):
        """
        Estimate reservoir stored volume in TMC (Thousand Million Cubic Feet)
        based on Almatti stage-storage relationship.
        """
        wl = max(self.bed_level, min(float(water_level_m), 522.0))
        if wl <= self.mddl:
            # Dead storage only (approx 7.2 TMC)
            fraction = max(0.0, (wl - self.bed_level) / (self.mddl - self.bed_level))
            storage = 7.22 * (fraction ** 1.8)
        else:
            # Dead storage + parabolic live storage
            head = wl - self.mddl
            max_head = self.frl - self.mddl
            fraction = min(1.2, head / max_head)
            storage = 7.22 + self.live_capacity_tmc * (fraction ** 2.1)
        return round(float(storage), 2)

    def estimate_flood_spread_area(self, water_level_m):
        """
        Estimate reservoir water spread area in square kilometers (sq km).
        Almatti water spread:
        - At 510m: ~180 sq km
        - At 515m: ~310 sq km
        - At Warning 517m: ~390 sq km
        - At FRL 519.60m: ~488 sq km
        - At Surcharge > 519.60m: spreads over adjacent low-lying riverbanks (+70 sq km/m)
        """
        wl = float(water_level_m)
        if wl <= self.mddl:
            spread = 120.0 * max(0.1, (wl - self.bed_level) / (self.mddl - self.bed_level))
        elif wl <= self.frl:
            fraction = (wl - self.mddl) / (self.frl - self.mddl)
            spread = 120.0 + (488.0 - 120.0) * (fraction ** 1.6)
        else:
            # Surcharge inundation beyond FRL
            surcharge = wl - self.frl
            spread = 488.0 + surcharge * 75.0
        return round(float(spread), 1)

    def compute_flow_dynamics(self, current_wl, prev_wl, dt_hours=1.0, rain_24h_mm=0.0):
        """
        Comprehensive water behaviour analysis:
        Computes depth, velocity, flow direction, net flow status, and flood wave travel time.
        """
        depth = self.compute_depth(current_wl)
        rate_cm_h = self.compute_rate_of_rise(current_wl, prev_wl, dt_hours)
        storage_tmc = self.estimate_storage_volume(current_wl)
        capacity_pct = round((storage_tmc / self.gross_capacity_tmc) * 100.0, 1)
        spread_sqkm = self.estimate_flood_spread_area(current_wl)

        # Flow velocity proxy in river channel (m/s) using Manning's hydraulic approximation
        # Base velocity 0.4 m/s at low stages, up to 2.5 m/s during heavy runoff
        hydraulic_radius = max(1.0, depth * 0.45)
        energy_slope = 0.00045  # Mean Krishna River bed gradient in this stretch
        roughness_n = 0.035     # Natural river channel with weeds & gravel
        channel_velocity_ms = (1.0 / roughness_n) * (hydraulic_radius ** (2/3)) * (energy_slope ** 0.5)

        # Inflow surge acceleration if heavy upstream rain or fast rate of rise
        if rate_cm_h > 2.0:
            channel_velocity_ms += (rate_cm_h / 10.0) * 0.25
        channel_velocity_ms = min(3.5, max(0.2, channel_velocity_ms))

        # Flow direction
        flow_direction = {
            "bearing_degrees": 115,
            "direction_cardinal": "East-Southeast (ESE)",
            "upstream_source": "Western Ghats & Maharashtra Krishna (Kurundwad / Koyna)",
            "downstream_sink": "Narayanapur Reservoir (Basava Sagara Dam, ~65 km downstream)"
        }

        # Flood wave travel time to downstream Narayanapur Reservoir
        distance_km = 65.0
        wave_speed_kmh = channel_velocity_ms * 3.6 * 1.35  # Dynamic wave celerity ~ 1.35x velocity
        travel_time_hours = round(distance_km / max(1.0, wave_speed_kmh), 1)

        # Surge state classification
        if rate_cm_h > 5.0:
            surge_status = "Rapid Inflow Surge (Flash Flood Risk)"
            surge_severity = "Critical"
        elif rate_cm_h > 1.5:
            surge_status = "Moderate Rising Trend (Monsoon Influx)"
            surge_severity = "Warning"
        elif rate_cm_h < -1.5:
            surge_status = "Active Reservoir Drawdown / Outflow Release"
            surge_severity = "Receding"
        else:
            surge_status = "Hydraulically Steady State"
            surge_severity = "Stable"

        return {
            "water_level_m": round(float(current_wl), 3),
            "water_depth_m": depth,
            "rate_of_rise_cm_per_hour": rate_cm_h,
            "storage_volume_tmc": storage_tmc,
            "storage_capacity_percent": min(100.0, capacity_pct),
            "water_spread_area_sqkm": spread_sqkm,
            "channel_flow_velocity_ms": round(float(channel_velocity_ms), 2),
            "flow_direction": flow_direction,
            "flood_wave_travel_time_hours": travel_time_hours,
            "surge_status": surge_status,
            "surge_severity": surge_severity
        }


if __name__ == "__main__":
    analyzer = WaterBehaviourAnalyzer()
    res = analyzer.compute_flow_dynamics(518.5, 518.2, dt_hours=6.0, rain_24h_mm=25.0)
    for k, v in res.items():
        print(f"{k}: {v}")
