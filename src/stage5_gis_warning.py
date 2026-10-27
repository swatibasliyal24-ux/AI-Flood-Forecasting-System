"""
Stage 5: GIS Mapping & Early Warning Module
Interactive Leaflet/Folium GIS Visualization with Low, Medium, and High Risk Zones,
Inundation Buffers, Krishna River Network, Hotspot Sectors, and Early Warning Generator.
Target: Almatti Reservoir (Lat: 16.33 N, Lon: 75.88 E)
GNIT / AKTU Academic Mini Project
"""

import folium
from folium import plugins
from .data_loader import ALMATTI_METADATA, BASIN_STATIONS


# Krishna River geometry (centerline coordinates from Western Ghats to Bay of Bengal)
KRISHNA_RIVER_COORDINATES = [
    [17.40, 73.75],     # Koyna Dam
    [16.85, 74.30],     # Sangli
    [16.6844, 74.6022], # Kurundwad (Maharashtra-Karnataka border)
    [16.56, 75.10],     # Jamkhandi upstream
    [16.45, 75.50],     # Bilagi reach
    [16.33, 75.88],     # ALMATTI DAM (TARGET)
    [16.25, 76.10],     # Nidagundi / Krishna downstream
    [16.20, 76.36],     # Narayanapur Reservoir (Basava Sagara Dam)
    [16.38, 77.35],     # Deosugur
    [16.33, 77.70],     # P.D. Jurala Dam (Telangana border)
    [16.08, 78.90],     # Srisailam Dam
    [16.50, 79.25],     # Nagarjuna Sagar Dam
    [16.50, 80.60],     # Prakasam Barrage (Vijayawada)
    [15.80, 80.85]      # Krishna Delta / Bay of Bengal
]

# -------------------------------------------------------------------------
# Hydrological Risk Zonal Polygons for Almatti Reservoir and Krishna Basin
# -------------------------------------------------------------------------

# 1. LOW RISK ZONE (Green): Normal Operating Pool (< 517.00 m MSL)
# Confined within the core reservoir bed and stable embankments (~350 km² spread)
ALMATTI_LOW_RISK_ZONE = [
    [16.330, 75.880],   # Almatti Dam Wall
    [16.345, 75.855],
    [16.370, 75.810],
    [16.390, 75.750],
    [16.405, 75.680],
    [16.415, 75.590],   # Upstream tail towards Bilagi
    [16.385, 75.570],
    [16.355, 75.630],
    [16.330, 75.690],
    [16.305, 75.750],
    [16.295, 75.810],
    [16.310, 75.855],
    [16.330, 75.880]    # Close polygon
]

# 2. MEDIUM RISK ZONE (Amber/Orange): Warning Surcharge Buffer (517.00 m - 519.00 m MSL)
# Water swells over lower terraces, Bilagi backwater plains, and tributary confluences (~420 km²)
ALMATTI_MEDIUM_RISK_ZONE = [
    [16.330, 75.895],
    [16.360, 75.875],
    [16.395, 75.830],
    [16.430, 75.760],
    [16.450, 75.680],
    [16.465, 75.560],   # Expanding into Bilagi lowlands
    [16.430, 75.500],
    [16.375, 75.550],
    [16.320, 75.640],
    [16.280, 75.720],
    [16.270, 75.800],
    [16.290, 75.860],
    [16.330, 75.895]
]

# 3. HIGH RISK ZONE (Red): Critical Flood Inundation Buffer (>= 519.00 m MSL to 519.60 m FRL & above)
# Extreme flood expansion reaching Jamkhandi floodplain, north Bilagi farmlands, and Ghataprabha confluence (~510+ km²)
ALMATTI_HIGH_RISK_ZONE = [
    [16.330, 75.915],
    [16.380, 75.895],
    [16.425, 75.845],
    [16.470, 75.770],
    [16.495, 75.670],
    [16.515, 75.480],   # North-west expansion reaching Jamkhandi floodplain
    [16.485, 75.360],   # Upper Krishna riverside floodplain
    [16.430, 75.450],
    [16.360, 75.520],
    [16.290, 75.610],
    [16.240, 75.690],   # South towards Ghataprabha/Bagalkot confluence
    [16.230, 75.780],
    [16.260, 75.850],
    [16.330, 75.915]
]

# Backward-compatibility alias
ALMATTI_WATER_SPREAD_POLYGON = ALMATTI_LOW_RISK_ZONE
ALMATTI_FLOOD_INUNDATION_BUFFER = ALMATTI_HIGH_RISK_ZONE


# -------------------------------------------------------------------------
# Critical Hotspot Locations Categorized by Risk Tier
# -------------------------------------------------------------------------
RISK_HOTSPOTS = [
    # --- HIGH RISK HOTSPOTS (Red) ---
    {
        "name": "Jamkhandi Krishna Floodplain",
        "category": "High Risk",
        "lat": 16.510,
        "lon": 75.310,
        "wl_threshold": ">= 519.00 m MSL (CWC Danger Level)",
        "threat": "Critical riverbank overflow; road links & low-lying agrarian lands submerged.",
        "action": "Immediate evacuation of riverside settlements to relief camps; suspend ferry and bridge traffic.",
        "color": "#ef4444",
        "badge_color": "#fee2e2",
        "border_color": "#b91c1c",
        "icon": "exclamation-triangle"
    },
    {
        "name": "Bilagi Backwater Flood Zone",
        "category": "High Risk",
        "lat": 16.350,
        "lon": 75.620,
        "wl_threshold": ">= 518.50 m MSL",
        "threat": "Almatti backwater surcharge submerges lowlands and village access embankments.",
        "action": "Issue Red Alert to taluk administration; mobilize SDRF rescue boats.",
        "color": "#ef4444",
        "badge_color": "#fee2e2",
        "border_color": "#b91c1c",
        "icon": "exclamation-triangle"
    },
    {
        "name": "Korti-Kolhar Submerged Reach",
        "category": "High Risk",
        "lat": 16.390,
        "lon": 75.710,
        "wl_threshold": ">= 518.80 m MSL",
        "threat": "Extensive backwater spreading into sugarcane farmlands and settlement approaches.",
        "action": "Deploy pump drainage and seal embankment sluice gates.",
        "color": "#ef4444",
        "badge_color": "#fee2e2",
        "border_color": "#b91c1c",
        "icon": "exclamation-triangle"
    },
    {
        "name": "Bagalkot Confluence Lowlands",
        "category": "High Risk",
        "lat": 16.190,
        "lon": 75.700,
        "wl_threshold": ">= 519.20 m MSL",
        "threat": "Backwater surge at Ghataprabha-Krishna confluence causing backflow flooding in low sectors.",
        "action": "Activate Bagalkot town flood protection walls and round-the-clock monitoring.",
        "color": "#ef4444",
        "badge_color": "#fee2e2",
        "border_color": "#b91c1c",
        "icon": "exclamation-triangle"
    },

    # --- MEDIUM RISK HOTSPOTS (Amber/Orange) ---
    {
        "name": "Galagali Krishna Riverbank Reach",
        "category": "Medium Risk",
        "lat": 16.380,
        "lon": 75.450,
        "wl_threshold": "517.00 - 519.00 m MSL (CWC Warning Level)",
        "threat": "Riverbank water levels approaching protective bunds and low-level culverts.",
        "action": "Orange alert advisory; restrict public access to riverbank; secure cattle and farm machinery.",
        "color": "#f59e0b",
        "badge_color": "#fef3c7",
        "border_color": "#d97706",
        "icon": "exclamation-circle"
    },
    {
        "name": "Nidagundi Spillway Discharge Corridor",
        "category": "Medium Risk",
        "lat": 16.360,
        "lon": 75.930,
        "wl_threshold": "517.50 m MSL",
        "threat": "High water discharge velocity in downstream channel upon radial gate opening.",
        "action": "Sound siren warnings along downstream banks; warn sandbed and fishing communities.",
        "color": "#f59e0b",
        "badge_color": "#fef3c7",
        "border_color": "#d97706",
        "icon": "exclamation-circle"
    },
    {
        "name": "Hungund Riverbank Buffer Reach",
        "category": "Medium Risk",
        "lat": 16.060,
        "lon": 76.060,
        "wl_threshold": "517.20 m MSL",
        "threat": "Moderate backflow swell in tributary streams draining into Krishna basin.",
        "action": "Standby vigil on irrigation canal headworks and drainage culverts.",
        "color": "#f59e0b",
        "badge_color": "#fef3c7",
        "border_color": "#d97706",
        "icon": "exclamation-circle"
    },

    # --- LOW RISK HOTSPOTS (Green) ---
    {
        "name": "Almatti Dam Operations HQ & Crest",
        "category": "Low Risk",
        "lat": 16.330,
        "lon": 75.880,
        "wl_threshold": "< 517.00 m MSL (Safe Operating Band)",
        "threat": "Normal operational reservoir storage; zero structural or inundation threat.",
        "action": "Standard CWC telemetry synchronization and routine hydropower scheduling.",
        "color": "#10b981",
        "badge_color": "#d1fae5",
        "border_color": "#047857",
        "icon": "check-circle"
    },
    {
        "name": "Mudhol High Ground Refuge Area",
        "category": "Low Risk",
        "lat": 16.340,
        "lon": 75.280,
        "wl_threshold": "Elevated Topography (> 535 m MSL)",
        "threat": "Naturally elevated ground; completely outside backwater inundation limits.",
        "action": "Designated emergency assembly and relief camp staging point.",
        "color": "#10b981",
        "badge_color": "#d1fae5",
        "border_color": "#047857",
        "icon": "check-circle"
    },
    {
        "name": "Sitimani Hydro Electric Station",
        "category": "Low Risk",
        "lat": 16.270,
        "lon": 75.840,
        "wl_threshold": "Stable Inflow (< 517.00 m)",
        "threat": "Operating within safe design penstock and turbine head margins.",
        "action": "Normal power generation dispatch to Karnataka state grid.",
        "color": "#10b981",
        "badge_color": "#d1fae5",
        "border_color": "#047857",
        "icon": "check-circle"
    },
    {
        "name": "Narayanapur Buffer Reservoir Basin",
        "category": "Low Risk",
        "lat": 16.200,
        "lon": 76.360,
        "wl_threshold": "FRL: 492.25 m MSL",
        "threat": "Adequate cushion available to absorb regulated discharges from Almatti spillway.",
        "action": "Coordinated reservoir operations under Krishna Bhagya Jala Nigam Ltd (KBJNL).",
        "color": "#10b981",
        "badge_color": "#d1fae5",
        "border_color": "#047857",
        "icon": "check-circle"
    }
]


def generate_gis_map(risk_info, current_wl=517.5, stage3_data=None, filter_risk="ALL"):
    """
    Creates an interactive Folium GIS map visualizing:
    - Low Risk Zone (Normal operating water spread < 517.0 m MSL in Green)
    - Medium Risk Zone (Warning inundation buffer 517.0 - 519.0 m MSL in Amber)
    - High Risk Zone (Critical flood submergence >= 519.0 m MSL & FRL in Red)
    - Vulnerable taluk & infrastructure hotspots with risk-coded markers & popups
    - Almatti Dam target telemetry with dynamic pulsing status
    - Krishna River centerline & West-to-East flow vectors
    - Upstream/downstream CWC & KBJNL monitoring stations
    - On-map interactive legend and layer controls
    """
    active_category = risk_info.get("category", "Low Risk")
    active_color = risk_info.get("color", "#10b981")
    active_fri = risk_info.get("risk_index", 20.0)

    # Base Map centered at Almatti Reservoir and backwaters
    m = folium.Map(
        location=[16.35, 75.72],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # Base Tile Layers
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap (Standard)",
        control=True
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Satellite",
        name="Esri Satellite Imagery",
        control=True
    ).add_to(m)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True
    ).add_to(m)

    # Feature Groups for Interactive Toggling
    fg_high = folium.FeatureGroup(name="🔴 High Risk Zone (>= 519m MSL | Danger)", show=True)
    fg_med = folium.FeatureGroup(name="🟡 Medium Risk Zone (517 - 519m MSL | Warning)", show=True)
    fg_low = folium.FeatureGroup(name="🟢 Low Risk Zone (< 517m MSL | Normal)", show=True)
    fg_river = folium.FeatureGroup(name="🌊 Krishna River Mainstem & Flow", show=True)
    fg_stations = folium.FeatureGroup(name="📡 Basin Telemetry Network", show=True)

    # Determine which risk tiers to render
    show_high = filter_risk in ("ALL", "HIGH")
    show_med = filter_risk in ("ALL", "MEDIUM")
    show_low = filter_risk in ("ALL", "LOW")

    # 1. HIGH RISK ZONE (Red Polygon & Tooltips)
    if show_high:
        high_popup_html = f"""
        <div style="font-family: sans-serif; min-width: 220px;">
            <div style="background: #fee2e2; border-left: 4px solid #ef4444; padding: 6px 10px; border-radius: 4px;">
                <b style="color: #991b1b; font-size: 13px;">🔴 HIGH RISK INUNDATION ZONE</b>
            </div>
            <p style="margin: 6px 0 2px 0; font-size: 12px;"><b>Threshold:</b> Water Level &ge; 519.00 m MSL (CWC Danger Level)</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>FRL Surcharge:</b> 519.60 m MSL</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>Inundation Spread:</b> ~510+ km&sup2;</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>Affected Sectors:</b> Jamkhandi Floodplain, Bilagi Backwaters, Korti-Kolhar, Bagalkot Confluence</p>
            <div style="margin-top: 6px; padding: 4px 8px; background: #fef2f2; border-radius: 4px; font-size: 11px; color: #b91c1c;">
                <b>Directive:</b> Phased radial gate discharge (10,000 - 50,000 cusecs); mandatory floodplain evacuations.
            </div>
        </div>
        """
        folium.Polygon(
            locations=ALMATTI_HIGH_RISK_ZONE,
            color="#ef4444",
            weight=2.5,
            dash_array="6, 6",
            fill=True,
            fill_color="#ef4444",
            fill_opacity=0.38,
            popup=folium.Popup(high_popup_html, max_width=320),
            tooltip="🔴 High Risk Zone: Critical Surcharge & Inundation Buffer (&ge; 519.00 m MSL)"
        ).add_to(fg_high)

    # 2. MEDIUM RISK ZONE (Amber/Orange Polygon & Tooltips)
    if show_med:
        med_popup_html = f"""
        <div style="font-family: sans-serif; min-width: 220px;">
            <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 6px 10px; border-radius: 4px;">
                <b style="color: #92400e; font-size: 13px;">🟡 MEDIUM RISK WARNING BUFFER</b>
            </div>
            <p style="margin: 6px 0 2px 0; font-size: 12px;"><b>Threshold:</b> Water Level 517.00 - 519.00 m MSL (CWC Warning Level)</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>Storage Buffer to FRL:</b> 0.60 m - 2.60 m</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>Water Spread:</b> ~420 km&sup2;</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>Affected Sectors:</b> Bilagi lowlands, Galagali, Nidagundi canal headworks</p>
            <div style="margin-top: 6px; padding: 4px 8px; background: #fffbeb; border-radius: 4px; font-size: 11px; color: #92400e;">
                <b>Directive:</b> Place 26 radial spillway gates on 24x7 standby; alert riverside settlements.
            </div>
        </div>
        """
        folium.Polygon(
            locations=ALMATTI_MEDIUM_RISK_ZONE,
            color="#f59e0b",
            weight=2.0,
            dash_array="5, 5",
            fill=True,
            fill_color="#f59e0b",
            fill_opacity=0.32,
            popup=folium.Popup(med_popup_html, max_width=320),
            tooltip="🟡 Medium Risk Zone: 1st Warning Level Buffer (517.00 - 519.00 m MSL)"
        ).add_to(fg_med)

    # 3. LOW RISK ZONE (Green Polygon & Tooltips)
    if show_low:
        low_popup_html = f"""
        <div style="font-family: sans-serif; min-width: 220px;">
            <div style="background: #d1fae5; border-left: 4px solid #10b981; padding: 6px 10px; border-radius: 4px;">
                <b style="color: #065f46; font-size: 13px;">🟢 LOW RISK NORMAL BASIN</b>
            </div>
            <p style="margin: 6px 0 2px 0; font-size: 12px;"><b>Threshold:</b> Water Level &lt; 517.00 m MSL (Normal Operational Band)</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>Live Storage:</b> &lt; 100 TMC</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>Normal Surface Area:</b> ~350 km&sup2;</p>
            <p style="margin: 2px 0; font-size: 12px;"><b>Status:</b> Confined within safe structural reservoir embankments</p>
            <div style="margin-top: 6px; padding: 4px 8px; background: #ecfdf5; border-radius: 4px; font-size: 11px; color: #065f46;">
                <b>Directive:</b> Routine power generation and irrigation canal discharge schedule.
            </div>
        </div>
        """
        folium.Polygon(
            locations=ALMATTI_LOW_RISK_ZONE,
            color="#10b981",
            weight=2.0,
            fill=True,
            fill_color="#10b981",
            fill_opacity=0.45,
            popup=folium.Popup(low_popup_html, max_width=320),
            tooltip="🟢 Low Risk Zone: Normal Operational Water Spread (&lt; 517.00 m MSL)"
        ).add_to(fg_low)

    # 4. Critical Hotspot Markers for Low, Medium, and High Risk
    for hp in RISK_HOTSPOTS:
        cat = hp["category"]
        if cat == "High Risk" and not show_high:
            continue
        if cat == "Medium Risk" and not show_med:
            continue
        if cat == "Low Risk" and not show_low:
            continue

        target_fg = fg_high if cat == "High Risk" else (fg_med if cat == "Medium Risk" else fg_low)

        hp_html = f"""
        <div style="font-family: sans-serif; min-width: 210px; max-width: 260px;">
            <div style="background: {hp['badge_color']}; border-left: 3px solid {hp['color']}; padding: 4px 8px; border-radius: 4px; margin-bottom: 6px;">
                <b style="color: {hp['color']}; font-size: 12px;">{hp['category'].upper()}: {hp['name']}</b>
            </div>
            <p style="margin: 2px 0; font-size: 11px;"><b>Critical Threshold:</b> {hp['wl_threshold']}</p>
            <p style="margin: 2px 0; font-size: 11px;"><b>Threat:</b> {hp['threat']}</p>
            <div style="margin-top: 6px; font-size: 11px; padding: 4px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px;">
                <b>Action:</b> {hp['action']}
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[hp["lat"], hp["lon"]],
            radius=7,
            color=hp["border_color"],
            fill=True,
            fill_color=hp["color"],
            fill_opacity=0.90,
            weight=2,
            popup=folium.Popup(hp_html, max_width=280),
            tooltip=f"{'🔴' if cat == 'High Risk' else ('🟡' if cat == 'Medium Risk' else '🟢')} {hp['name']} ({cat})"
        ).add_to(target_fg)

    # 5. Almatti Dam Target Pin with Pulsing Live Status Circle
    dam_html = f"""
    <div style="font-family: sans-serif; min-width: 220px;">
        <h4 style="margin: 0 0 5px 0; color: #0f172a;">{ALMATTI_METADATA['name']}</h4>
        <div style="background: {active_color}; color: white; padding: 2px 8px; border-radius: 4px; display: inline-block; font-size: 11px; font-weight: bold; margin-bottom: 6px;">
            {active_category.upper()} (Risk Index: {active_fri}/100)
        </div>
        <p style="margin: 2px 0; font-size: 12px;"><b>Current Water Level:</b> <span style="font-weight: bold; color: {active_color};">{current_wl:.2f} m MSL</span></p>
        <p style="margin: 2px 0; font-size: 12px;"><b>Full Reservoir Level (FRL):</b> {ALMATTI_METADATA['frl_m']:.2f} m</p>
        <p style="margin: 2px 0; font-size: 12px;"><b>Danger Level:</b> {ALMATTI_METADATA['danger_level_m']:.2f} m | <b>Warning:</b> {ALMATTI_METADATA['warning_level_m']:.2f} m</p>
        <p style="margin: 2px 0; font-size: 12px;"><b>Buffer to FRL:</b> {ALMATTI_METADATA['frl_m'] - current_wl:.2f} m</p>
    </div>
    """
    folium.Marker(
        location=[16.33, 75.88],
        popup=folium.Popup(dam_html, max_width=300),
        tooltip=f"🎯 Almatti Dam Target Reservoir: {current_wl:.2f}m ({active_category})",
        icon=folium.Icon(
            color="red" if active_category == "High Risk" else ("orange" if active_category == "Medium Risk" else "green"),
            icon="tint",
            prefix="fa"
        )
    ).add_to(m)

    # Halo status ring around Almatti
    folium.Circle(
        location=[16.33, 75.88],
        radius=4500,
        color=active_color,
        fill=True,
        fill_color=active_color,
        fill_opacity=0.25,
        weight=2,
        tooltip=f"Almatti Reservoir Status Halo: {active_category}"
    ).add_to(m)

    # 6. Krishna River Flow Path & Downstream Discharge Vector
    folium.PolyLine(
        locations=KRISHNA_RIVER_COORDINATES,
        color="#2563eb",
        weight=4,
        opacity=0.85,
        tooltip="🌊 Krishna River Mainstem (West-to-East Flow into Bay of Bengal)"
    ).add_to(fg_river)

    # Flow vector arrow from Almatti to Narayanapur
    folium.PolyLine(
        locations=[[16.33, 75.88], [16.20, 76.36]],
        color="#dc2626" if active_category == "High Risk" else ("#f97316" if active_category == "Medium Risk" else "#059669"),
        weight=3,
        dash_array="5, 8",
        tooltip="Downstream Regulated Flow Vector &rarr; Narayanapur Reservoir (65 km)"
    ).add_to(fg_river)

    # 7. Upstream and Downstream Krishna Basin Stations
    for st_name, st_info in BASIN_STATIONS.items():
        if st_name == "Almatti Reservoir":
            continue

        lat, lon = st_info["lat"], st_info["lon"]
        role = st_info["role"]
        st_color = "darkblue" if "Downstream" in role else "purple"

        st_html = f"""
        <div style="font-family: sans-serif; min-width: 170px;">
            <h5 style="margin: 0 0 4px 0; color: #1e293b;">{st_name}</h5>
            <p style="margin: 2px 0; font-size: 11px;"><b>Role:</b> {role}</p>
            <p style="margin: 2px 0; font-size: 11px;"><b>Type:</b> {st_info['type']}</p>
            <p style="margin: 2px 0; font-size: 11px;"><b>Coordinates:</b> {lat:.3f}&deg; N, {lon:.3f}&deg; E</p>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color=st_color,
            fill=True,
            fill_color=st_color,
            fill_opacity=0.85,
            weight=1.5,
            popup=folium.Popup(st_html, max_width=250),
            tooltip=f"{st_name} ({role})"
        ).add_to(fg_stations)

    # Add all feature groups to map
    fg_high.add_to(m)
    fg_med.add_to(m)
    fg_low.add_to(m)
    fg_river.add_to(m)
    fg_stations.add_to(m)

    # 8. Floating HTML Legend embedded on Map
    legend_html = """
    <div style="
        position: absolute;
        bottom: 24px;
        right: 18px;
        z-index: 1000;
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 11px;
        min-width: 190px;
        line-height: 1.4;
    ">
        <div style="font-weight: 700; font-size: 12px; color: #0f172a; margin-bottom: 6px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px;">
            🗺️ Flood Risk Classification
        </div>
        <div style="margin: 3px 0; display: flex; align-items: center;">
            <span style="display: inline-block; width: 14px; height: 14px; background: #ef4444; border: 1px solid #b91c1c; border-radius: 3px; margin-right: 8px;"></span>
            <span><b>High Risk:</b> &ge; 519.0 m (Danger)</span>
        </div>
        <div style="margin: 3px 0; display: flex; align-items: center;">
            <span style="display: inline-block; width: 14px; height: 14px; background: #f59e0b; border: 1px solid #d97706; border-radius: 3px; margin-right: 8px;"></span>
            <span><b>Medium Risk:</b> 517 - 519 m (Warning)</span>
        </div>
        <div style="margin: 3px 0; display: flex; align-items: center;">
            <span style="display: inline-block; width: 14px; height: 14px; background: #10b981; border: 1px solid #047857; border-radius: 3px; margin-right: 8px;"></span>
            <span><b>Low Risk:</b> &lt; 517.0 m (Normal)</span>
        </div>
        <div style="margin-top: 6px; padding-top: 5px; border-top: 1px solid #f1f5f9; display: flex; align-items: center;">
            <span style="display: inline-block; width: 16px; height: 3px; background: #2563eb; margin-right: 6px;"></span>
            <span style="color: #475569;">Krishna River Channel</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add Layer Control to allow toggling layers directly on Leaflet
    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    return m


def generate_early_warning_bulletin(risk_info, stage1_rain, stage2_wl, stage3_behaviour, horizon_h=6):
    """
    Constructs an official disaster management early warning bulletin.
    """
    category = risk_info.get("category", "Low Risk")
    fri = risk_info.get("risk_index", 0.0)
    current_wl = stage2_wl.get("current_wl", 516.5)
    pred_wl = stage2_wl.get(horizon_h, current_wl)
    pred_rain = stage1_rain.get(horizon_h, 0.0)

    if category == "High Risk":
        alert_code = "RED_ALERT_STAGE_3"
        headline = "CRITICAL FLOOD WARNING: HIGH RISK SURGE IMMINENT"
        urgency = "IMMEDIATE (Action within 1 - 2 hours)"
        gates_directive = "Open radial crest gates in phased manner (10,000 - 50,000 cusecs) to maintain buffer below 519.60m FRL."
        downstream_directive = "Issue red alert to Narayanapur Reservoir authorities and Bagalkot/Vijayapura district magistrates."
        public_advisory = "Evacuate low-lying floodplains in Bilagi, Jamkhandi, and Mudhol taluks. Prohibit fishing and bridge crossings."
    elif category == "Medium Risk":
        alert_code = "ORANGE_ALERT_STAGE_2"
        headline = "FLOOD ADVISORY: MEDIUM RISK INFLOW ELEVATION"
        urgency = "STANDBY (Monitor hourly updates)"
        gates_directive = "Keep spillway operators on 24x7 standby. Verify radial gate motorized hoists."
        downstream_directive = "Transmit 6-hourly hydrological telemetry bulletin to downstream Karnataka & Telangana authorities."
        public_advisory = "Alert riverside farmsteads against sudden water-level swells. Secure livestock away from riverbanks."
    else:
        alert_code = "GREEN_ALERT_STAGE_1"
        headline = "NORMAL HYDROLOGICAL CONDITIONS"
        urgency = "ROUTINE"
        gates_directive = "Maintain standard irrigation and power generation canal outflows."
        downstream_directive = "Routine daily CWC telemetry synchronization."
        public_advisory = "No flood threat detected. Water levels comfortably within reservoir operating band."

    return {
        "alert_code": alert_code,
        "headline": headline,
        "urgency": urgency,
        "category": category,
        "risk_index": fri,
        "horizon_hours": horizon_h,
        "current_water_level_m": current_wl,
        "predicted_water_level_m": pred_wl,
        "predicted_rainfall_mm": pred_rain,
        "estimated_spread_sqkm": stage3_behaviour.get("water_spread_area_sqkm", 350.0),
        "channel_velocity_ms": stage3_behaviour.get("channel_flow_velocity_ms", 0.8),
        "wave_travel_time_h": stage3_behaviour.get("flood_wave_travel_time_hours", 10.0),
        "directives": {
            "dam_operations": gates_directive,
            "downstream_warning": downstream_directive,
            "public_safety": public_advisory
        },
        "affected_areas": risk_info.get("affected_areas", [])
    }
