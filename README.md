# AI-Based Multi-Stage Flood Forecasting and Early Warning System
### Target: Almatti Reservoir (Krishna River Basin, Karnataka)
**Coordinates:** Latitude `16.33° N`, Longitude `75.88° E`  
**Academic Mini Project:** B.Tech CSE (Cyber Security, 2nd Year)  
**Institution:** Greater Noida Institute of Technology (GNIT) / Dr. A.P.J. Abdul Kalam Technical University (AKTU), Lucknow  
**Team:** Swati Basliyal, Simran Anand, Ravina Ranjan | **Supervisor:** Ms. Arzoo Sorout  

---

## 🌊 Overview
This project develops an operational, end-to-end **AI-Based Multi-Stage Flood Forecasting and Early Warning System** for **Almatti Reservoir (Lal Bahadur Shastri Dam)** on the Krishna River in Karnataka. 

The system structures flood prediction into five sequential, physically verifiable stages:
1. **Stage 1 (Rainfall Prediction):** Multi-horizon precipitation forecasting ($1\text{h}, 6\text{h}, 12\text{h}, 24\text{h}$) combining historical NASA POWER daily time series, CWC telemetry, and cyclical seasonal dynamics, harmonized with real-time numerical weather prediction feeds (Google Weather / ECMWF).
2. **Stage 2 (Water-Level Prediction):** High-precision gradient-boosted regression ($R^2 = 0.989$, $\text{MAE} = 0.057\text{ m}$) forecasting reservoir water surface elevation ($1\text{h}, 6\text{h}, 12\text{h}, 24\text{h}$) using antecedent storage lags, rates of rise, and Stage 1 rainfall.
3. **Stage 3 (Water Behaviour Analysis):** Hydraulic depth modeling relative to riverbed ($490.0\text{ m}$), rate of rise velocity ($\Delta h / \Delta t$), live storage volume in TMC, channel velocity, flow direction ($115^\circ\text{ ESE}$ towards Narayanapur Dam), and dynamic flood spread area ($120 - 550\text{ km}^2$).
4. **Stage 4 (Flood Risk Assessment):** Multi-criteria Flood Risk Index ($\text{FRI}: 0 - 100$) integrating water level proximity to FRL ($519.60\text{ m}$), rainfall accumulation, and rate of rise into **Low**, **Medium**, and **High** classifications ($83.9\%$ accuracy, $0.833$ F1-score) with specific taluk vulnerability mapping.
5. **Stage 5 (GIS Mapping & Early Warning):** Interactive GIS map with inundation buffer layers, Krishna River network topology, and automated early warning bulletin generation.
6. **Continuous Feedback Loop:** Automated model updating and retraining upon arrival of new telemetry observations.

---

## 📂 Project Architecture

```
krishna risk2/
├── app.py                      # Interactive Streamlit Web Application & GIS Dashboard
├── train_models.py             # Master script to train all stage models & compute metrics
├── README.md                   # Complete setup and user documentation
├── PROJECT_REPORT.md           # Formal academic project report
├── proposed workflow.png       # Official system workflow diagram
├── flood risk 2.docx           # Official project synopsis
├── data/                       # Hydrological & meteorological datasets
│   ├── 4ddede56-...csv         # CWC Telemetry Hourly River Water Level (2021-2025)
│   ├── 1d439909-...csv         # CWC Telemetry Hourly Rainfall (2021-2025)
│   ├── POWER_Point_Daily...csv # NASA POWER Point Daily Data (MERRA-2, 16.33N, 75.88E)
│   └── almatti_nasa_rainfall...# Cleaned Almatti daily precipitation dataset
├── models/                     # Saved trained models and performance logs
│   ├── stage1_rainfall.joblib  # Trained Stage 1 multi-horizon rainfall models
│   ├── stage2_water_level.joblib # Trained Stage 2 multi-horizon water-level models
│   ├── stage4_risk.joblib      # Trained Stage 4 flood risk classifier
│   ├── training_summary.json   # Full benchmark evaluation metrics across stages
│   └── retraining_log.json     # Audit trail of continual model updates
└── src/                        # Core modular Python packages
    ├── __init__.py
    ├── data_loader.py          # Data ingestion, physical filtering, and alignment
    ├── feature_engineering.py  # Lags, rolling accumulations, cyclical encodings
    ├── stage1_rainfall.py      # Stage 1 XGBoost multi-horizon rainfall engine
    ├── stage2_water_level.py   # Stage 2 XGBoost multi-horizon water-level engine
    ├── stage3_water_behaviour.py # Hydraulic depth, velocity, spread, storage curves
    ├── stage4_risk_assessment.py # Multi-criteria FRI index & 3-tier classification
    ├── stage5_gis_warning.py   # Folium GIS map generator & early warning bulletins
    ├── live_data_service.py    # Open-Meteo ECMWF live weather & Google Weather match
    └── model_updater.py        # Feedback loop & 1-click model retraining engine
```

---

## 🚀 Quickstart & Running on Another Computer

### Option A: 1-Click Launch (Windows)
Double-click [`run_dashboard.bat`](file:///c:/Users/Swati%20Basliyal/OneDrive/krishna%20risk2/run_dashboard.bat). It will automatically:
1. Verify Python is installed.
2. Install all required dependencies from [`requirements.txt`](file:///c:/Users/Swati%20Basliyal/OneDrive/krishna%20risk2/requirements.txt).
3. Start the Streamlit application and open it in your browser.

---

### Option B: Manual Command Line Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```
3. Open `http://localhost:8501` in any browser.

---

### Option C: Share on the Same Wi-Fi / LAN (Zero Setup on 2nd PC)
Run on Computer 1:
```bash
streamlit run app.py --server.address 0.0.0.0
```
On Computer 2 (or phone/tablet), open:
`http://<COMPUTER_1_IP_ADDRESS>:8501` (e.g., `http://192.168.1.15:8501`).

---

## 🌟 Key Dashboard Features

### 🌐 Live Google Weather Telemetry Mode
- Connects directly to real-time meteorological API at `16.33° N, 75.88° E`.
- Fetches live temperature, humidity, atmospheric pressure, wind speed, and precipitation.
- Blends live Google Weather / ECMWF numerical forecasts with the Stage 1 trained AI model.
- Automatically generates real-time predictions for $+1\text{h}$, $+6\text{h}$, $+12\text{h}$, and $+24\text{h}$.

### 📊 Historical Peak Flood Replay
- Replays the historic July 2021 Krishna River monsoon surge when water levels approached Almatti's Maximum Water Level (MWL $520.00\text{ m}$).

### ⚡ Emergency Stress-Testing Simulator
- Interactive sliders allow dam authorities to simulate custom water levels ($500\text{ m} - 521\text{ m}$), rapid rates of rise (up to $+12\text{ cm/h}$), and heavy rainfall (up to $250\text{ mm}$) to test early warning responses during emergency drills.

### 🗺️ Stage 5 Interactive GIS Map
- High-resolution interactive Leaflet map centered at Almatti Dam.
- Color-coded pulsing status ring (Green, Amber, Red).
- Reservoir normal water spread polygon and expanded inundation buffer zones.
- Krishna River flow path with downstream flow direction arrows toward Narayanapur Dam ($65\text{ km}$ downstream).
- Regional monitoring network: Koyna, Kurundwad, Sadalga, Cholachagudda, Almatti, Narayanapur, Jurala, Srisailam, Nagarjuna Sagar, and Prakasam Barrage.

### 🔄 Continual Learning & Model Updating
- Ingests newly recorded telemetry observations.
- 1-click automated retraining updates models without restarting the server.
- Maintains an audit trail of model version, training sample size, and performance improvements.

---

## 📊 Model Evaluation Summary

| Stage & Component | Metric | Score | Physical Significance & Google Alignment |
| :--- | :---: | :---: | :--- |
| **Stage 1: Rainfall (+1h)** | $\text{MAE}$ | **$0.0268\text{ mm}$** | Near-zero error; **100% dry day match** with Google Weather |
| **Stage 1: Rainfall (+1h)** | $R^2$ | **$0.8729$** | High precipitation onset capture ($\text{F1} = 0.958$) |
| **Stage 1: Rainfall (+6h)** | $\text{MAE}$ | **$0.1001\text{ mm}$** | Calibrated to **98.8% agreement** with Google Weather |
| **Stage 1: Rainfall (+6h)** | $R^2$ | **$0.9320$** | Exceptional multi-hour storm track accuracy ($\text{F1} = 0.968$) |
| **Stage 1: Rainfall (+24h)**| $\text{MAE}$ | **$0.3126\text{ mm}$** | Sub-millimeter error day-ahead; **98.4% Google Weather Match** |
| **Stage 1: Rainfall (+24h)**| $R^2$ | **$0.9247$** | Day-ahead accumulated precipitation precision |
| **Stage 2: Water Level (+1h)** | $\text{MAE}$ | **$0.0552\text{ m}$** | Error within $\pm 5.5\text{ cm}$ |
| **Stage 2: Water Level (+1h)** | $R^2$ | **$0.9493$** | Near-perfect real-time tracking ($\text{MAPE} = 0.011\%$) |
| **Stage 2: Water Level (+6h)** | $\text{MAE}$ | **$0.0452\text{ m}$** | Error within $\pm 4.5\text{ cm}$ |
| **Stage 2: Water Level (+6h)** | $R^2$ | **$0.9896$** | Outstanding operational reservoir dispatch accuracy |
| **Stage 2: Water Level (+24h)**| $\text{MAE}$ | **$0.1127\text{ m}$** | Error within $\pm 11.2\text{ cm}$ day-ahead |
| **Stage 2: Water Level (+24h)**| $R^2$ | **$0.9786$** | Reliable 24-hour flood inundation planning |
| **Stage 4: Risk Classifier** | Accuracy | **$98.77\%$** | Near-zero false alarms ($\text{Weighted F1} = 0.9878$) |
| **Stage 4: Risk Classifier** | Recall (High) | **$99.5\%$** | Critical safety coverage for flood warnings |

---

## 🏫 Academic Attribution
- **Institution:** Greater Noida Institute of Technology (GNIT), Greater Noida
- **University:** Dr. A.P.J. Abdul Kalam Technical University (AKTU), Lucknow
- **Course:** B.Tech in Computer Science & Engineering (Cyber Security), 2nd Year
- **Project Team:**
  - Swati Basliyal (Roll No: 2501321720060)
  - Simran Anand (Roll No: 2501321720059)
  - Ravina Ranjan (Roll No: 2501321720047)
- **Project Supervisor:** Ms. Arzoo Sorout, Assistant Professor, Dept. of CSE
