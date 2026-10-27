"""
Stage 4: Flood Risk Assessment Module
Multi-Factor Risk Scoring, Low/Medium/High Classification, and Confusion Matrix
Target: Almatti Reservoir (Lat: 16.33 N, Lon: 75.88 E)
GNIT / AKTU Academic Mini Project
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from .data_loader import ALMATTI_METADATA

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


class FloodRiskClassifier:
    """
    Flood Risk Assessment and Classification Engine.
    Combines hydraulic water level, rate of rise, and forecast rainfall
    into an explainable Flood Risk Index (FRI) and multi-class status (Low, Medium, High).
    """

    def __init__(self):
        self.frl = ALMATTI_METADATA["frl_m"]
        self.danger_level = ALMATTI_METADATA["danger_level_m"]
        self.warning_level = ALMATTI_METADATA["warning_level_m"]
        self.bed_level = ALMATTI_METADATA["bed_level_m"]
        self.model = None
        self.metrics = {}
        self.classes = ["Low Risk", "Medium Risk", "High Risk"]

    def compute_risk_index(self, water_level_m, rainfall_24h_mm=0.0, rate_of_rise_cm_h=0.0):
        """
        Calculates the Composite Flood Risk Index (FRI: 0 to 100).
        Weightings:
        - Water level proximity to Danger/FRL: 55%
        - Predicted 24h rainfall volume: 25%
        - Rate of rise / hydraulic velocity: 20%
        """
        wl = float(water_level_m)
        r = float(rainfall_24h_mm)
        rise = float(rate_of_rise_cm_h)

        # 1. Water Level Score (0 to 100)
        if wl < 512.0:
            s_wl = 10.0 * (wl - self.bed_level) / (512.0 - self.bed_level)
        elif wl < self.warning_level:  # 512 to 517
            s_wl = 10.0 + 40.0 * (wl - 512.0) / (self.warning_level - 512.0)
        elif wl < self.danger_level:   # 517 to 519
            s_wl = 50.0 + 35.0 * (wl - self.warning_level) / (self.danger_level - self.warning_level)
        else:                          # >= 519 (Danger)
            s_wl = 85.0 + 15.0 * min(1.0, (wl - self.danger_level) / (self.frl - self.danger_level))
        s_wl = float(np.clip(s_wl, 0.0, 100.0))

        # 2. Rainfall Score (0 to 100)
        if r < 5.0:
            s_rain = r * 2.0
        elif r < 25.0:
            s_rain = 10.0 + (r - 5.0) * 1.5
        elif r < 65.0:
            s_rain = 40.0 + (r - 25.0) * 1.0
        else:
            s_rain = 80.0 + min(20.0, (r - 65.0) * 0.4)
        s_rain = float(np.clip(s_rain, 0.0, 100.0))

        # 3. Rate of Rise Score (0 to 100)
        if rise <= 0.0:
            s_rise = 0.0
        elif rise < 2.0:
            s_rise = rise * 15.0
        elif rise < 5.0:
            s_rise = 30.0 + (rise - 2.0) * 15.0
        else:
            s_rise = 75.0 + min(25.0, (rise - 5.0) * 5.0)
        s_rise = float(np.clip(s_rise, 0.0, 100.0))

        # Composite Flood Risk Index
        fri = 0.55 * s_wl + 0.25 * s_rain + 0.20 * s_rise
        return round(float(np.clip(fri, 0.0, 100.0)), 1)

    def classify_risk(self, water_level_m, rainfall_24h_mm=0.0, rate_of_rise_cm_h=0.0):
        """
        Classifies risk into Low, Medium, or High Risk with rule-based safety overrides.
        """
        wl = float(water_level_m)
        fri = self.compute_risk_index(wl, rainfall_24h_mm, rate_of_rise_cm_h)

        # Safety-critical regulatory overrides:
        # CWC guidelines dictate automatic Danger if water reaches 519m
        if wl >= self.danger_level or fri >= 70.0:
            category = "High Risk"
            color = "#ef4444"  # Red
            badge = "CRITICAL"
            action = "Immediate crest gates opening, downstream evacuation alerts, flood wave bulletin to Narayanapur Dam."
        elif wl >= self.warning_level or fri >= 38.0:
            category = "Medium Risk"
            color = "#f59e0b"  # Amber
            badge = "ALERT"
            action = "Close barrage monitoring, prepare spillway radial gates, alert taluk disaster control rooms."
        else:
            category = "Low Risk"
            color = "#10b981"  # Emerald Green
            badge = "NORMAL"
            action = "Routine reservoir storage operation, standard power generation, normal telemetry logging."

        # Affected tehsils and geographical impact assessment
        affected_areas = self._get_affected_areas(category, wl)

        return {
            "risk_index": fri,
            "category": category,
            "badge": badge,
            "color": color,
            "action_guideline": action,
            "water_level_m": wl,
            "rainfall_24h_mm": round(rainfall_24h_mm, 2),
            "rate_of_rise_cm_h": round(rate_of_rise_cm_h, 2),
            "affected_areas": affected_areas
        }

    def _get_affected_areas(self, category, water_level_m):
        """
        Identifies specific taluks / areas affected in Bagalkot and Vijayapura districts.
        """
        if category == "High Risk":
            return [
                {"name": "Bilagi Taluk (Backwater Flood Zone)", "risk": "High", "impact": "Submergence of low agricultural plains"},
                {"name": "Jamkhandi (Krishna Riverbank)", "risk": "High", "impact": "Riverbank overflow, bridge road submerged"},
                {"name": "Bagalkot Rural (Ghataprabha/Krishna Confluence)", "risk": "High", "impact": "Inundation of floodplain settlements"},
                {"name": "Muddebihal / Nidagundi Downstream Reach", "risk": "High", "impact": "Spillway discharge downstream wave velocity"}
            ]
        elif category == "Medium Risk":
            return [
                {"name": "Bilagi Taluk Lowlands", "risk": "Medium", "impact": "Water approaching agricultural boundary embankments"},
                {"name": "Bagalkot Island Settlements", "risk": "Medium", "impact": "Standby warning issued to riverside villages"},
                {"name": "Nidagundi Downstream Canals", "risk": "Medium", "impact": "Regulated discharge canal monitoring"}
            ]
        else:
            return [
                {"name": "All Krishna Basin Sectors (Almatti Command Area)", "risk": "Low", "impact": "Within safe embankments"}
            ]

    def train_classifier(self, df_features):
        """
        Train ML Classifier for flood risk classification on historical dataset.
        Generates classification report and confusion matrix.
        """
        # Generate historical labels based on CWC criteria
        labels = []
        for _, row in df_features.iterrows():
            wl = row.get("Water_Level_m", 515.0)
            r = row.get("Rainfall_Hourly_mm", 0.0) * 24.0
            rise = row.get("rate_of_rise_1h", 0.0) * 100.0
            res = self.classify_risk(wl, r, rise)
            labels.append(res["category"])

        y = np.array(labels)
        feature_cols = [c for c in df_features.columns if c not in ["Water_Level_m", "DateTime", "Date"] and not c.startswith("target_")]
        X = df_features[feature_cols].values

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        labels_order = ["Low Risk", "Medium Risk", "High Risk"]
        # Ensure only classes present in test are considered
        present_labels = [c for c in labels_order if c in y_test or c in y_pred]

        cm = confusion_matrix(y_test, y_pred, labels=labels_order)
        report = classification_report(y_test, y_pred, labels=labels_order, output_dict=True, zero_division=0)

        self.model = clf
        self.metrics = {
            "confusion_matrix": cm.tolist(),
            "labels": labels_order,
            "accuracy": round(float(np.mean(y_test == y_pred)), 4),
            "macro_f1": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "classification_report": report
        }
        print(f"[Stage 4] Classifier Accuracy: {self.metrics['accuracy']*100:.2f}%, Weighted F1: {self.metrics['weighted_f1']:.4f}")
        return self.metrics

    def save(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(MODEL_DIR, "stage4_risk.joblib")
        payload = {
            "model": self.model,
            "metrics": self.metrics,
            "classes": self.classes
        }
        joblib.dump(payload, filepath)
        print(f"Stage 4 risk model saved to {filepath}")

    @classmethod
    def load(cls, filepath=None):
        if filepath is None:
            filepath = os.path.join(MODEL_DIR, "stage4_risk.joblib")
        classifier = cls()
        if os.path.exists(filepath):
            payload = joblib.load(filepath)
            classifier.model = payload["model"]
            classifier.metrics = payload["metrics"]
            classifier.classes = payload["classes"]
        return classifier
