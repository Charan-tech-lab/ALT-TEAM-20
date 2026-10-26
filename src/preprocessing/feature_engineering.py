"""Feature Engineering Pipeline for Smart Building Reinforcement Learning.

Extracts cyclic temporal indicators, thermal trend dynamics, solar irradiance momentum,
and standardized observation vectors suitable for neural network policy optimization.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class BuildingFeaturePipeline:
    """Prepares and transforms raw building data into RL-ready state representations."""

    def __init__(self):
        # Normalization ranges (min, max) for scaling state components into [-1, 1] or [0, 1]
        self.feature_bounds = {
            "indoor_temp": (15.0, 32.0),
            "outdoor_temp": (-10.0, 42.0),
            "outdoor_temp_ma3": (-10.0, 42.0),
            "occupancy": (0.0, 120.0),
            "solar_irradiance": (0.0, 1050.0),
            "solar_pv_kw": (0.0, 65.0),
            "base_load_kw": (5.0, 60.0),
            "electricity_price": (0.05, 0.45),
            "flexible_load_req_kw": (0.0, 25.0),
            "battery_soc": (0.0, 1.0),
            "peak_so_far_kw": (0.0, 150.0),
        }

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enriches the dataframe with engineered features."""
        enriched = df.copy()

        # 1. Cyclic Temporal Representations
        hour = enriched.index.hour.values
        enriched["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        enriched["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        day_of_week = enriched.index.dayofweek.values
        enriched["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
        enriched["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

        # 2. Thermal Momentum (3-hour and 6-hour moving averages)
        enriched["outdoor_temp_ma3"] = (
            enriched["outdoor_temp"].rolling(window=3, min_periods=1).mean()
        )
        enriched["temp_gradient"] = enriched["outdoor_temp"].diff().fillna(0.0)

        # 3. Solar Horizon Feature (lead 1-hour solar irradiance indicator)
        enriched["solar_lead_1h"] = enriched["solar_irradiance"].shift(-1).bfill()

        # 4. Workday vs Weekend Flag
        enriched["is_workday"] = (day_of_week < 5).astype(float)

        return enriched

    def normalize_feature(self, value: float, feature_name: str) -> float:
        """Normalizes a single scalar feature to approximately [-1, 1] or [0, 1]."""
        if feature_name in self.feature_bounds:
            f_min, f_max = self.feature_bounds[feature_name]
            # Scale to [-1, 1]
            return float(np.clip(2.0 * (value - f_min) / (f_max - f_min) - 1.0, -1.5, 1.5))
        return float(value)

    def build_state_vector(
        self,
        indoor_temp: float,
        row: pd.Series | dict,
        battery_soc: float,
        peak_so_far_kw: float,
    ) -> np.ndarray:
        """Constructs an aligned 12-dimensional state vector for the RL agent.

        State representation:
        [
            norm(indoor_temp),
            norm(outdoor_temp),
            norm(outdoor_temp_ma3),
            norm(occupancy),
            norm(solar_pv_kw),
            norm(base_load_kw),
            norm(electricity_price),
            norm(battery_soc),
            norm(peak_so_far_kw),
            hour_sin,
            hour_cos,
            is_workday
        ]
        """
        state = np.array([
            self.normalize_feature(indoor_temp, "indoor_temp"),
            self.normalize_feature(row["outdoor_temp"], "outdoor_temp"),
            self.normalize_feature(row.get("outdoor_temp_ma3", row["outdoor_temp"]), "outdoor_temp_ma3"),
            self.normalize_feature(row["occupancy"], "occupancy"),
            self.normalize_feature(row["solar_pv_kw"], "solar_pv_kw"),
            self.normalize_feature(row["base_load_kw"], "base_load_kw"),
            self.normalize_feature(row["electricity_price"], "electricity_price"),
            self.normalize_feature(battery_soc, "battery_soc"),
            self.normalize_feature(peak_so_far_kw, "peak_so_far_kw"),
            float(row.get("hour_sin", 0.0)),
            float(row.get("hour_cos", 1.0)),
            float(row.get("is_workday", 1.0)),
        ], dtype=np.float32)

        return state
