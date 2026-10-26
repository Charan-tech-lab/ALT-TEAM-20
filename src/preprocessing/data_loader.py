"""Building Energy Data Loader and Synthetic Profile Generator.

Generates and pre-processes realistic multi-zone smart building energy profiles
conforming to the Building Data Genome Project 2 standards and ASHRAE guidelines.
Includes outdoor weather, solar irradiance, occupancy patterns, base electrical loads,
dynamic Time-of-Use (TOU) electricity tariffs, solar PV generation, and flexible load demands.
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd


class BuildingDataGenerator:
    """Generates synthetic yet thermodynamically and statistically realistic building energy data."""

    def __init__(
        self,
        year: int = 2024,
        pv_capacity_kw: float = 60.0,
        max_occupancy: int = 120,
        base_load_min_kw: float = 12.0,
        base_load_max_kw: float = 45.0,
        seed: int = 42,
    ):
        self.year = year
        self.pv_capacity_kw = pv_capacity_kw
        self.max_occupancy = max_occupancy
        self.base_load_min_kw = base_load_min_kw
        self.base_load_max_kw = base_load_max_kw
        self.rng = np.random.default_rng(seed)

    def generate_annual_profile(self) -> pd.DataFrame:
        """Generates an 8760-hour full year profile."""
        start_date = f"{self.year}-01-01 00:00:00"
        timestamps = pd.date_range(start=start_date, periods=8760, freq="h")
        df = pd.DataFrame(index=timestamps)
        df.index.name = "timestamp"

        day_of_year = df.index.dayofyear.values
        hour = df.index.hour.values
        day_of_week = df.index.dayofweek.values  # 0=Monday, 6=Sunday
        is_weekend = (day_of_week >= 5).astype(int)

        # 1. Outdoor Air Temperature (Seasonal + Diurnal + Synoptic Weather Waves)
        # Seasonal base: lowest in Jan/Feb (~0°C to 5°C), highest in July/Aug (~30°C to 36°C)
        seasonal_temp = 16.0 + 13.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365.0)
        # Diurnal swing: peak at 15:00, trough at 05:00
        diurnal_temp = 5.5 * np.sin(2 * np.pi * (hour - 9) / 24.0)
        # Weather wave variations (3-7 day weather fronts)
        weather_fronts = 4.0 * np.sin(2 * np.pi * day_of_year / 6.5) + self.rng.normal(0, 1.2, len(df))
        outdoor_temp = seasonal_temp + diurnal_temp + weather_fronts
        df["outdoor_temp"] = np.round(outdoor_temp, 2)

        # 2. Outdoor Relative Humidity (%)
        # Typically anti-correlated with temperature
        rel_humidity = 65.0 - 18.0 * np.sin(2 * np.pi * (hour - 9) / 24.0) + self.rng.normal(0, 5.0, len(df))
        df["relative_humidity"] = np.clip(np.round(rel_humidity, 1), 20.0, 98.0)

        # 3. Solar Irradiance (W/m^2)
        # Sun angle proxy based on hour of day and day of year
        solar_elevation = np.maximum(0.0, np.sin(np.pi * (hour - 6) / 12.0))
        solar_elevation = np.where((hour >= 6) & (hour <= 18), solar_elevation, 0.0)
        seasonal_solar_factor = 0.75 + 0.25 * np.sin(2 * np.pi * (day_of_year - 80) / 365.0)
        cloud_cover_noise = np.clip(self.rng.beta(2, 2, len(df)) * 1.2, 0.3, 1.0)
        solar_irradiance = 1000.0 * solar_elevation * seasonal_solar_factor * cloud_cover_noise
        df["solar_irradiance"] = np.round(solar_irradiance, 1)

        # 4. Solar PV Generation (kW)
        # System efficiency ~ 16%, rooftop panel area sized for pv_capacity_kw peak
        pv_generation = (df["solar_irradiance"] / 1000.0) * self.pv_capacity_kw * np.clip(1.0 - 0.004 * (df["outdoor_temp"] - 25.0), 0.75, 1.05)
        df["solar_pv_kw"] = np.round(np.maximum(0.0, pv_generation), 2)

        # 5. Occupancy Schedule (Persons)
        # Commercial office profile: Monday-Friday 08:00 to 18:00
        occ_prob = np.zeros(len(df))
        for i in range(len(df)):
            h = hour[i]
            wk = is_weekend[i]
            if wk == 1:
                # Minimal weekend staff
                occ_prob[i] = 0.05 if 10 <= h <= 16 else 0.01
            else:
                if 8 <= h < 9:
                    occ_prob[i] = 0.45
                elif 9 <= h < 12:
                    occ_prob[i] = 0.90
                elif 12 <= h < 13:
                    occ_prob[i] = 0.65  # Lunch break
                elif 13 <= h < 17:
                    occ_prob[i] = 0.95
                elif 17 <= h < 19:
                    occ_prob[i] = 0.35
                elif 19 <= h < 21:
                    occ_prob[i] = 0.10
                else:
                    occ_prob[i] = 0.02

        occupancy = (occ_prob * self.max_occupancy) + self.rng.integers(-3, 4, len(df))
        df["occupancy"] = np.clip(occupancy, 0, self.max_occupancy).astype(int)

        # 6. Non-HVAC Base Electrical Load (kW) - Computing, appliances, emergency lighting
        base_load_trend = self.base_load_min_kw + (self.base_load_max_kw - self.base_load_min_kw) * (df["occupancy"] / self.max_occupancy)
        base_load = base_load_trend + self.rng.normal(0, 1.8, len(df))
        df["base_load_kw"] = np.round(np.clip(base_load, self.base_load_min_kw * 0.8, self.base_load_max_kw * 1.25), 2)

        # 7. Dynamic Electricity Tariffs / Time-of-Use (TOU) ($/kWh)
        # Summer peak hours cost significantly more
        prices = np.zeros(len(df))
        for i in range(len(df)):
            h = hour[i]
            m = df.index[i].month
            is_summer = 6 <= m <= 9
            if is_weekend[i] == 1:
                prices[i] = 0.09
            else:
                if 14 <= h < 20:  # On-Peak
                    prices[i] = 0.38 if is_summer else 0.28
                elif (7 <= h < 14) or (20 <= h < 22):  # Shoulder
                    prices[i] = 0.16
                else:  # Off-Peak (22:00 - 07:00)
                    prices[i] = 0.08

        df["electricity_price"] = np.round(prices, 3)

        # 8. Flexible Deferrable Loads (kW demand)
        # E.g. Electric Vehicle fleet charging or water pre-heating
        flex_loads = np.zeros(len(df))
        for i in range(len(df)):
            h = hour[i]
            wk = is_weekend[i]
            if wk == 0 and 9 <= h <= 17:
                flex_loads[i] = self.rng.uniform(5.0, 20.0)
            else:
                flex_loads[i] = 0.0
        df["flexible_load_req_kw"] = np.round(flex_loads, 2)

        return df


def load_or_generate_building_data(
    data_dir: str | Path | None = None,
    force_regenerate: bool = False,
    year: int = 2024,
) -> pd.DataFrame:
    """Loads existing processed building genome dataset or generates a new one."""
    if data_dir is None:
        # Default to repo data dir
        repo_root = Path(__file__).resolve().parent.parent.parent
        data_dir = repo_root / "data"
    else:
        data_dir = Path(data_dir)

    raw_path = data_dir / "raw" / f"building_energy_genome_{year}.csv"
    processed_path = data_dir / "processed" / f"building_energy_genome_{year}.csv"

    if not force_regenerate and processed_path.exists():
        df = pd.read_csv(processed_path, parse_dates=["timestamp"], index_col="timestamp")
        return df

    # Generate new dataset
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    generator = BuildingDataGenerator(year=year)
    df = generator.generate_annual_profile()

    df.to_csv(raw_path)
    df.to_csv(processed_path)
    return df


if __name__ == "__main__":
    df = load_or_generate_building_data()
    print("Generated building dataset shape:", df.shape)
    print(df.head())
