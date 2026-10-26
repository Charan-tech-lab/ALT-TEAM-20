"""Comprehensive Evaluation Metrics for Smart Building Energy Management.

Calculates key performance indicators (KPIs) aligned with ASHRAE standards,
energy grid demand response benchmarks, and operational cost savings.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


def compute_building_kpis(history: list[dict[str, Any]], baseline_kpis: dict[str, float] | None = None) -> dict[str, float]:
    """Computes all standard evaluation metrics from an episode history log.

    Metrics computed:
    - Total Energy Consumption (kWh)
    - Electricity Cost ($)
    - Peak Grid Demand (kW)
    - Cumulative Reward
    - Thermal Comfort Violations (Degree-Hours)
    - Average Indoor Temperature Deviation (°C)
    - Energy Savings (%) vs baseline
    - Electricity Cost Reduction (%) vs baseline
    - Peak Demand Reduction (%) vs baseline
    - Renewable Self-Consumption (%)
    """
    if not history:
        return {}

    df = pd.DataFrame(history)

    # 1. Total Energy Consumption (kWh)
    # Total electricity imported from the grid
    total_grid_import_kwh = float(df["net_grid_power_kw"].clip(lower=0.0).sum())
    total_building_demand_kwh = float((df["base_load_kw"] + df["hvac_elec_kw"] + df["lighting_kw"] + df["flex_load_kw"]).sum())

    # 2. Total Electricity Cost ($)
    total_cost_dollar = float(df["cost_dollar"].sum())

    # 3. Peak Grid Demand (kW)
    peak_demand_kw = float(df["net_grid_power_kw"].max())

    # 4. Cumulative Reward
    cumulative_reward = float(df["reward"].sum())

    # 5. Thermal Comfort Violations (Degree-Hours during occupied hours)
    # Target comfort band: 21.0°C to 24.0°C
    occupied_mask = df["occupancy"] > 0
    occupied_temps = df.loc[occupied_mask, "indoor_temp"] if occupied_mask.any() else df["indoor_temp"]

    low_violations = np.maximum(0.0, 21.0 - occupied_temps)
    high_violations = np.maximum(0.0, occupied_temps - 24.0)
    total_comfort_violations_dh = float((low_violations + high_violations).sum())

    # 6. Average Indoor Temperature Deviation from ideal 22.5°C
    avg_temp_dev = float(np.abs(occupied_temps - 22.5).mean()) if len(occupied_temps) > 0 else 0.0

    # 7. Solar Renewable Self-Consumption (%)
    total_solar_gen = float(df["solar_pv_kw"].sum())
    total_solar_exported = float((-df["net_grid_power_kw"].clip(upper=0.0)).sum())
    solar_self_consumed = max(0.0, total_solar_gen - total_solar_exported)
    solar_self_consumption_pct = float((solar_self_consumed / (total_solar_gen + 1e-6)) * 100.0)

    # 8. Comparison with Baseline (if provided)
    kpis = {
        "total_energy_kwh": round(total_grid_import_kwh, 2),
        "total_building_demand_kwh": round(total_building_demand_kwh, 2),
        "total_cost_dollar": round(total_cost_dollar, 2),
        "peak_demand_kw": round(peak_demand_kw, 2),
        "cumulative_reward": round(cumulative_reward, 2),
        "comfort_violations_deg_hours": round(total_comfort_violations_dh, 2),
        "avg_temp_deviation_deg_c": round(avg_temp_dev, 3),
        "solar_self_consumption_pct": round(min(100.0, solar_self_consumption_pct), 1),
    }

    if baseline_kpis:
        base_energy = baseline_kpis.get("total_energy_kwh", 1.0)
        base_cost = baseline_kpis.get("total_cost_dollar", 1.0)
        base_peak = baseline_kpis.get("peak_demand_kw", 1.0)

        kpis["energy_savings_pct"] = round(float((base_energy - total_grid_import_kwh) / (base_energy + 1e-6) * 100.0), 2)
        kpis["cost_reduction_pct"] = round(float((base_cost - total_cost_dollar) / (base_cost + 1e-6) * 100.0), 2)
        kpis["peak_demand_reduction_pct"] = round(float((base_peak - peak_demand_kw) / (base_peak + 1e-6) * 100.0), 2)
    else:
        kpis["energy_savings_pct"] = 0.0
        kpis["cost_reduction_pct"] = 0.0
        kpis["peak_demand_reduction_pct"] = 0.0

    return kpis
