"""Multi-Objective Reward Formulation for Smart Building Energy Control.

Combines electricity cost minimization, ASHRAE Standard 55 occupant thermal comfort,
peak power demand capping, visual/lighting satisfaction, and flexible load service.
"""

from __future__ import annotations
import numpy as np


class BuildingRewardCalculator:
    """Calculates composite multi-objective rewards for RL agents."""

    def __init__(
        self,
        weight_cost: float = 1.0,
        weight_comfort: float = 3.5,
        weight_peak: float = 0.8,
        weight_flex: float = 0.5,
        weight_light: float = 0.4,
        peak_threshold_kw: float = 65.0,
        solar_feed_in_tariff: float = 0.05,
    ):
        self.w_cost = weight_cost
        self.w_comfort = weight_comfort
        self.w_peak = weight_peak
        self.w_flex = weight_flex
        self.w_light = weight_light
        self.peak_threshold_kw = peak_threshold_kw
        self.feed_in_tariff = solar_feed_in_tariff

    def compute_reward(
        self,
        indoor_temp: float,
        occupancy: int,
        net_grid_power_kw: float,
        electricity_price: float,
        battery_degradation_cost: float,
        lighting_action: float,
        solar_irradiance: float,
        flex_load_deficit_kw: float,
    ) -> tuple[float, dict[str, float]]:
        """Computes the scalar reward and broken-down diagnostic components.

        Returns:
            reward: Scalar reward (negative cost/penalty) for RL optimization.
            breakdown: Dictionary of individual cost components for logging.
        """
        # 1. Electricity Economic Cost ($)
        if net_grid_power_kw >= 0.0:
            grid_cost = (net_grid_power_kw * 1.0) * electricity_price  # kW * 1hr * $/kWh
        else:
            # Solar export revenue credit
            grid_cost = (net_grid_power_kw * 1.0) * self.feed_in_tariff

        cost_component = grid_cost + battery_degradation_cost

        # 2. Thermal Comfort Penalty (ASHRAE Standard 55)
        is_occupied = occupancy > 0
        if is_occupied:
            target_mid = 22.5
            tolerance = 1.5  # [21.0°C to 24.0°C]
            temp_diff = abs(indoor_temp - target_mid)
            violation = max(0.0, temp_diff - tolerance)
            # Quadratic penalty scaled by occupant presence
            comfort_penalty = (violation ** 2) * (1.0 + 0.015 * occupancy)
        else:
            # Unoccupied setback band [18.0°C to 27.5°C]
            target_mid = 22.75
            tolerance = 4.75
            temp_diff = abs(indoor_temp - target_mid)
            violation = max(0.0, temp_diff - tolerance)
            comfort_penalty = 0.15 * (violation ** 2)

        # 3. Peak Demand Penalty
        peak_exceedance = max(0.0, net_grid_power_kw - self.peak_threshold_kw)
        peak_penalty = (peak_exceedance / 10.0) ** 2

        # 4. Lighting & Visual Comfort Penalty
        # Natural daylight contribution (lux proxy)
        daylight_contribution = min(1.0, solar_irradiance / 600.0)
        total_light_level = lighting_action + 0.5 * daylight_contribution
        if is_occupied and total_light_level < 0.70:
            light_penalty = ((0.70 - total_light_level) * 3.0) ** 2
        else:
            light_penalty = 0.0

        # 5. Flexible Load Deficit Penalty
        flex_penalty = (flex_load_deficit_kw / 10.0) ** 2

        # Aggregate Weighted Cost (Negative of Reward)
        total_cost = (
            self.w_cost * cost_component
            + self.w_comfort * comfort_penalty
            + self.w_peak * peak_penalty
            + self.w_light * light_penalty
            + self.w_flex * flex_penalty
        )

        reward = -float(total_cost)

        breakdown = {
            "reward": reward,
            "cost_dollar": float(cost_component),
            "thermal_comfort_penalty": float(comfort_penalty),
            "peak_penalty": float(peak_penalty),
            "lighting_penalty": float(light_penalty),
            "flexible_penalty": float(flex_penalty),
            "indoor_temp": float(indoor_temp),
            "net_grid_power_kw": float(net_grid_power_kw),
        }

        return reward, breakdown
