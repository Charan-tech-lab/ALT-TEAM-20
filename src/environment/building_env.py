"""Gymnasium Smart Building Energy Management Environment.

Provides a standardized reinforcement learning interface for HVAC control,
battery storage arbitration, intelligent lighting, and flexible demand response.
Supports both continuous action spaces (PPO / DDPG) and discrete action spaces (DQN).
"""

from __future__ import annotations
import itertools
from typing import Any
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from .thermodynamics import BuildingThermodynamics
from .battery import BatteryStorage
from .reward_functions import BuildingRewardCalculator
from ..preprocessing.feature_engineering import BuildingFeaturePipeline
from ..preprocessing.data_loader import load_or_generate_building_data


class SmartBuildingEnv(gym.Env):
    """Gymnasium environment for smart building intelligent energy management."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        episode_length_hours: int = 168,  # 1 week standard episode (168 steps)
        discrete_actions: bool = False,
        reward_weights: dict[str, float] | None = None,
        max_lighting_kw: float = 12.0,
        initial_temp: float = 22.5,
        start_step: int | None = None,
    ):
        super().__init__()

        # Data initialization
        if df is None:
            df = load_or_generate_building_data()

        self.feature_pipeline = BuildingFeaturePipeline()
        self.df = self.feature_pipeline.fit_transform(df)

        self.episode_length = min(episode_length_hours, len(self.df) - 1)
        self.discrete_actions = discrete_actions
        self.max_lighting_kw = max_lighting_kw
        self.initial_temp = initial_temp
        self.fixed_start_step = start_step

        # Internal subsystem models
        self.thermo = BuildingThermodynamics()
        self.battery = BatteryStorage()
        weights = reward_weights or {}
        self.reward_calc = BuildingRewardCalculator(
            weight_cost=weights.get("cost", 1.0),
            weight_comfort=weights.get("comfort", 3.5),
            weight_peak=weights.get("peak", 0.8),
            weight_flex=weights.get("flex", 0.5),
            weight_light=weights.get("light", 0.4),
        )

        # Observation Space: 12 standardized features
        # [indoor_temp, outdoor_temp, ma3_temp, occupancy, solar_pv, base_load, price, soc, peak, hr_sin, hr_cos, workday]
        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(12,),
            dtype=np.float32,
        )

        # Action Space Setup
        # Continuous action: [hvac (-1 to 1), lighting (-1 to 1), battery (-1 to 1), flex (-1 to 1)]
        self.action_space_continuous = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Discrete Action mapping for DQN
        self.discrete_hvac_levels = [-0.6, 0.0, 0.4, 0.8, 1.0] # Heat, Off, Low, Med, High
        self.discrete_battery_levels = [-1.0, 0.0, 1.0]         # Discharge, Idle, Charge
        self.discrete_light_levels = [0.4, 1.0]                 # Dimmed, Full
        self.discrete_flex_levels = [0.0, 1.0]                  # Defer, Dispatch

        self.discrete_action_combinations = list(itertools.product(
            self.discrete_hvac_levels,
            self.discrete_battery_levels,
            self.discrete_light_levels,
            self.discrete_flex_levels,
        ))

        if self.discrete_actions:
            self.action_space = spaces.Discrete(len(self.discrete_action_combinations))
        else:
            self.action_space = self.action_space_continuous

        # Runtime Episode State
        self.current_step = 0
        self.episode_start_idx = 0
        self.indoor_temp = self.initial_temp
        self.peak_demand_so_far = 0.0
        self.deferred_flex_queue_kw = 0.0

        # Episode Diagnostics and History
        self.history: list[dict[str, Any]] = []

    def _get_current_data_row(self) -> pd.Series:
        idx = (self.episode_start_idx + self.current_step) % len(self.df)
        return self.df.iloc[idx]

    def _get_observation(self) -> np.ndarray:
        row = self._get_current_data_row()
        return self.feature_pipeline.build_state_vector(
            indoor_temp=self.indoor_temp,
            row=row,
            battery_soc=self.battery.soc,
            peak_so_far_kw=self.peak_demand_so_far,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        if self.fixed_start_step is not None:
            self.episode_start_idx = self.fixed_start_step
        elif options and "start_step" in options:
            self.episode_start_idx = options["start_step"]
        elif seed is not None:
            rng = np.random.default_rng(seed)
            max_start = max(1, len(self.df) - self.episode_length - 1)
            self.episode_start_idx = int(rng.integers(0, max_start))
        else:
            max_start = max(1, len(self.df) - self.episode_length - 1)
            self.episode_start_idx = int(np.random.randint(0, max_start))

        self.current_step = 0
        self.indoor_temp = self.initial_temp
        self.peak_demand_so_far = 0.0
        self.deferred_flex_queue_kw = 0.0
        self.battery.reset()
        self.history.clear()

        obs = self._get_observation()
        info = {
            "start_time": str(self.df.index[self.episode_start_idx]),
            "initial_temp": self.indoor_temp,
            "initial_soc": self.battery.soc,
        }
        return obs, info

    def _decode_action(self, action: Any) -> tuple[float, float, float, float]:
        """Decodes either discrete or continuous action into [hvac, light_frac, battery, flex_frac]."""
        if self.discrete_actions:
            hvac, batt, light, flex = self.discrete_action_combinations[int(action)]
            return float(hvac), float(light), float(batt), float(flex)
        else:
            action_arr = np.asarray(action, dtype=np.float32).flatten()
            hvac = float(np.clip(action_arr[0], -1.0, 1.0))
            # Map [-1, 1] to [0.1, 1.0] for lighting
            light = float(np.clip((action_arr[1] + 1.0) / 2.0 * 0.9 + 0.1, 0.1, 1.0))
            batt = float(np.clip(action_arr[2], -1.0, 1.0))
            # Map [-1, 1] to [0.0, 1.0] for flex load
            flex = float(np.clip((action_arr[3] + 1.0) / 2.0, 0.0, 1.0))
            return hvac, light, batt, flex

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        row = self._get_current_data_row()
        hvac_act, light_act, batt_act, flex_act = self._decode_action(action)

        # 1. Lighting Electrical Power
        lighting_elec_kw = light_act * self.max_lighting_kw

        # 2. Advance Building Thermodynamics
        next_temp, hvac_elec_kw, q_hvac = self.thermo.step(
            current_indoor_temp=self.indoor_temp,
            outdoor_temp=row["outdoor_temp"],
            solar_irradiance=row["solar_irradiance"],
            occupancy=int(row["occupancy"]),
            base_load_kw=row["base_load_kw"],
            lighting_power_kw=lighting_elec_kw,
            hvac_action=hvac_act,
            dt_hours=1.0,
        )

        # 3. Battery Energy Storage Dispatch
        batt_grid_power_kw, next_soc, degradation_cost = self.battery.step(
            action=batt_act,
            dt_hours=1.0,
        )

        # 4. Flexible Load Management
        new_flex_demand = row.get("flexible_load_req_kw", 0.0)
        self.deferred_flex_queue_kw += new_flex_demand
        dispatched_flex_kw = flex_act * self.deferred_flex_queue_kw
        self.deferred_flex_queue_kw -= dispatched_flex_kw

        # End of working day deadline penalty for flex loads (at 18:00)
        hour = row.name.hour if hasattr(row.name, "hour") else 12
        if hour == 18 and self.deferred_flex_queue_kw > 0.1:
            flex_deficit = self.deferred_flex_queue_kw
            # Force emergency dispatch
            dispatched_flex_kw += flex_deficit
            self.deferred_flex_queue_kw = 0.0
        else:
            flex_deficit = 0.0

        # 5. Net Building Electrical Demand Calculation (kW)
        # Net Grid Power = Base + HVAC + Lighting + Flexible + Battery_Charge - Solar_PV
        total_building_load_kw = (
            row["base_load_kw"]
            + hvac_elec_kw
            + lighting_elec_kw
            + dispatched_flex_kw
        )
        net_grid_power_kw = total_building_load_kw + batt_grid_power_kw - row["solar_pv_kw"]

        # Track Peak Power Demand
        self.peak_demand_so_far = max(self.peak_demand_so_far, net_grid_power_kw)

        # 6. Reward & Diagnostic Breakdown
        reward, breakdown = self.reward_calc.compute_reward(
            indoor_temp=next_temp,
            occupancy=int(row["occupancy"]),
            net_grid_power_kw=net_grid_power_kw,
            electricity_price=row["electricity_price"],
            battery_degradation_cost=degradation_cost,
            lighting_action=light_act,
            solar_irradiance=row["solar_irradiance"],
            flex_load_deficit_kw=flex_deficit,
        )

        # 7. Update State
        self.indoor_temp = next_temp
        self.current_step += 1

        terminated = False
        truncated = self.current_step >= self.episode_length

        # 8. Record Diagnostics
        step_log = {
            "step": self.current_step,
            "timestamp": str(row.name),
            "indoor_temp": self.indoor_temp,
            "outdoor_temp": float(row["outdoor_temp"]),
            "occupancy": int(row["occupancy"]),
            "solar_pv_kw": float(row["solar_pv_kw"]),
            "base_load_kw": float(row["base_load_kw"]),
            "hvac_elec_kw": float(hvac_elec_kw),
            "lighting_kw": float(lighting_elec_kw),
            "flex_load_kw": float(dispatched_flex_kw),
            "battery_power_kw": float(batt_grid_power_kw),
            "net_grid_power_kw": float(net_grid_power_kw),
            "battery_soc": float(next_soc),
            "electricity_price": float(row["electricity_price"]),
            "reward": float(reward),
            "cost_dollar": float(breakdown["cost_dollar"]),
            "thermal_comfort_penalty": float(breakdown["thermal_comfort_penalty"]),
            "peak_penalty": float(breakdown["peak_penalty"]),
            "hvac_action": float(hvac_act),
            "battery_action": float(batt_act),
            "lighting_action": float(light_act),
            "flex_action": float(flex_act),
        }
        self.history.append(step_log)

        next_obs = self._get_observation()
        info = {
            **breakdown,
            "total_building_load_kw": total_building_load_kw,
            "battery_soc": next_soc,
            "peak_demand_so_far": self.peak_demand_so_far,
        }

        return next_obs, reward, terminated, truncated, info
