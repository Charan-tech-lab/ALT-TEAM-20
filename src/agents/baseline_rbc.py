"""Conventional Rule-Based Controller (RBC) for Building Energy Management.

Represents industry-standard Building Automation System (BAS) logic:
- Fixed occupied/unoccupied thermostatic setpoints with deadbands
- Scheduled lighting timer without daylight harvesting
- Naive battery self-consumption without Time-of-Use price awareness
- Unscheduled immediate flexible load dispatch
"""

from __future__ import annotations
import numpy as np
from .base_agent import BaseAgent


class RuleBasedController(BaseAgent):
    """Industry-standard Rule-Based Controller (RBC) benchmark."""

    def __init__(self, discrete_actions: bool = False, num_discrete_actions: int = 60):
        super().__init__(state_dim=12, action_dim=4 if not discrete_actions else num_discrete_actions, name="RuleBasedControl")
        self.discrete_actions = discrete_actions

    def select_action(self, state: np.ndarray, evaluate: bool = True) -> np.ndarray | int:
        """Computes rule-based HVAC, lighting, battery, and flexible load actions.

        State format:
        [indoor_temp_norm, outdoor_temp_norm, ma3_norm, occupancy_norm, solar_pv_norm,
         base_load_norm, price_norm, soc_norm, peak_norm, hr_sin, hr_cos, is_workday]
        """
        # Un-normalize key quantities for physical threshold logic
        # indoor_temp was scaled: 2 * (temp - 15) / (32 - 15) - 1
        t_in_norm = state[0]
        indoor_temp = ((t_in_norm + 1.0) / 2.0) * (32.0 - 15.0) + 15.0

        occ_norm = state[3]
        occupancy = ((occ_norm + 1.0) / 2.0) * 120.0

        hr_sin = state[9]
        hr_cos = state[10]
        # Calculate approximate hour of day [0, 24)
        hour = (np.arctan2(hr_sin, hr_cos) * 12.0 / np.pi) % 24.0

        is_workday = state[11] > 0.5
        is_occupied_time = is_workday and (8.0 <= hour <= 18.0) and (occupancy > 5)

        # 1. HVAC Thermostatic Heuristics
        if is_occupied_time:
            target_temp = 22.5
            deadband = 0.5
            if indoor_temp > target_temp + deadband:
                # Cooling proportional to deviation
                hvac = min(1.0, 0.4 + 0.3 * (indoor_temp - target_temp))
            elif indoor_temp < target_temp - deadband:
                # Heating
                hvac = -min(1.0, 0.4 + 0.3 * (target_temp - indoor_temp))
            else:
                hvac = 0.0
        else:
            # Unoccupied Night/Weekend Setback
            cooling_setback = 26.5
            heating_setback = 18.0
            if indoor_temp > cooling_setback:
                hvac = 0.5
            elif indoor_temp < heating_setback:
                hvac = -0.5
            else:
                hvac = 0.0

        # 2. Scheduled Lighting Control (Fixed timer, no daylight dimming)
        if is_occupied_time:
            lighting = 1.0  # Full 100% illumination
        else:
            lighting = 0.1  # Standby / safety lighting

        # 3. Naive Battery Storage (Dumb solar self-consumption: no TOU rate awareness)
        # Charges when solar is available, discharges when solar is 0
        solar_norm = state[4]
        solar_pv = ((solar_norm + 1.0) / 2.0) * 65.0
        if solar_pv > 15.0:
            battery = 0.6  # Charge from solar
        elif hour >= 19.0 or hour <= 6.0:
            battery = -0.3  # Mild discharge at night
        else:
            battery = 0.0

        # 4. Immediate Flexible Load Dispatch
        flex = 1.0  # Run immediately without load shifting

        if self.discrete_actions:
            # Map continuous rule values to the nearest discrete action index
            # For simplicity in evaluation, return index 0 or closest combination
            return 0

        return np.array([hvac, lighting, battery, flex], dtype=np.float32)
