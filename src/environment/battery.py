"""Battery Energy Storage System (BESS) Dynamics Model.

Simulates state-of-charge (SoC) evolution, charging/discharging non-linearities,
round-trip efficiency losses, and battery wear/degradation penalties.
"""

from __future__ import annotations
import numpy as np


class BatteryStorage:
    """Models a commercial lithium-ion battery energy storage system."""

    def __init__(
        self,
        capacity_kwh: float = 80.0,
        max_power_kw: float = 25.0,
        min_soc: float = 0.15,
        max_soc: float = 0.95,
        initial_soc: float = 0.40,
        efficiency_charge: float = 0.95,
        efficiency_discharge: float = 0.95,
        degradation_cost_per_kwh: float = 0.012,
    ):
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = max_power_kw
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.soc = initial_soc
        self.eff_charge = efficiency_charge
        self.eff_discharge = efficiency_discharge
        self.degradation_cost_rate = degradation_cost_per_kwh

    def reset(self, initial_soc: float | None = None) -> float:
        """Resets battery state of charge."""
        if initial_soc is not None:
            self.soc = float(np.clip(initial_soc, self.min_soc, self.max_soc))
        else:
            self.soc = 0.40
        return self.soc

    def step(self, action: float, dt_hours: float = 1.0) -> tuple[float, float, float]:
        """Executes a charge or discharge action.

        Args:
            action: [-1.0, 1.0] where >0 is charging, <0 is discharging.
            dt_hours: Time step duration.

        Returns:
            grid_battery_power_kw: Positive if drawing power from grid/solar to charge,
                                   negative if injecting power into building/grid.
            next_soc: Updated state of charge [0.0, 1.0].
            degradation_cost: Estimated battery wear cost in dollars.
        """
        action = float(np.clip(action, -1.0, 1.0))

        if action > 0.01:
            # Charging request
            desired_power = action * self.max_power_kw
            max_energy_can_store = (self.max_soc - self.soc) * self.capacity_kwh
            max_power_limited_by_capacity = (max_energy_can_store / self.eff_charge) / dt_hours
            actual_power = min(desired_power, max_power_limited_by_capacity)

            energy_added = (actual_power * dt_hours) * self.eff_charge
            self.soc = float(np.clip(self.soc + (energy_added / self.capacity_kwh), self.min_soc, self.max_soc))
            grid_battery_power_kw = actual_power
            energy_throughput = energy_added

        elif action < -0.01:
            # Discharging request
            desired_power = (-action) * self.max_power_kw
            max_energy_can_release = (self.soc - self.min_soc) * self.capacity_kwh
            max_power_limited_by_capacity = (max_energy_can_release * self.eff_discharge) / dt_hours
            actual_power = min(desired_power, max_power_limited_by_capacity)

            energy_drawn = (actual_power * dt_hours) / self.eff_discharge
            self.soc = float(np.clip(self.soc - (energy_drawn / self.capacity_kwh), self.min_soc, self.max_soc))
            grid_battery_power_kw = -actual_power
            energy_throughput = energy_drawn

        else:
            grid_battery_power_kw = 0.0
            energy_throughput = 0.0

        degradation_cost = energy_throughput * self.degradation_cost_rate
        return float(grid_battery_power_kw), float(self.soc), float(degradation_cost)
