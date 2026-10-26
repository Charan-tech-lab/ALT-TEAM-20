"""Building Thermodynamics and Heat Pump Physics Simulation Model.

Implements lumped-parameter Resistance-Capacitance (RC) equivalent thermal networks
modeling building envelope conduction, solar insolation through glazing, internal
sensible heat generation from occupants and plug loads, and heat pump COP dynamics.
"""

from __future__ import annotations
import numpy as np


class BuildingThermodynamics:
    """Simulates realistic thermal dynamics for a multi-zone commercial/residential smart building."""

    def __init__(
        self,
        c_air_kwh_per_c: float = 18.0,      # Thermal capacity of indoor space (kWh/°C)
        r_envelope_c_per_kw: float = 2.4,   # Envelope thermal resistance (°C/kW)
        window_area_m2: float = 85.0,       # Window glazing surface area
        shgc: float = 0.35,                 # Solar Heat Gain Coefficient of glazing
        max_hvac_cooling_kw: float = 45.0,  # Max electrical power of HVAC heat pump (kW)
        max_hvac_heating_kw: float = 35.0,  # Max electrical power of heating mode (kW)
        nominal_cop_cooling: float = 3.6,   # Nominal Coefficient of Performance
        nominal_cop_heating: float = 3.2,
        person_sensible_heat_kw: float = 0.09, # Sensible heat per occupant (90W)
    ):
        self.c_air = c_air_kwh_per_c
        self.r_env = r_envelope_c_per_kw
        self.window_area = window_area_m2
        self.shgc = shgc
        self.max_cooling_kw = max_hvac_cooling_kw
        self.max_heating_kw = max_hvac_heating_kw
        self.nom_cop_cool = nominal_cop_cooling
        self.nom_cop_heat = nominal_cop_heating
        self.person_heat = person_sensible_heat_kw

    def compute_cop_cooling(self, outdoor_temp: float) -> float:
        """COP drops at extreme outdoor ambient temperatures."""
        cop = self.nom_cop_cool - 0.05 * (outdoor_temp - 24.0)
        return float(np.clip(cop, 1.8, 5.0))

    def compute_cop_heating(self, outdoor_temp: float) -> float:
        """Heating COP drops in freezing ambient temperatures."""
        cop = self.nom_cop_heat + 0.04 * (outdoor_temp - 7.0)
        return float(np.clip(cop, 1.5, 4.5))

    def step(
        self,
        current_indoor_temp: float,
        outdoor_temp: float,
        solar_irradiance: float,
        occupancy: int,
        base_load_kw: float,
        lighting_power_kw: float,
        hvac_action: float,  # [-1.0 (max heating) to +1.0 (max cooling)]
        dt_hours: float = 1.0,
    ) -> tuple[float, float, float]:
        """Advances indoor temperature by dt_hours under HVAC and environmental forcing.

        Returns:
            next_indoor_temp: (°C)
            hvac_elec_power_kw: Electrical power consumed by HVAC (kW)
            hvac_thermal_kw: Thermal power transferred to/from zone (kW)
        """
        # 1. Envelope heat conduction (positive if outdoor is warmer than indoor)
        q_conduction = (outdoor_temp - current_indoor_temp) / self.r_env

        # 2. Solar radiation through glazed fenestration (kW)
        q_solar = (self.window_area * self.shgc * solar_irradiance) / 1000.0

        # 3. Internal sensible heat gains (kW)
        q_occupancy = occupancy * self.person_heat
        q_plug_loads = base_load_kw * 0.75  # ~75% of electrical base load turns to heat
        q_lighting = lighting_power_kw * 0.85
        q_internal = q_occupancy + q_plug_loads + q_lighting

        # 4. HVAC active heating/cooling
        hvac_elec_power_kw = 0.0
        q_hvac_thermal = 0.0  # Positive heats room, negative cools room

        if hvac_action > 0.01:
            # Active Cooling mode
            cooling_fraction = np.clip(hvac_action, 0.0, 1.0)
            hvac_elec_power_kw = cooling_fraction * self.max_cooling_kw
            cop = self.compute_cop_cooling(outdoor_temp)
            q_hvac_thermal = -1.0 * hvac_elec_power_kw * cop  # Heat removed
        elif hvac_action < -0.01:
            # Active Heating mode
            heating_fraction = np.clip(-hvac_action, 0.0, 1.0)
            hvac_elec_power_kw = heating_fraction * self.max_heating_kw
            cop = self.compute_cop_heating(outdoor_temp)
            q_hvac_thermal = heating_fraction * self.max_heating_kw * cop  # Heat added

        # 5. Energy balance differential: C * (dT/dt) = sum(Q)
        net_thermal_rate_kw = q_conduction + q_solar + q_internal + q_hvac_thermal
        temp_delta = (net_thermal_rate_kw * dt_hours) / self.c_air
        next_indoor_temp = float(np.clip(current_indoor_temp + temp_delta, 10.0, 45.0))

        return next_indoor_temp, float(hvac_elec_power_kw), float(q_hvac_thermal)
