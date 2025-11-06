import math
from typing import Literal, Optional, Dict

import numpy as np

from utils.aux_tools import (
    atmosphere,  # atmosphere(h[m], Tba[K]) -> (T[K], p[Pa], rho[kg/m^3], mu[Pa*s])
    ft2m,
    min2s,
    SEA_LEVEL_TEMPERATURE,
)


# ---------------------------------------------------------------------
# Aerodynamic utilities (conceptual/professional-level corrections)
# ---------------------------------------------------------------------

def _beta(M: float) -> float:
    """Return beta = sqrt(1 - M^2) with numerical protection."""
    return float(np.sqrt(max(1e-9, 1.0 - M ** 2)))


def _PG_factor(M: float) -> float:
    """Return the Prandtl-Glauert compressibility factor for pressure-derived quantities."""
    return 1.0 / _beta(M)


def _KT_factor(M: float, Cp0_mag: float = 0.6) -> float:
    """
    Return the Karman-Tsien correction factor, more stable than PG as M -> 1.
    Typical form adapted for the correction of mean pressure coefficients.
    """
    b = _beta(M)
    return (1.0 / b) * (1.0 + 0.2 * M ** 2) / (1.0 + 0.5 * M ** 2 * (1.0 + Cp0_mag / b))


def _wave_drag_korn(M: float, t_c: float, sweep_LE_deg: float, CL: float) -> float:
    """
    Estimate wave drag using a simplified form of Korn's equation plus 20 drag counts.

    Equations:
        M_dd = 0.95 / cos(Lambda) - (t/c)/cos^2(Lambda) - CL / (10*cos^3(Lambda))
        M_c  = M_dd - 0.1 / (80^(1/3))
        C_D,wave = 20*(M - M_c)^4*1e-4, if M > M_c; otherwise 0.

    Parameters
    ----------
    M : float
        Mach number.
    t_c : float
        Typical airfoil thickness ratio (e.g., 0.12).
    sweep_LE_deg : float
        Leading-edge sweep angle [deg].
    CL : float
        Lift coefficient at the given condition.

    Returns
    -------
    float
        Wave drag coefficient.
    """
    cosl = math.cos(math.radians(sweep_LE_deg))
    cosl = max(1e-6, cosl)
    Mdd = 0.95 / cosl - t_c / (cosl ** 2) - CL / (10.0 * cosl ** 3)
    Mc = Mdd - 0.1 / (80.0 ** (1.0 / 3.0))
    if M <= Mc:
        return 0.0
    CD_wave_counts = 20.0 * (M - Mc) ** 4
    return CD_wave_counts * 1e-4


def _estimate_oswald_e(AR: float, sweep_LE_deg: float) -> float:
    """
    Estimate Oswald efficiency factor (e) according to Raymer Eq. 12.48-12.49.

    Parameters
    ----------
    AR : float
        Wing aspect ratio.
    sweep_LE_deg : float
        Leading-edge sweep angle [deg].

    Returns
    -------
    float
        Oswald efficiency factor (dimensionless).
    """
    lam = math.radians(sweep_LE_deg)
    if sweep_LE_deg <= 0.0:
        e = 1.78 * (1 - 0.045 * AR ** 0.68) - 0.64
    elif sweep_LE_deg >= 30.0:
        e = 4.61 * (1 - 0.045 * AR ** 0.68) * (math.cos(lam)) ** 0.15 - 3.1
    else:
        e_straight = 1.78 * (1 - 0.045 * AR ** 0.68) - 0.64
        e_swept = 4.61 * (1 - 0.045 * AR ** 0.68) * (math.cos(lam)) ** 0.15 - 3.1
        f = sweep_LE_deg / 30.0
        e = e_straight + f * (e_swept - e_straight)
    return max(0.6, min(0.95, e))  # practical range limitation


# =====================================================================
# Main class - aerodynamic model
# =====================================================================

class Aerodynamics:
    """
    Aerodynamic model for estimating drag and required thrust.

    The aerodynamic polar follows a parabolic form:
        C_D = C_D0 + k*C_L^2, with k = 1 / (pi*AR*e)

    Main improvements implemented:
    1. Compressibility correction applied only to the **pressure** portion of C_D0
       (Prandtl-Glauert or Karman-Tsien). The **friction** term receives a mild
       correction (~1 + 0.12*M^2) for compressible boundary-layer effects.
    2. Wave drag (Korn) model near and beyond critical Mach number.
    """

    PRESETS: Dict[str, Dict] = {
        # Typical values consistent with Raymer/ATA documentation.
        "b737-800": dict(S_m2=124.6, AR=9.45, sweep_angle_deg=25.0, t_over_c=0.12, e=0.82,
                         cd0=0.0205, f_friction=0.65, aircraft_type='jet_airliner'),
        "a320-200": dict(S_m2=122.6, AR=9.4, sweep_angle_deg=25.0, t_over_c=0.12, e=0.82,
                         cd0=0.0210, f_friction=0.65, aircraft_type='jet_airliner'),
        "atr72-600": dict(S_m2=61.0, AR=12.7, sweep_angle_deg=5.0, t_over_c=0.15, e=0.85,
                          cd0=0.0260, f_friction=0.70, aircraft_type='turboprop'),
        "e195-e2": dict(S_m2=92.5, AR=9.3, sweep_angle_deg=25.0, t_over_c=0.12, e=0.83,
                        cd0=0.0215, f_friction=0.65, aircraft_type='jet_airliner'),
    }

    def __init__(
            self,
            S_m2: float,
            AR: float,
            aircraft_type: Optional[Literal['jet_airliner', 'turboprop']] = None,
            cd0: Optional[float] = None,
            e: Optional[float] = None,
            sweep_angle_deg: Optional[float] = None,
            t_over_c: Optional[float] = None,
            method_cd0_comp: Literal['PG', 'KT'] = 'KT',
            include_wave: bool = True,
            f_friction: Optional[float] = None,
    ) -> None:
        self.S_m2 = float(S_m2)
        self.AR = float(AR)
        self.aircraft_type = aircraft_type

        # Leading-edge sweep angle (default by aircraft type)
        if sweep_angle_deg is None:
            if aircraft_type == 'jet_airliner':
                self.sweep_angle_deg = 25.0
            elif aircraft_type == 'turboprop':
                self.sweep_angle_deg = 5.0
            else:
                self.sweep_angle_deg = 0.0
        else:
            self.sweep_angle_deg = float(sweep_angle_deg)

        # Oswald efficiency factor (default by configuration)
        if e is None:
            self.e = _estimate_oswald_e(self.AR, self.sweep_angle_deg)
        else:
            self.e = float(e)

        # Zero-lift drag coefficient at low Mach
        if cd0 is None:
            self.cd0_base = 0.022 if self.aircraft_type == 'jet_airliner' else 0.027
        else:
            self.cd0_base = float(cd0)

        # Friction-to-pressure drag split within CD0
        if f_friction is None:
            self.f_friction = 0.65 if self.aircraft_type == 'jet_airliner' else 0.70
        else:
            self.f_friction = float(f_friction)

        self.t_over_c = 0.12 if t_over_c is None else float(t_over_c)
        self.method_cd0_comp = method_cd0_comp
        self.include_wave = bool(include_wave)

        # Induced drag factor
        self.k = 1.0 / (np.pi * self.AR * self.e)

    @classmethod
    def from_preset(cls, name: str) -> "Aerodynamics":
        """Instantiate a predefined aircraft configuration by name."""
        p = cls.PRESETS.get(name.lower())
        if p is None:
            raise ValueError(f"Preset '{name}' not found. Available: {list(cls.PRESETS.keys())}")
        return cls(**p)

    def get_required_thrust_kN(
            self,
            weight_N: float,
            altitude_ft: float,
            mach: float,
            roc_ft_min: float = 0.0,
            delta_isa_temperature_K: float = 0.0,
            configuration: Literal['clean', 'takeoff', 'landing'] = 'clean',
            percentage_of_rated_thrust: Optional[float] = None,
            rated_thrust_kN_per_engine: Optional[float] = None,
            num_engines: int = 2
    ) -> float:
        """
        Compute total required thrust (kN) for a given flight condition.

        Level flight:   T = D
        Climb segment:  T = D + W*sin(gamma), with sin(gamma) ~ ROC / V

        Improvements included:
        - C_D0(M) = C_D0,press*F_comp(M) + C_D0,fric*(1 + 0.12*M^2)
        - C_D,wave (Korn) when enabled

        Parameters
        ----------
        weight_N : float
            Aircraft weight [N].
        altitude_ft : float
            Altitude [ft].
        mach : float
            Mach number.
        roc_ft_min : float, optional
            Rate of climb [ft/min].
        delta_isa_temperature_K : float, optional
            Temperature deviation from ISA [K].
        configuration : {'clean', 'takeoff', 'landing'}
            Aircraft aerodynamic configuration.
        percentage_of_rated_thrust : float, optional
            Override mode: return percentage of nominal thrust.
        rated_thrust_kN_per_engine : float, optional
            Nominal engine thrust (per engine) [kN].
        num_engines : int
            Number of engines.

        Returns
        -------
        float
            Total required thrust [kN].
        """

        if percentage_of_rated_thrust is not None:
            if rated_thrust_kN_per_engine is None:
                raise ValueError("`rated_thrust_kN_per_engine` is required when using override mode.")
            total_rated = rated_thrust_kN_per_engine * num_engines
            return total_rated * (percentage_of_rated_thrust / 100.0)

        T_K, _, rho, _ = atmosphere(
            altitude_ft * ft2m,
            Tba=SEA_LEVEL_TEMPERATURE + delta_isa_temperature_K
        )
        a_ms = float(np.sqrt(1.4 * 287.05 * T_K))
        V_ms = float(mach * a_ms)
        q = 0.5 * rho * V_ms ** 2

        CL = weight_N / (q * self.S_m2)

        if mach >= 0.78 and mach < 1.0:
            self.method_cd0_comp = "KT"
        CD0_fric0 = self.cd0_base * self.f_friction
        CD0_press0 = self.cd0_base * (1.0 - self.f_friction)

        F_fric = 1.0 + 0.12 * mach ** 2
        if self.method_cd0_comp == 'PG':
            F_comp = _PG_factor(mach)
        else:
            F_comp = _KT_factor(mach, Cp0_mag=0.6)

        CD0_comp = CD0_fric0 * F_fric + CD0_press0 * F_comp

        delta_cd0 = 0.0
        if configuration == 'takeoff':
            delta_cd0 = 0.020
        elif configuration == 'landing':
            delta_cd0 = 0.075

        CD0_total = CD0_comp + delta_cd0

        CDi = self.k * CL ** 2
        CDw = _wave_drag_korn(mach, self.t_over_c, self.sweep_angle_deg, CL) if self.include_wave else 0.0
        CD = CD0_total + CDi + CDw

        D_N = CD * q * self.S_m2

        roc_ms = roc_ft_min * ft2m / min2s
        sin_gamma = roc_ms / V_ms if V_ms > 0.0 else 0.0
        T_req_N = D_N + weight_N * sin_gamma

        return T_req_N / 1000.0  # [kN]

    def convert_thrust_to_percentage(self, thrust_kN: float, rated_thrust_kN_total: float) -> float:
        """Convert total thrust (kN) to percentage of nominal total thrust."""
        if rated_thrust_kN_total <= 0:
            return 0.0
        return (thrust_kN / rated_thrust_kN_total) * 100.0
