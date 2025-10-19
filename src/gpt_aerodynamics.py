import math
from typing import Literal, Optional, Dict

import numpy as np

from utils.aux_tools import (
    atmosphere,  # atmosphere(h[m], Tba[K]) -> (T[K], p[Pa], rho[kg/m³], mu[Pa·s])
    ft2m,
    min2s,
    SEA_LEVEL_TEMPERATURE,
)


# ---------------------------------------------------------------------
# Utilidades aerodinâmicas (correções em nível conceitual/profissional)
# ---------------------------------------------------------------------

def _beta(M: float) -> float:
    """β = √(1-M²) com proteção numérica."""
    return float(np.sqrt(max(1e-9, 1.0 - M ** 2)))


def _PG_factor(M: float) -> float:
    """Fator de Prandtl–Glauert para grandezas derivadas do campo de pressão."""
    return 1.0 / _beta(M)


def _KT_factor(M: float, Cp0_mag: float = 0.6) -> float:
    """
    Fator de Kármán–Tsien (mais estável que PG conforme M→1).
    Forma típica adaptada para correção de coeficientes de pressão médios.
    """
    b = _beta(M)
    # Numerador suave: (1 + 0.2 M²); termo no denominador com Cp0 reduz blow-up
    return (1.0 / b) * (1.0 + 0.2 * M ** 2) / (1.0 + 0.5 * M ** 2 * (1.0 + Cp0_mag / b))


def _wave_drag_korn(M: float, t_c: float, sweep_LE_deg: float, CL: float) -> float:
    """
    Drag de onda por uma forma simples da equação de Korn + '20 drag counts'.

    M_dd = 0.95/cosΛ - (t/c)/cos²Λ - CL/(10·cos³Λ)
    M_c  = M_dd - 0.1/80^(1/3)
    C_D,wave = 20·(M - M_c)^4·1e-4  se M > M_c; caso contrário, 0.

    Entradas:
        t_c  : espessura relativa típica do aerofólio (fração, ex.: 0.12)
        sweep_LE_deg : enflechamento na borda de ataque [graus]
        CL   : coeficiente de sustentação na condição
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
    Estima o fator de eficiência de Oswald (e) conforme Raymer Eq. 12.48–12.49.

    Args:
        AR (float): razão de aspecto da asa
        sweep_LE_deg (float): enflechamento na borda de ataque [graus]
    Returns:
        float: fator e (adimensional)
    """
    lam = math.radians(sweep_LE_deg)
    if sweep_LE_deg <= 0.0:
        e = 1.78 * (1 - 0.045 * AR ** 0.68) - 0.64
    elif sweep_LE_deg >= 30.0:
        e = 4.61 * (1 - 0.045 * AR ** 0.68) * (math.cos(lam)) ** 0.15 - 3.1
    else:
        # interpolação linear 0–30°
        e_straight = 1.78 * (1 - 0.045 * AR ** 0.68) - 0.64
        e_swept = 4.61 * (1 - 0.045 * AR ** 0.68) * (math.cos(lam)) ** 0.15 - 3.1
        f = sweep_LE_deg / 30.0
        e = e_straight + f * (e_swept - e_straight)
    return max(0.6, min(0.95, e))  # limita a faixa prática


# =====================================================================
# Classe principal – compatível com a sua interface
# =====================================================================

class Aerodynamics:
    """
    Modelo aerodinâmico para cálculo de arrasto e empuxo requerido.

    Polar parabólica:
        C_D = C_{D0} + K C_L^2,   com K = 1/(π·AR·e).

    Melhorias importantes (literatura clássica):
    1) Correção de compressibilidade aplicada somente ao termo de **pressão** do C_{D0}
       (Prandtl–Glauert ou Kármán–Tsien). O termo de **fricção** recebe um fator leve
       ~ (1 + 0.12 M²) para efeitos compressíveis em camada-limite turbulenta.
    2) Drag de onda (Korn) próximo/acima de M_crit.
    """

    # -------------------------- PRESETS --------------------------
    PRESETS: Dict[str, Dict] = {
        # Valores típicos (ordem de grandeza coerente com Raymer/ATA docs).
        # S[m²], AR[-], sweep_25[deg], t/c[-], e[-], CD0_base[-], split fricção
        "b737-800": dict(S_m2=124.6, AR=9.45, sweep_angle_deg=25.0, t_over_c=0.12, e=0.82, cd0=0.0205, f_friction=0.65,
                         aircraft_type='jet_airliner'),
        "a320-200": dict(S_m2=122.6, AR=9.4, sweep_angle_deg=25.0, t_over_c=0.12, e=0.82, cd0=0.0210, f_friction=0.65,
                         aircraft_type='jet_airliner'),
        "atr72-600": dict(S_m2=61.0, AR=12.7, sweep_angle_deg=5.0, t_over_c=0.15, e=0.85, cd0=0.0260, f_friction=0.70,
                          aircraft_type='turboprop'),
        "e195-e2": dict(S_m2=92.5, AR=9.3, sweep_angle_deg=25.0, t_over_c=0.12, e=0.83, cd0=0.0215, f_friction=0.65,
                        aircraft_type='jet_airliner'),
    }

    def __init__(
            self,
            S_m2: float,
            AR: float,
            aircraft_type: Optional[Literal['jet_airliner', 'turboprop']] = None,
            cd0: Optional[float] = None,
            e: Optional[float] = None,
            sweep_angle_deg: Optional[float] = None,
            # --------- NOVOS (opcionais; não quebram chamadas antigas) ---------
            t_over_c: Optional[float] = None,  # espessura relativa p/ Korn
            method_cd0_comp: Literal['PG', 'KT'] = 'KT',  # méthodo de compressibilidade
            include_wave: bool = True,  # habilita Korn
            f_friction: Optional[float] = None,  # fração de CD0 que é fricção
    ) -> None:
        self.S_m2 = float(S_m2)
        self.AR = float(AR)
        self.aircraft_type = aircraft_type

        # Enflechamento (25% da corda) – default por tipo
        if sweep_angle_deg is None:
            if aircraft_type == 'jet_airliner':
                self.sweep_angle_deg = 25.0
            elif aircraft_type == 'turboprop':
                self.sweep_angle_deg = 5.0
            else:
                self.sweep_angle_deg = 0.0
        else:
            self.sweep_angle_deg = float(sweep_angle_deg)

        # Fator de Ostwald – default por tipo
        if e is None:
            self.e = _estimate_oswald_e(self.AR, self.sweep_angle_deg)
        else:
            self.e = float(e)

        # CD0 base (baixo Mach, limpo)
        if cd0 is None:
            self.cd0_base = 0.022 if self.aircraft_type == 'jet_airliner' else 0.027
        else:
            self.cd0_base = float(cd0)

        # Split fricção × pressão no CD0 (fração de fricção)
        if f_friction is None:
            self.f_friction = 0.65 if self.aircraft_type == 'jet_airliner' else 0.70
        else:
            self.f_friction = float(f_friction)

        # Espessura relativa p/ Korn
        self.t_over_c = 0.12 if t_over_c is None else float(t_over_c)

        # Méthodo de compressibilidade e onda
        self.method_cd0_comp = method_cd0_comp
        self.include_wave = bool(include_wave)

        # Fator de arrasto induzido
        self.k = 1.0 / (np.pi * self.AR * self.e)

    # ---------------------- FACTORY DE PRESETS ----------------------
    @classmethod
    def from_preset(cls, name: str) -> "Aerodynamics":
        """Cria uma instância a partir de um nome de preset (ex.: 'b737-800')."""
        p = cls.PRESETS.get(name.lower())
        if p is None:
            raise ValueError(f"Preset '{name}' não encontrado. Opções: {list(cls.PRESETS.keys())}")
        return cls(**p)

    # ---------------------- CÁLCULO DE EMPUXO ----------------------
    def get_required_thrust_kN(
            self,
            weight_N: float,
            altitude_ft: float,
            mach: float,
            roc_ft_min: float = 0.0,
            delta_isa_temperature_K: float = 0.0,
            configuration: Literal['clean', 'takeoff', 'landing'] = 'clean',
            # modo override (mantidos)
            percentage_of_rated_thrust: Optional[float] = None,
            rated_thrust_kN_per_engine: Optional[float] = None,
            num_engines: int = 2
    ) -> float:
        """
        Empuxo total requerido (kN) para a condição indicada.

        - Nivelado: T = D
        - Subida:   T = D + W·sinγ, com sinγ ≈ ROC/V

        Melhorias:
        - C_{D0}(M) = C_{D0,press}·F_comp(M) + C_{D0,fric}·(1+0.12 M²)
        - C_{D,wave} (Korn) se habilitado
        """

        # ---------- OVERRIDE ----------
        if percentage_of_rated_thrust is not None:
            if rated_thrust_kN_per_engine is None:
                raise ValueError("`rated_thrust_kN_per_engine` é obrigatório no modo override.")
            total_rated = rated_thrust_kN_per_engine * num_engines
            return total_rated * (percentage_of_rated_thrust / 100.0)

        # ---------- ATMOSFERA & VELOCIDADE ----------
        T_K, _, rho, _ = atmosphere(
            altitude_ft * ft2m,
            Tba=SEA_LEVEL_TEMPERATURE + delta_isa_temperature_K
        )
        a_ms = float(np.sqrt(1.4 * 287.05 * T_K))
        V_ms = float(mach * a_ms)
        q = 0.5 * rho * V_ms ** 2

        # ---------- SUSTENTAÇÃO ----------
        CL = weight_N / (q * self.S_m2)

        # ---------- C_D0 COM CORREÇÃO DE COMPRESSIBILIDADE ----------
        if mach >= 0.78 and mach < 1.0:
            self.method_cd0_comp = "KT"  # mais estável perto do transônico
        CD0_fric0 = self.cd0_base * self.f_friction
        CD0_press0 = self.cd0_base * (1.0 - self.f_friction)

        F_fric = 1.0 + 0.12 * mach ** 2
        if self.method_cd0_comp == 'PG':
            F_comp = _PG_factor(mach)
        else:
            F_comp = _KT_factor(mach, Cp0_mag=0.6)  # mais suave perto do transônico

        CD0_comp = CD0_fric0 * F_fric + CD0_press0 * F_comp

        # ---------- INCREMENTOS DE CONFIGURAÇÃO ----------
        # (Raymer – valores típicos)
        delta_cd0 = 0.0
        if configuration == 'takeoff':
            delta_cd0 = 0.020
        elif configuration == 'landing':
            delta_cd0 = 0.075

        CD0_total = CD0_comp + delta_cd0

        # ---------- INDUZIDO + ONDA ----------
        CDi = self.k * CL ** 2
        CDw = _wave_drag_korn(mach, self.t_over_c, self.sweep_angle_deg, CL) if self.include_wave else 0.0
        CD = CD0_total + CDi + CDw

        # ---------- ARRASTO & EMPUXO ----------
        D_N = CD * q * self.S_m2

        roc_ms = roc_ft_min * ft2m / min2s
        sin_gamma = roc_ms / V_ms if V_ms > 0.0 else 0.0
        T_req_N = D_N + weight_N * sin_gamma

        return T_req_N / 1000.0  # kN

    def convert_thrust_to_percentage(self, thrust_kN: float, rated_thrust_kN_total: float) -> float:
        """Converte empuxo total (kN) em % do total nominal."""
        if rated_thrust_kN_total <= 0:
            return 0.0
        return (thrust_kN / rated_thrust_kN_total) * 100.0


if __name__ == "__main__":
    # --- B737-800 em cruzeiro ---
    b737 = Aerodynamics.from_preset("b737-800")

    peso_cruzeiro_N = 61_500 * 9.80665  # 61.5 t
    Treq_kN = b737.get_required_thrust_kN(
        weight_N=peso_cruzeiro_N,
        altitude_ft=35_000,
        mach=0.789,
        configuration='clean'
    )
    print(f"B737-800 | Cruzeiro M0.789 @ FL350 | Treq total = {Treq_kN:.1f} kN  (~{Treq_kN / 2:.1f} kN por motor)")

    # --- ATR 72-600 em cruzeiro (exemplo) ---
    atr = Aerodynamics.from_preset("atr72-600")
    peso_N = 22_000 * 9.80665
    Treq_kN = atr.get_required_thrust_kN(
        weight_N=peso_N, altitude_ft=25_000, mach=0.45, configuration='clean'
    )
    print(f"ATR 72-600 | M0.45 @ FL250 | Treq total ≈ {Treq_kN:.1f} kN")
    # (Para turbo-hélice, costuma-se converter para potência: P ≈ T·V/η_p)
