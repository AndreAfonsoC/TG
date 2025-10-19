from typing import Literal, Optional

import numpy as np

from utils.aux_tools import atmosphere, ft2m, min2s


class Aerodynamics:
    """
    Modela a aerodinâmica de uma aeronave para calcular o arrasto e o empuxo requerido.

    Esta classe utiliza uma polar de arrasto parabólica para determinar o empuxo necessário
    em diferentes fases do voo, considerando efeitos de configuração, compressibilidade
    e taxa de subida. É genérica e pode ser configurada para diferentes tipos de aeronaves,
    como jatos comerciais ou turbo-hélices.

    A polar de arrasto é definida por:
    $$ C_D = C_{D0} + k \cdot C_L^2 $$
    Onde o fator de arrasto induzido 'k' é:
    $$ k = \frac{1}{\pi \cdot AR \cdot e} $$
    """

    def __init__(
            self,
            S_m2: float,
            AR: float,
            aircraft_type: Optional[Literal['jet_airliner', 'turboprop']] = None,
            cd0: Optional[float] = None,
            e: Optional[float] = None,
            sweep_angle_deg: Optional[float] = None
    ) -> None:
        """
        Inicializa o modelo aerodinâmico da aeronave.

        Args:
            S_m2 (float): Área de referência da asa [m²].
            AR (float): Razão de aspecto da asa (adimensional).
            aircraft_type (str, opcional): Tipo de aeronave para estimar parâmetros.
            cd0 (float, opcional): Coeficiente de arrasto parasita base. Se não fornecido, será estimado.
            e (float, opcional): Fator de eficiência de Oswald. Se não fornecido, será estimado.
            sweep_angle_deg (float, opcional): Ângulo de enflechamento da asa [graus]. Usado para estimar 'e'.
        """
        self.S_m2 = S_m2
        self.AR = AR
        self.aircraft_type = aircraft_type

        # --- Lógica para estimar parâmetros não fornecidos ---

        # 1. Estimar Enflechamento (se não fornecido)
        if sweep_angle_deg is None:
            if self.aircraft_type == 'jet_airliner':
                self.sweep_angle_deg = 33.0  # Típico para B737
            elif self.aircraft_type == 'turboprop':
                self.sweep_angle_deg = 5.0  # Típico para ATR-72 (asa quase reta)
            else:
                self.sweep_angle_deg = 0.0  # Padrão genérico
        else:
            self.sweep_angle_deg = sweep_angle_deg

        # 2. Estimar Fator de Oswald 'e' (se não fornecido)
        if e is None:
            if self.aircraft_type == 'jet_airliner':
                self.e = 0.80  # Valor conservador para jatos comerciais modernos
            elif self.aircraft_type == 'turboprop':
                self.e = 0.84  # Valor típico para turbo-hélices com asa de baixo enflechamento
            else:
                self.e = 0.82  # Padrão genérico
        else:
            self.e = e

        # 3. Estimar Coeficiente de Arrasto Parasita 'cd0' (se não fornecido)
        if cd0 is None:
            if self.aircraft_type == 'jet_airliner':
                self.cd0_base = 0.022  # Típico para um B737-800 em configuração limpa
            elif self.aircraft_type == 'turboprop':
                self.cd0_base = 0.027  # Típico para um ATR-72 (maior arrasto de forma)
            else:
                self.cd0_base = 0.025  # Padrão genérico
        else:
            self.cd0_base = cd0

        self.k = 1 / (np.pi * self.AR * self.e)

    def get_required_thrust_kN(
            self,
            weight_N: float,
            altitude_ft: float,
            mach: float,
            roc_ft_min: float = 0.0,
            delta_isa_temperature_K: float = 0.0,
            configuration: Literal['clean', 'takeoff', 'landing'] = 'clean',
            # --- PARÂMETROS ADICIONADOS PARA O MODO DE "OVERRIDE" ---
            percentage_of_rated_thrust: Optional[float] = None,
            rated_thrust_kN_per_engine: Optional[float] = None,
            num_engines: int = 2
    ) -> float:
        """
        Calcula o empuxo total requerido pela aeronave (soma de todos os motores) para uma condição de voo.

        Este méthodo opera em dois modos:
        1. MODO DE CÁLCULO (padrão): Se `percentage_of_rated_thrust` for None, calcula o empuxo
           necessário para superar o arrasto e a componente do peso.
        2. MODO DE OVERRIDE: Se `percentage_of_rated_thrust` for fornecido, ignora todos os
           cálculos aerodinâmicos e retorna o empuxo total correspondente a essa porcentagem.

        Args:
            weight_N (float): Peso total da aeronave na condição de voo [Newtons].
            altitude_ft (float): Altitude de voo [pés].
            mach (float): Número de Mach de voo.
            roc_ft_min (float, opcional): Taxa de subida [pés por minuto]. Padrão é 0.
            delta_isa_temperature_K (float, opcional): Variação da temperatura em relação à ISA [K].
            configuration (str, opcional): Configuração da aeronave ('clean', 'takeoff', 'landing').
            percentage_of_rated_thrust (float, opcional): Se fornecido, ativa o modo de override. Valor de 0 a 100.
            rated_thrust_kN_per_engine (float, opcional): Empuxo nominal por motor [kN]. Obrigatório se usar o modo de override.
            num_engines (int): Número de motores na aeronave.

        Returns:
            float: Empuxo total requerido pela aeronave [kN].
        """
        # --- MODO DE OVERRIDE ---
        if percentage_of_rated_thrust is not None:
            if rated_thrust_kN_per_engine is None:
                raise ValueError(
                    "`rated_thrust_kN_per_engine` deve ser fornecido ao usar `percentage_of_rated_thrust`.")

            total_rated_thrust = rated_thrust_kN_per_engine * num_engines
            return total_rated_thrust * (percentage_of_rated_thrust / 100.0)

        # --- MODO DE CÁLCULO AERODINÂMICO (lógica original) ---
        temp_K, press_Pa, rho_kg_m3, sound_speed_ms = atmosphere(altitude_ft * ft2m,
                                                                 Tba=delta_isa_temperature_K)
        velocity_ms = mach * sound_speed_ms
        q_Pa = 0.5 * rho_kg_m3 * velocity_ms ** 2

        cl = weight_N / (q_Pa * self.S_m2)

        if mach < 0.95:
            cd0_comp = self.cd0_base / np.sqrt(1 - mach ** 2)
        else:
            cd0_comp = self.cd0_base

        delta_cd0 = 0.0
        if configuration == 'takeoff':
            delta_cd0 = 0.020
        elif configuration == 'landing':
            delta_cd0 = 0.075
        cd0_total = cd0_comp + delta_cd0

        cd_total = cd0_total + self.k * cl ** 2
        drag_N = cd_total * q_Pa * self.S_m2

        roc_ms = roc_ft_min * ft2m / min2s
        sin_gamma = roc_ms / velocity_ms if velocity_ms > 0 else 0
        weight_component_N = weight_N * sin_gamma

        required_thrust_N = drag_N + weight_component_N

        return required_thrust_N / 1000

    def convert_thrust_to_percentage(
            self,
            thrust_kN: float,
            rated_thrust_kN: float
    ) -> float:
        """
        Converte um valor de empuxo em kN para uma porcentagem do empuxo nominal.

        Args:
            thrust_kN (float): Empuxo a ser convertido [kN].
            rated_thrust_kN (float): Empuxo nominal de referência do motor [kN].

        Returns:
            float: O empuxo como uma porcentagem (0 a 100).
        """
        if rated_thrust_kN <= 0:
            return 0.0
        # O empuxo de entrada é o TOTAL da aeronave, assim como o nominal de referência.
        return (thrust_kN / rated_thrust_kN) * 100


if __name__ == '__main__':
    # --- Exemplo para o B737-800 ---
    b737_aero = Aerodynamics(
        S_m2=124.6,
        AR=9.45,
        aircraft_type='jet_airliner'
    )

    # Exemplo do MODO DE CÁLCULO (Cruzeiro)
    peso_cruzeiro_N = (70000) * 9.81
    empuxo_cruzeiro_kN = b737_aero.get_required_thrust_kN(
        weight_N=peso_cruzeiro_N,
        altitude_ft=35000,
        mach=0.789
    )
    print(f"B737 - Empuxo Requerido em Cruzeiro (Total): {empuxo_cruzeiro_kN:.2f} kN")
    print(f"B737 - Empuxo por Motor (2 motores): {empuxo_cruzeiro_kN / 2:.2f} kN")

    # Exemplo do MODO DE OVERRIDE (Decolagem)
    empuxo_decolagem_kN = b737_aero.get_required_thrust_kN(
        weight_N=0,  # Ignorado no modo override
        altitude_ft=0,  # Ignorado no modo override
        mach=0,  # Ignorado no modo override
        percentage_of_rated_thrust=100.0,
        rated_thrust_kN_per_engine=121.4,
        num_engines=2
    )
    print(f"\nB737 - Empuxo Total em Decolagem (100%): {empuxo_decolagem_kN:.2f} kN")
