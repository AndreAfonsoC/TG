from typing import Literal, Optional

import numpy as np

from utils.aux_tools import (
    atmosphere,
    ft2m,
    min2s,
    SEA_LEVEL_TEMPERATURE,
)


class Aerodynamics:
    """
    Modela a aerodinâmica de uma aeronave para calcular o arrasto e o empuxo requerido.

    Esta classe utiliza uma polar de arrasto parabólica, uma abordagem padrão em análise de
    desempenho de aeronaves, para determinar o empuxo necessário em diferentes fases do voo.

    Referências Principais:
    - Anderson, J. D. (2016). "Introduction to Flight".
    - Raymer, D. P. (2018). "Aircraft Design: A Conceptual Approach".

    A polar de arrasto é definida por (Anderson, "Introduction to Flight", Cap. 6):
    $$ C_D = C_{D0} + C_{Di} = C_{D0} + k \cdot C_L^2 $$
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
            aircraft_type (str, opcional): Tipo de aeronave para estimar parâmetros ('jet_airliner', 'turboprop').
            cd0 (float, opcional): Coeficiente de arrasto parasita base (configuração limpa, baixo Mach).
                                   Se não fornecido, será estimado com base no tipo de aeronave.
            e (float, opcional): Fator de eficiência de Oswald. Se não fornecido, será estimado.
            sweep_angle_deg (float, opcional): Ângulo de enflechamento da asa a 25% da corda [graus].
                                               Usado para estimar 'e'.
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
            # A estimativa de 'e' é baseada no princípio de que o enflechamento da asa reduz
            # a eficiência da envergadura, diminuindo o fator de Oswald.
            # (Referência: Raymer, "Aircraft Design", Cap. 12, Seção 12.5).
            # A lógica abaixo é uma simplificação baseada em valores típicos para diferentes classes de enflechamento.
            if self.sweep_angle_deg >= 30:
                self.e = 0.80  # Típico para jatos comerciais com enflechamento moderado/alto
            elif self.sweep_angle_deg >= 10:
                self.e = 0.82  # Intermediário
            else:
                self.e = 0.85  # Típico para asas retas ou com baixo enflechamento (turbo-hélices)
        else:
            self.e = e

        # 3. Estimar Coeficiente de Arrasto Parasita 'cd0' (se não fornecido)
        if cd0 is None:
            # Valores de CD0 são empíricos e baseados em dados de aeronaves similares.
            # (Referência: Raymer, "Aircraft Design", Tabela 12.2 - "Component Drag Buildup").
            if self.aircraft_type == 'jet_airliner':
                self.cd0_base = 0.022  # Típico para um B737-800 em configuração limpa
            elif self.aircraft_type == 'turboprop':
                self.cd0_base = 0.027  # Típico para um ATR-72 (maior arrasto de forma)
            else:
                self.cd0_base = 0.025  # Padrão genérico
        else:
            self.cd0_base = cd0

        # Fator de arrasto induzido (k) é constante para uma dada geometria
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
        Calcula o empuxo total requerido pela aeronave para uma condição de voo.

        Opera em dois modos:
        1. MODO DE CÁLCULO (padrão): Calcula o empuxo necessário com base nas equações de movimento.
           - Voo nivelado (ROC=0): T = D
           - Voo em subida (ROC > 0): T = D + W * sin(gamma)
           (Referência: Anderson, "Introduction to Flight", Cap. 6, Seção 6.4).
        2. MODO DE OVERRIDE: Se `percentage_of_rated_thrust` for fornecido, ignora os cálculos
           aerodinâmicos e retorna o empuxo correspondente.

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

        # --- MODO DE CÁLCULO AERODINÂMICO ---
        temp_K, _, rho_kg_m3, _ = atmosphere(
            altitude_ft * ft2m,
            Tba=SEA_LEVEL_TEMPERATURE + delta_isa_temperature_K
        )
        sound_speed_ms = np.sqrt(1.4 * 287.05 * temp_K)
        velocity_ms = mach * sound_speed_ms
        q_Pa = 0.5 * rho_kg_m3 * velocity_ms ** 2

        # $$ C_L = \frac{W}{q \cdot S} $$
        # (Referência: Anderson, "Introduction to Flight", Cap. 5, Seção 5.3).
        cl = weight_N / (q_Pa * self.S_m2)

        # Correção de compressibilidade de Prandtl-Glauert para o arrasto de onda em regime subsônico.
        # $$ C_{D0,comp} = \frac{C_{D0}}{\sqrt{1 - M^2}} $$
        # Nota: Esta é uma simplificação que corrige principalmente o arrasto de pressão. O arrasto de
        # atrito também varia com o Mach de forma não linear, mas para estudos conceituais, esta correção é
        # uma aproximação aceita. (Referência: Anderson, "Fundamentals of Aerodynamics", Cap. 11).
        if mach < 0.95:
            cd0_comp = self.cd0_base / np.sqrt(1 - mach ** 2)
        else:
            cd0_comp = self.cd0_base

        # Incremento de arrasto para configuração de high-lift (flaps/trem de pouso).
        # Valores de incremento baseados em dados empíricos.
        # (Referência: Raymer, "Aircraft Design", Tabela 12.4 - "Drag increments for flaps and landing gear").
        delta_cd0 = 0.0
        if configuration == 'takeoff':
            delta_cd0 = 0.020
        elif configuration == 'landing':
            delta_cd0 = 0.075
        cd0_total = cd0_comp + delta_cd0

        cd_total = cd0_total + self.k * cl ** 2
        drag_N = cd_total * q_Pa * self.S_m2

        # Componente do peso devido à taxa de subida.
        # $$ \sin(\gamma) = \frac{ROC}{V} $$
        roc_ms = roc_ft_min * ft2m / min2s
        sin_gamma = roc_ms / velocity_ms if velocity_ms > 0 else 0
        weight_component_N = weight_N * sin_gamma

        required_thrust_N = drag_N + weight_component_N

        return required_thrust_N / 1000

    def convert_thrust_to_percentage(
            self,
            thrust_kN: float,
            rated_thrust_kN_total: float
    ) -> float:
        """
        Converte um valor de empuxo total em kN para uma porcentagem do empuxo nominal total.

        Args:
            thrust_kN (float): Empuxo total a ser convertido [kN].
            rated_thrust_kN_total (float): Empuxo nominal total de referência (soma de todos os motores) [kN].

        Returns:
            float: O empuxo como uma porcentagem (0 a 100).
        """
        if rated_thrust_kN_total <= 0:
            return 0.0
        return (thrust_kN / rated_thrust_kN_total) * 100


if __name__ == '__main__':
    # --- Exemplo para o B737-800 ---
    b737_aero = Aerodynamics(
        S_m2=124.6,
        AR=9.45,
        aircraft_type='jet_airliner'
    )

    # Exemplo do MODO DE CÁLCULO (Cruzeiro)
    peso_cruzeiro_N = (61500) * 9.81
    empuxo_cruzeiro_kN = b737_aero.get_required_thrust_kN(
        weight_N=peso_cruzeiro_N,
        altitude_ft=35000,
        mach=0.789
    )
    # print(f"B737 - Empuxo Requerido em Cruzeiro (Total): {empuxo_cruzeiro_kN:.2f} kN")
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
