import math
from typing import Literal, Dict, List

import numpy as np

# Conversão de unidades
ft2m = 0.3048
kt2ms = 0.514444
lb2N = 4.44822
min2s = 60.0

# Condições ao nível do mar padrão
SEA_LEVEL_GRAVITY = 9.80665  # m/s²

# ==============================================================================
# FATORES DE EMISSÃO (BASEADO EM ESTEQUIOMETRIA)
# ==============================================================================
# Reação do Querosene (aproximado como C12H23):
# C12H23 + 17.75 O2 -> 12 CO2 + 11.5 H2O
# Massa molar C12H23 ≈ 167 g/mol | Massa molar CO2 = 44 g/mol | Massa molar H2O = 18 g/mol
# Fator CO2 = (12 * 44) / 167 ≈ 3.15
# Fator H2O = (11.5 * 18) / 167 ≈ 1.24
CO2_PER_KEROSENE_MASS = 3.15  # kg de CO2 por kg de querosene queimado; fonte: https://ansperformance.eu/economics/cba/standard-inputs/latest/chapters/amount_of_emissions_released_by_fuel_burn.html
H2O_PER_KEROSENE_MASS = 1.237  # kg de H2O por kg de querosene queimado; fonte: https://ansperformance.eu/economics/cba/standard-inputs/latest/chapters/amount_of_emissions_released_by_fuel_burn.html
KEROSENE_PCI = 45000  # Poder calorífico inferior do querosene [kJ/kg]; fonte: Turns, S. R. (2012). An Introduction to Combustion: Concepts and Applications (3rd ed.). McGraw-Hill.

# Reação do Hidrogênio:
# 2 H2 + O2 -> 2 H2O
# Massa molar H2 ≈ 2 g/mol | Massa molar H2O = 18 g/mol
# Fator H2O = (2 * 18) / (2 * 2) ≈ 8.94
H2O_PER_HYDROGEN_MASS = 8.94  # kg de H2O por kg de hidrogênio queimado; fonte: Turns, S. R. (2012). An Introduction to Combustion: Concepts and Applications (3rd ed.). McGraw-Hill.
H2_PCI = 120000  # Poder calorífico inferior do hidrogênio [kJ/kg]; fonte: Turns, S. R. (2012). An Introduction to Combustion: Concepts and Applications (3rd ed.). McGraw-Hill.


# ==============================================

def atmosphere(z, Tba=288.15):
    '''
    Funçao que retorna a Temperatura, Pressao e Densidade para uma determinada
    altitude z [m]. Essa funçao usa o modelo padrao de atmosfera para a
    temperatura no solo de Tba.
    '''

    # Zbase (so para referencia)
    # 0 11019.1 20063.1 32161.9 47350.1 50396.4

    # DEFINING CONSTANTS
    # Earth radius
    r = 6356766
    # gravity
    g0 = SEA_LEVEL_GRAVITY
    # air gas constant
    R = 287.05287
    # layer boundaries
    Ht = [0, 11000, 20000, 32000, 47000, 50000]
    # temperature slope in each layer
    A = [-6.5e-3, 0, 1e-3, 2.8e-3, 0]
    # pressure at the base of each layer
    pb = [101325, 22632, 5474.87, 868.014, 110.906]
    # temperature at the base of each layer
    Tstdb = [288.15, 216.65, 216.65, 228.65, 270.65];
    # temperature correction
    Tb = Tba - Tstdb[0]
    # air viscosity
    mu0 = 18.27e-6  # [Pa s]
    T0 = 291.15  # [K]
    C = 120  # [K]

    # geopotential altitude
    H = r * z / (r + z)

    # selecting layer
    if H < Ht[0]:
        raise ValueError('Under sealevel')
    elif H <= Ht[1]:
        i = 0
    elif H <= Ht[2]:
        i = 1
    elif H <= Ht[3]:
        i = 2
    elif H <= Ht[4]:
        i = 3
    elif H <= Ht[5]:
        i = 4
    else:
        raise ValueError('Altitude beyond model boundaries')

    # Calculating temperature
    T = Tstdb[i] + A[i] * (H - Ht[i]) + Tb

    # Calculating pressure
    if A[i] == 0:
        p = pb[i] * np.exp(-g0 * (H - Ht[i]) / R / (Tstdb[i] + Tb))
    else:
        p = pb[i] * (T / (Tstdb[i] + Tb)) ** (-g0 / A[i] / R)

    # Calculating density
    rho = p / R / T

    # Calculating viscosity with Sutherland's Formula
    mu = mu0 * (T0 + C) / (T + C) * (T / T0) ** (1.5)

    return T, p, rho, mu

# Condições ao nível do mar padrão
SEA_LEVEL_TEMPERATURE, SEA_LEVEL_PRESSURE, SEA_LEVEL_DENSITY, SEA_LEVEL_VISCOSITY = atmosphere(0.0)
SEA_LEVEL_PRESSURE = SEA_LEVEL_PRESSURE / 1000.0  # Converter Pa para kPa

def calculate_energy_from_fuel(
        consumed_fuel_kg: float,
        chi: float,
        burn_strategy: Literal["proportional", "hydrogen_only", "kerosene_only"] = "proportional",
        kerosene_pci_kJ_kg: float = 45000.0,
        hydrogen_pci_kJ_kg: float = 120000.0,
) -> dict:
    """
    Calcula a energia liberada por cada tipo de combustível com base na massa consumida.

    Args:
        consumed_fuel_kg (float): Massa total de combustível consumida.
        chi (float): Fração mássica de hidrogênio na mistura.
        burn_strategy (str): Estratégia de queima ('proportional', 'hydrogen_only', 'kerosene_only').
        kerosene_pci_kJ_kg (float): Poder calorífico inferior do querosene [kJ/kg].
        hydrogen_pci_kJ_kg (float): Poder calorífico inferior do hidrogênio [kJ/kg].

    Returns:
        dict: Dicionário com a energia liberada por cada combustível [kJ].
    """
    energy_h2_kJ = 0.0
    energy_qav_kJ = 0.0

    if burn_strategy == 'hydrogen_only':
        energy_h2_kJ = consumed_fuel_kg * hydrogen_pci_kJ_kg
    elif burn_strategy == 'kerosene_only':
        energy_qav_kJ = consumed_fuel_kg * kerosene_pci_kJ_kg
    elif burn_strategy == 'proportional':
        h2_consumed = consumed_fuel_kg * chi
        qav_consumed = consumed_fuel_kg * (1 - chi)
        energy_h2_kJ = h2_consumed * hydrogen_pci_kJ_kg
        energy_qav_kJ = qav_consumed * kerosene_pci_kJ_kg

    return {'energy_h2_kJ': energy_h2_kJ, 'energy_qav_kJ': energy_qav_kJ}


def calculate_fuel_consumption_breakdown(
        consumed_fuel_kg: float,
        chi: float,
        burn_strategy: str
) -> dict:
    """
    Decompõe a massa total de combustível consumida nas massas de H2 e querosene.

    Args:
        consumed_fuel_kg (float): Massa total de combustível consumida na fase.
        chi (float): Fração mássica de hidrogênio na mistura inicial da missão.
        burn_strategy (str): Estratégia de queima da fase.

    Returns:
        dict: Dicionário com as massas de cada combustível consumido [kg].
    """
    h2_consumed_kg = 0.0
    qav_consumed_kg = 0.0

    if burn_strategy == 'hydrogen_only':
        h2_consumed_kg = consumed_fuel_kg
    elif burn_strategy == 'kerosene_only':
        qav_consumed_kg = consumed_fuel_kg
    elif burn_strategy == 'proportional':
        # Só realiza a decomposição se houver uma mistura real
        if 0.0 < chi < 1.0:
            h2_consumed_kg = consumed_fuel_kg * chi
            qav_consumed_kg = consumed_fuel_kg * (1 - chi)
        elif chi == 1.0:  # Missão 100% H2
            h2_consumed_kg = consumed_fuel_kg
        else:  # Missão 100% QAV
            qav_consumed_kg = consumed_fuel_kg

    return {'h2_consumed_kg': h2_consumed_kg, 'qav_consumed_kg': qav_consumed_kg}


def discretize_phase(phase_data: Dict, max_segment_duration_min: float) -> List[Dict]:
    """
    Discretiza uma fase de voo longa em segmentos menores de duração aproximadamente igual.

    Args:
        phase_data (Dict): Dicionário contendo os dados da fase original
                           (incluindo 'name', 'duration_min', e outros parâmetros).
        max_segment_duration_min (float): A duração máxima desejada para cada segmento [minutos].

    Returns:
        List[Dict]: Uma lista de dicionários, onde cada dicionário representa um segmento
                    da fase original, com duração ajustada e nome sequencial.
                    Retorna a lista original com um único elemento se a duração
                    original já for menor ou igual à duração máxima do segmento.
    """
    original_duration = phase_data.get('duration_min', 0.0)
    original_name = phase_data.get('name', 'UnnamedPhase')

    # Se a fase já for curta o suficiente, não discretiza
    if original_duration <= max_segment_duration_min:
        return [phase_data.copy()]  # Retorna uma cópia para evitar modificação do original

    # Calcula o número de segmentos necessários
    num_segments = math.ceil(original_duration / max_segment_duration_min)

    # Calcula a duração de cada segmento (distribui igualmente)
    segment_duration = original_duration / num_segments

    discretized_phases = []
    for i in range(num_segments):
        # Cria uma cópia dos dados originais para o novo segmento
        new_phase = phase_data.copy()

        # Atualiza o nome e a duração
        new_phase['name'] = f"{original_name}_{i + 1}"
        new_phase['duration_min'] = segment_duration

        # Adiciona o novo segmento à lista
        discretized_phases.append(new_phase)

    return discretized_phases
