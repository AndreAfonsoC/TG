import logging
import os

import pandas as pd

from src.aerodynamics import Aerodynamics
from src.mission import MissionManager
from src.turboprop import Turboprop
from utils.aux_tools import (
    SEA_LEVEL_GRAVITY,
    discretize_phase,
)

# --- Configuração do Logger ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
logger = logging.getLogger("main_simulation_turboprop")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Handlers (arquivo e console)
    log_file = os.path.join(LOG_DIR, "main_simulation_turboprop.log")
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-16')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # Formato mais simples
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Configura o pandas
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.options.display.float_format = "{:,.4f}".format

# ==============================================================================
# 1. INPUTS PRINCIPAIS DA SIMULAÇÃO
# ==============================================================================
logger.info("Definindo inputs da simulação...")

# --- Aeronave (ATR 72-600 Reference) ---
AIRCRAFT_PRESET = "atr72-600"
ZERO_FUEL_WEIGHT_KG = 20_400 # Valor típico estimado para ATR 72
NUM_ENGINES = 2

# --- Motor (PW127 Reference) ---
ENGINE_CLASS = Turboprop
REF_POT_TH_TARGET_KW = 2457.0 # Potência termodinâmica de referência (aprox. PW127M takeoff)
MAX_GEARBOX_POWER_KW = 2050.0  # Potência máxima na caixa de engrenagens (aprox. PW127M takeoff)
T04 = 1600.0  # Temperatura de entrada na turbina de referência (K)
PRC = 15.77  # Relação de compressão do compressor
PR_TL = 2.87  # Relação de pressão total no tubo de escape
SEA_LEVEL_AIR_FLOW_KG_S = 8.49

# --- Missão (Regional Típica) ---
MISSION_CHI = 0.3 # Fração inicial de H2
TANK_TYPE = "TYPE_IV"
MAX_SEGMENT_DURATION_MIN = 15
FUEL_GUESS_BOUNDS = (100, 10000) # Limites menores que jato, pois consome menos
USE_AERO_REFINEMENT = True  # Habilita o refinamento do perfil de empuxo via aerodinâmica
USE_DISCRETIZE_PHASES = True  # Habilita a discretização automática de fases longas


# --- Perfil de Voo Base ---
def get_base_flight_profile() -> list:
    """Retorna o perfil de voo base com estimativas de empuxo e dados aero."""
    # Contém dados completos, incluindo os necessários para Aerodynamics
    return [
        {'name': 'Taxi (Saída)',   'duration_min': 5,   'altitude_ft': 0,     'mach': 0.000, 'thrust_percentage': 7, 'roc_ft_min': 0,     'configuration': 'clean',   'burn_strategy': 'kerosene_only'},    # 'kerosene_only', 'proportional', 'hydrogen_only'
        {'name': 'Decolagem',      'duration_min': 1,   'altitude_ft': 0,     'mach': 0.000, 'thrust_percentage': 80,  'roc_ft_min': 0,  'configuration': 'takeoff', 'burn_strategy': 'kerosene_only'},
        {'name': 'Subida 1',       'duration_min': 6,   'altitude_ft': 3330,  'mach': 0.225, 'thrust_percentage': 35,  'roc_ft_min': 1600,  'configuration': 'clean',   'burn_strategy': 'kerosene_only'},
        {'name': 'Subida 2',       'duration_min': 6,   'altitude_ft': 10000, 'mach': 0.315, 'thrust_percentage': 25,  'roc_ft_min': 1000,  'configuration': 'clean',   'burn_strategy': 'proportional'},
        {'name': 'Subida 3',       'duration_min': 6,   'altitude_ft': 16667, 'mach': 0.405, 'thrust_percentage': 15,   'roc_ft_min': 500,  'configuration': 'clean',   'burn_strategy': 'proportional'},
        {'name': 'Cruzeiro',       'duration_min': 25, 'altitude_ft': 20000, 'mach': 0.450, 'thrust_percentage': 12,  'roc_ft_min': 0,     'configuration': 'clean',   'burn_strategy': 'proportional'},
        {'name': 'Loiter',         'duration_min': 25,  'altitude_ft': 14000, 'mach': 0.300, 'thrust_percentage': 10,   'roc_ft_min': 0,     'configuration': 'clean',   'burn_strategy': 'proportional'},
        {'name': 'Descida 1',      'duration_min': 6,   'altitude_ft': 16667, 'mach': 0.405, 'thrust_percentage': 10,   'roc_ft_min': -1000, 'configuration': 'clean',   'burn_strategy': 'proportional'},
        {'name': 'Descida 2',      'duration_min': 6,   'altitude_ft': 10000, 'mach': 0.315, 'thrust_percentage': 10,   'roc_ft_min': -1200, 'configuration': 'clean',   'burn_strategy': 'proportional'},
        {'name': 'Descida 3',      'duration_min': 6,   'altitude_ft': 3330,  'mach': 0.225, 'thrust_percentage': 10,   'roc_ft_min': -700, 'configuration': 'landing', 'burn_strategy': 'proportional'},
        {'name': 'Pouso',          'duration_min': 1,   'altitude_ft': 0,     'mach': 0.180, 'thrust_percentage': 7,   'roc_ft_min': -400,  'configuration': 'landing', 'burn_strategy': 'proportional'},
        {'name': 'Taxi (Chegada)', 'duration_min': 5,   'altitude_ft': 0,     'mach': 0.000, 'thrust_percentage': 7,   'roc_ft_min': 0,     'configuration': 'clean',   'burn_strategy': 'proportional'},
    ]


# ==============================================================================
# 2. FUNÇÕES AUXILIARES
# ==============================================================================

def calibrate_engine(engine_config: dict) -> ENGINE_CLASS:
    """Calibra uma instância do motor e salva o ponto de projeto."""
    logger.info("\n\n" + "=" * 120)
    logger.info("FASE DE CONFIGURAÇÃO: Calibrando o motor para o ponto de projeto...")
    logger.info("\n" + "=" * 120 + "\n")

    engine_instance = ENGINE_CLASS(engine_config)
    engine_instance.set_sea_level_air_flow(SEA_LEVEL_AIR_FLOW_KG_S)
    engine_instance.calibrate_pot_th()

    logger.info("Motor calibrado com sucesso!")

    return engine_instance


def run_simulation_stage(stage_name: str, mission_manager: MissionManager, flight_profile: list, mission_chi: float,
                         tank_type: str) -> dict:
    """Configura, executa e retorna os resultados de um estágio da simulação."""
    logger.info("\n\n" + "=" * 120)
    logger.info(f"CONFIGURANDO {stage_name}")
    logger.info("\n" + "=" * 120 + "\n")

    mission_manager.clear_mission()  # Limpa fases anteriores

    discretized_profile = []
    for phase in flight_profile:
        if phase['duration_min'] > MAX_SEGMENT_DURATION_MIN and USE_DISCRETIZE_PHASES:
            logger.info(f"Discretizando fase '{phase['name']}' em segmentos de ~{MAX_SEGMENT_DURATION_MIN} min...")
            discretized_profile.extend(discretize_phase(phase, MAX_SEGMENT_DURATION_MIN))
        else:
            discretized_profile.append(phase)

    logger.info("Adicionando fases ao MissionManager:")
    for phase in discretized_profile:
        mission_manager.add_phase(
            name=phase['name'],
            duration_min=phase['duration_min'],
            altitude_ft=phase['altitude_ft'],
            mach=phase['mach'],
            thrust_percentage=phase['thrust_percentage'],  # Passa o percentual (fixo ou refinado)
            roc_ft_min=phase['roc_ft_min'],
            configuration=phase['configuration'],
            burn_strategy=phase['burn_strategy']
        )

    # Executa a solução da missão
    mission_manager.solve_mission_fuel(
        chi_burning=mission_chi,
        tank_type=tank_type,
        fuel_guess_bounds=FUEL_GUESS_BOUNDS
    )

    return mission_manager.results  # Retorna os resultados completos


def display_final_results(stage_name: str, results: dict, chi: float, zfw: float):
    """Exibe os resultados finais consolidados de uma simulação."""
    logger.info("\n\n" + "=" * 120)  # Separador visual
    logger.info(f"RESULTADOS CONSOLIDADOS - {stage_name}")
    logger.info("\n" + "=" * 120 + "\n")

    if results:
        fs_final_obj = results["final_fuel_system_object"]
        logger.info(f"Combustível Total Necessário: {results['Combustível Inicial (kg)']:.2f} kg")
        logger.info(f"  - H2 Inicial: {fs_final_obj.hydrogen_mass_initial:.2f} kg")
        logger.info(f"  - QAV Inicial: {fs_final_obj.kerosene_mass_initial:.2f} kg")
        tanque_peso = fs_final_obj.get_tank_weight()
        logger.info(f"Peso Tanque H2 ({fs_final_obj.tank_type}): {tanque_peso:.2f} kg")
        logger.info(f"Combustível + Tanque H2 ({fs_final_obj.tank_type}): {tanque_peso + results['Combustível Inicial (kg)']:.2f} kg")
        tow = zfw + fs_final_obj.get_total_weight_at_takeoff()
        logger.info(f"Peso Total Decolagem (TOW): {tow:,.2f} kg")
        logger.info("\n--- EMISSÕES TOTAIS ---")
        logger.info(f"CO2: {results['Emissão Total de CO2 (kg)']:.2f} kg")
        logger.info(f"H2O: {results['Emissão Total de H2O (kg)']:.2f} kg")
        logger.info("\n--- RESERVA FINAL ---")
        logger.info(f"H2 Remanescente: {fs_final_obj.get_remaining_hydrogen():.2f} kg")
        logger.info(f"QAV Remanescente: {fs_final_obj.get_remaining_kerosene():.2f} kg")
        # Log do DataFrame resumido
        logger.info("\n--- RESUMO POR FASE ---")
        logger.info(f"\n{results['summary_df'].round(3).to_string()}")
    else:
        logger.warning(f"Simulação do estágio '{stage_name}' não concluída.")


# ==============================================================================
# 3. EXECUÇÃO PRINCIPAL (DUAS ETAPAS)
# ==============================================================================

if __name__ == "__main__":

    # --- Calibração Inicial do Motor ---
    initial_engine_config = {
        'mach': 0.0,
        'altitude': 0.0,
        'hydrogen_fraction': 0.0,
        "pr_tl": PR_TL,
        "prc": PRC,
        "T04": T04,  # (K)
        "max_gearbox_power": MAX_GEARBOX_POWER_KW,  # kW -> se não fornecido, pode ser considerado como 0.8 * Pot_th (termodinâmica)
        "ref_pot_th": REF_POT_TH_TARGET_KW,  # kW -> se não fornecido, pode ser considerado como 0.8 * Pot_th (termodinâmica)
    }
    calibrated_engine = calibrate_engine(initial_engine_config)

    # --- Cria Modelo Aerodinâmico (para cálculo intermediário) ---
    aero_model = Aerodynamics.from_preset(AIRCRAFT_PRESET)

    # --- ETAPA 1: Simulação com Empuxo Fixo Estimado ---

    # Cria MissionManager
    mission_stage1_manager = MissionManager(engine=calibrated_engine, zero_fuel_weight=ZERO_FUEL_WEIGHT_KG,
                                            num_engines=NUM_ENGINES)

    # Obtém perfil base e define estratégia de queima
    profile_stage1 = get_base_flight_profile()

    # Confere envelope da Etapa 1
    rated_thrust_total_kN = calibrated_engine._design_point['rated_thrust'] * NUM_ENGINES
    for phase in profile_stage1:
        envelope = calibrated_engine.get_thrust_envelope(
            mach=phase['mach'],
            altitude_ft=phase['altitude_ft']
        )
        max_thrust_perc = aero_model.convert_thrust_to_percentage(
            envelope['max_thrust_kN'] * NUM_ENGINES,
            rated_thrust_total_kN
        )
        perc_thrust = phase['thrust_percentage']
        if perc_thrust > max_thrust_perc:
            raise ValueError(
                f"Fase '{phase['name']}': Empuxo necessário ({perc_thrust:.1f}%) excede máximo possível "
                f"({max_thrust_perc:.1f}%).")

    # Roda Etapa 1
    results_stage1 = run_simulation_stage(
        "ETAPA 1 (Empuxo Fixo)",
        mission_stage1_manager,
        profile_stage1,
        MISSION_CHI,
        TANK_TYPE
    )

    if not results_stage1:
        logger.error("Simulação da Etapa 1 falhou. Abortando.")
        exit()

    # Exibe resultados da Etapa 1
    display_final_results("ETAPA 1 (Empuxo Fixo)", results_stage1, MISSION_CHI, ZERO_FUEL_WEIGHT_KG)

    if USE_AERO_REFINEMENT:
        # --- CÁLCULO INTERMEDIÁRIO: Refinar Perfil de Empuxo ---
        logger.info("\n\n" + "=" * 120)  # Separador visual
        logger.info("CÁLCULO INTERMEDIÁRIO: Refinando Perfil de Empuxo usando Aerodinâmica")
        logger.info("\n" + "=" * 120 + "\n")

        df_detailed_stage1 = results_stage1['detailed_df']
        profile_stage2 = []

        for index, phase_s1 in df_detailed_stage1.iterrows():
            new_phase_s2 = phase_s1.to_dict()  # Copia todos os dados da fase discretizada

            # Calcula empuxo dinâmico APENAS se não for uma fase com empuxo fixo conhecido
            # Usa .split('_')[0] para pegar o nome base da fase (ex: 'Cruzeiro' de 'Cruzeiro_1')
            phase_base_name = phase_s1['Fase'].split('_')[0]
            if phase_base_name not in ['Taxi (Saída)', 'Decolagem', 'Descida 1', 'Descida 2', 'Descida 3', 'Pouso',
                                       'Taxi (Chegada)']:
                weight_N_s1 = phase_s1['Peso Médio (kg)'] * SEA_LEVEL_GRAVITY
                thrust_req_kN_s2 = aero_model.get_required_thrust_kN(
                    weight_N=weight_N_s1, altitude_ft=phase_s1['Altitude (ft)'], mach=phase_s1['Mach'],
                    roc_ft_min=phase_s1['ROC (ft/min)'], configuration=phase_s1['Configuração']
                )
                perc_thrust_s2 = aero_model.convert_thrust_to_percentage(thrust_req_kN_s2, rated_thrust_total_kN)

                envelope = calibrated_engine.get_thrust_envelope(
                    mach=phase_s1['Mach'],
                    altitude_ft=phase_s1['Altitude (ft)']
                )
                max_thrust_perc = aero_model.convert_thrust_to_percentage(
                    envelope['max_thrust_kN'] * NUM_ENGINES,
                    rated_thrust_total_kN
                )

                if perc_thrust_s2 > max_thrust_perc:
                    logger.warning(
                        f"Fase '{phase_s1['Fase']}': Empuxo calculado ({perc_thrust_s2:.1f}%) excede máximo ({max_thrust_perc:.1f}%). "
                        f"Modificando-o para o máximo possível...")
                    perc_thrust_s2 = max_thrust_perc
                else:
                    logger.info(
                        f"Fase '{phase_s1['Fase']}': Empuxo calculado ({perc_thrust_s2:.1f}%) dentro do envelope ({max_thrust_perc:.1f}%)."
                    )

                new_phase_s2['thrust_percentage'] = perc_thrust_s2  # Atualiza com o valor calculado
                logger.debug(
                    f"Fase '{new_phase_s2['Fase']}': Empuxo dinâmico calculado: {thrust_req_kN_s2:.2f} kN ({perc_thrust_s2:.1f}%)")
            else:
                # Mantém o thrust_percentage original para fases fixas
                original_phase = next((p for p in get_base_flight_profile() if p['name'] == phase_base_name), None)
                new_phase_s2['thrust_percentage'] = original_phase.get('thrust_percentage',7.0)  # Usa original ou fallback
                logger.debug(
                    f"Fase '{new_phase_s2['Fase']}': Mantendo empuxo fixo de {new_phase_s2['thrust_percentage']:.1f}%")

            # Configurando a fase para que tudo funcione corretamente na próxima simulação
            new_phase_s2['name'] = phase_s1['Fase']
            new_phase_s2['duration_min'] = phase_s1['Duração (min)']
            new_phase_s2['altitude_ft'] = phase_s1['Altitude (ft)']
            new_phase_s2['mach'] = phase_s1['Mach']
            new_phase_s2['roc_ft_min'] = phase_s1['ROC (ft/min)']
            new_phase_s2['configuration'] = phase_s1['Configuração']
            new_phase_s2['burn_strategy'] = phase_s1['Estratégia de Queima']
            profile_stage2.append(new_phase_s2)

        # --- ETAPA 2: Simulação com Empuxo Refinado ---
        # Reutiliza o mesmo MissionManager, apenas limpando e adicionando as novas fases
        # O motor já está configurado com o MISSION_CHI correto
        results_stage2 = run_simulation_stage(
            "ETAPA 2 (Empuxo Refinado)",
            mission_stage1_manager,
            profile_stage2,
            MISSION_CHI,
            TANK_TYPE
        )

        if not results_stage2:
            logger.error("Simulação da Etapa 2 falhou.")
        else:
            # Exibe resultados finais
            display_final_results("ETAPA 2 (Empuxo Refinado)", results_stage2, MISSION_CHI, ZERO_FUEL_WEIGHT_KG)
            logger.info("\nSimulação completa de duas etapas concluída com sucesso.")
