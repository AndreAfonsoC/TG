import logging
import os

import pandas as pd

from src.aerodynamics import Aerodynamics
from src.mission import MissionManager
from src.turbofan import Turbofan
from utils.aux_tools import (
    SEA_LEVEL_GRAVITY,
    discretize_phase,
)

# --- Configuração do Logger ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
logger = logging.getLogger("main_simulation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # Handlers (arquivo e console)
    log_file = os.path.join(LOG_DIR, "main_simulation.log")
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

# --- Aeronave ---
AIRCRAFT_PRESET = "b737-800"
ZERO_FUEL_WEIGHT_KG = 61_500
NUM_ENGINES = 2

# --- Motor ---
ENGINE_CLASS = Turbofan
RATED_THRUST_TARGET_KN = 121.4  # Por motor
FUEL_FLOW_TAKEOFF_TARGET_KGS = 1.293  # Por motor
T04_BOUNDS_CALIBRATION = (1400, 1800)
MDOT_BOUNDS_CALIBRATION = (300, 600)

# --- Missão ---
MISSION_CHI = 0.1  # Fração inicial de H2
TANK_TYPE = "TYPE_IV"
MAX_SEGMENT_DURATION_MIN = 15
FUEL_GUESS_BOUNDS = (10, 200e3)
USE_AERO_REFINEMENT = True  # Habilita o refinamento do perfil de empuxo via aerodinâmica
USE_DISCRETIZE_PHASES = True  # Habilita a discretização automática de fases longas


# --- Perfil de Voo Base ---
def get_base_flight_profile() -> list:
    """Retorna o perfil de voo base com estimativas de empuxo e dados aero."""
    # Contém dados completos, incluindo os necessários para Aerodynamics
    return [
        {'name': 'Taxi (Saída)',   'duration_min': 1,   'altitude_ft': 0,     'mach': 0.000, 'thrust_percentage': 7, 'roc_ft_min': 0,     'configuration': 'clean',   'burn_strategy': 'kerosene_only'},    # 'kerosene_only', 'proportional', 'hydrogen_only'
        {'name': 'Decolagem',      'duration_min': 1,   'altitude_ft': 0,     'mach': 0.000, 'thrust_percentage': 100,  'roc_ft_min': 0,  'configuration': 'takeoff', 'burn_strategy': 'proportional'},
        {'name': 'Subida 1',       'duration_min': 8,   'altitude_ft': 5830,  'mach': 0.298, 'thrust_percentage': 50,  'roc_ft_min': 2400,  'configuration': 'clean',   'burn_strategy': 'proportional'},
        {'name': 'Subida 2',       'duration_min': 8,   'altitude_ft': 17500, 'mach': 0.494, 'thrust_percentage': 30,  'roc_ft_min': 1500,  'configuration': 'clean',   'burn_strategy': 'proportional'},
        {'name': 'Subida 3',       'duration_min': 8,   'altitude_ft': 29170, 'mach': 0.691, 'thrust_percentage': 20,   'roc_ft_min': 750,  'configuration': 'clean',   'burn_strategy': 'proportional'},
        {'name': 'Cruzeiro',       'duration_min': 150, 'altitude_ft': 35000, 'mach': 0.789, 'thrust_percentage': 18,  'roc_ft_min': 0,     'configuration': 'clean',   'burn_strategy': 'proportional'},
        {'name': 'Loiter',         'duration_min': 45,  'altitude_ft': 15000, 'mach': 0.400, 'thrust_percentage': 15,   'roc_ft_min': 0,     'configuration': 'clean',   'burn_strategy': 'proportional'},
        {'name': 'Descida 1',      'duration_min': 8,   'altitude_ft': 29170, 'mach': 0.691, 'thrust_percentage': 10,   'roc_ft_min': -1500, 'configuration': 'clean',   'burn_strategy': 'proportional'},
        {'name': 'Descida 2',      'duration_min': 8,   'altitude_ft': 17500, 'mach': 0.494, 'thrust_percentage': 10,   'roc_ft_min': -2000, 'configuration': 'clean',   'burn_strategy': 'proportional'},
        {'name': 'Descida 3',      'duration_min': 8,   'altitude_ft': 5830,  'mach': 0.298, 'thrust_percentage': 10,   'roc_ft_min': -1000, 'configuration': 'landing', 'burn_strategy': 'proportional'},
        {'name': 'Pouso',          'duration_min': 1,   'altitude_ft': 0,     'mach': 0.200, 'thrust_percentage': 7,   'roc_ft_min': -500,  'configuration': 'landing', 'burn_strategy': 'proportional'},
        {'name': 'Taxi (Chegada)', 'duration_min': 5,   'altitude_ft': 0,     'mach': 0.000, 'thrust_percentage': 7,   'roc_ft_min': 0,     'configuration': 'clean',   'burn_strategy': 'proportional'},
    ]


# ==============================================================================
# 2. FUNÇÕES AUXILIARES
# ==============================================================================

def calibrate_engine(engine_config: dict, thrust_target: float, ff_target: float) -> ENGINE_CLASS:
    """Calibra uma instância do motor e salva o ponto de projeto."""
    logger.info("\n\n" + "=" * 120)
    logger.info("FASE DE CONFIGURAÇÃO: Calibrando o motor para o ponto de projeto...")
    logger.info("\n" + "=" * 120 + "\n")

    engine_instance = ENGINE_CLASS(engine_config)
    calibration_result = engine_instance.calibrate_turbofan(
        rated_thrust_kN=thrust_target, fuel_flow_kgs=ff_target,
        t04_bounds=T04_BOUNDS_CALIBRATION, m_dot_bounds=MDOT_BOUNDS_CALIBRATION
    )
    if not calibration_result["success"]:
        raise RuntimeError("A calibração do motor falhou.")

    logger.info("Motor calibrado com sucesso!")
    logger.info(f"  - T04 Projeto: {calibration_result['optimal_t04']:.2f} K")
    logger.info(f"  - Vazão Ar Projeto (SLS): {calibration_result['optimal_mass_flow_rate']:.2f} kg/s")
    logger.info(f"  - Empuxo Projeto (Verificado): {calibration_result['final_thrust_kN']:.2f} kN")
    logger.info(f"  - TSFC Projeto (Verificado): {calibration_result['final_tsfc']:.5f} kg/(s*kN)")
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
    initial_engine_config = {'mach': 0.0, 'altitude': 0.0, 'hydrogen_fraction': 0.0}
    calibrated_engine = calibrate_engine(initial_engine_config, RATED_THRUST_TARGET_KN, FUEL_FLOW_TAKEOFF_TARGET_KGS)

    # --- Cria Modelo Aerodinâmico (para cálculo intermediário) ---
    aero_model = Aerodynamics.from_preset(AIRCRAFT_PRESET)

    # --- ETAPA 1: Simulação com Empuxo Fixo Estimado ---

    # Cria MissionManager
    mission_stage1_manager = MissionManager(engine=calibrated_engine, zero_fuel_weight=ZERO_FUEL_WEIGHT_KG,
                                            num_engines=NUM_ENGINES, design_fuel_flow_kgs=FUEL_FLOW_TAKEOFF_TARGET_KGS,
                                            design_t04_k=calibrated_engine._design_point['t04'])

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
