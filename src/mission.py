import logging
import os
from typing import Literal, List, Dict, Union

import pandas as pd
from scipy.optimize import brentq

from src.systems import FuelSystem
from src.turbofan import Turbofan
from utils.aux_tools import (
    calculate_energy_from_fuel,
    calculate_fuel_consumption_breakdown,
    min2s
)

# from src.turboprop import Turboprop # Exemplo para uso futuro

# --- Configuração do Logger ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Obtém o logger para este módulo
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Evita adicionar múltiplos handlers se o módulo for recarregado
if not logger.handlers:
    # Cria um file handler que loga mensagens no arquivo
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "mission.log"), mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)

    # Cria um console handler para exibir logs no console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Define o formato do log
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Adiciona os handlers ao logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Cria uma dica de tipo (type hint) para o objeto do motor para um código mais limpo
EngineType = Union[
    Turbofan
]  # Todo: Adicionar Turboprop aqui depois: Union[Turbofan, Turboprop]


class MissionManager:
    """
    Orquestra a simulação de uma missão de voo completa, fase a fase.

    Esta classe é responsável por gerenciar a sequência de fases de voo,
    calcular o combustível total necessário através de um processo iterativo e
    agregar os resultados finais, como consumo total e emissões.
    """

    def __init__(self, engine: EngineType, zero_fuel_weight: float):
        """
        Inicializa o MissionManager.

        Args:
            engine: Uma instância de uma classe de motor compatível (ex: Turbofan)
                    que tenha sido calibrada e tenha seu ponto de projeto salvo.
            zero_fuel_weight (float): O Peso Zero Combustível (ZFW) da aeronave,
                                      representando o peso da aeronave sem combustível [kg].
        """
        if not hasattr(engine, "_design_point"):
            raise AttributeError(
                "O objeto de motor fornecido deve ser calibrado e ter um '_design_point' salvo antes de criar uma missão."
            )

        self.engine = engine
        self.zero_fuel_weight = zero_fuel_weight
        self.flight_phases: List[Dict] = []
        self.results: Dict = {}
        logger.info("MissionManager inicializado com sucesso.")

    def add_phase(
            self,
            name: str,
            duration_min: float,
            altitude_ft: float,
            mach: float,
            thrust_percentage: float,
            burn_strategy: Literal[
                "proportional", "hydrogen_only", "kerosene_only"
            ] = "proportional",
    ):
        """
        Adiciona uma única fase de voo ao perfil da missão.

        Args:
            name (str): Um nome descritivo para a fase (ex: "Subida 1", "Cruzeiro").
            duration_min (float): A duração da fase em minutos.
            altitude_ft (float): A altitude operacional para a fase em pés.
            mach (float): O número de Mach operacional para a fase.
            thrust_percentage (float): O empuxo necessário como uma porcentagem do empuxo
                                       de projeto do motor (ex: 85 para 85%).
            burn_strategy (str): A estratégia de consumo de combustível durante esta fase.
                                 - 'proportional': Consome H2 e querosene com base na mistura inicial da missão.
                                 - 'hydrogen_only': Consome apenas hidrogênio.
                                 - 'kerosene_only': Consome apenas querosene.
        """
        if not 0 <= thrust_percentage <= 100:
            raise ValueError("O 'thrust_percentage' deve estar entre 0 e 100.")

        phase_data = {
            "name": name,
            "duration_min": duration_min,
            "altitude_ft": altitude_ft,
            "mach": mach,
            "thrust_percentage": thrust_percentage / 100.0,  # Armazena como uma fração (0.0 a 1.0)
            "burn_strategy": burn_strategy,
        }
        self.flight_phases.append(phase_data)
        logger.info(f"Fase '{name}' adicionada ao perfil da missão.")

    def solve_mission_fuel(
            self,
            chi_initial_mission: float = 0.0,
            tank_type: Literal["TYPE_I", "TYPE_II", "TYPE_III", "TYPE_IV"] = "TYPE_IV",
            fuel_guess_bounds: tuple = (
                    1.0,
                    50000.0,
            ),  # Iniciar com 'a' > 0 para evitar divisão por zero
    ):
        """
        Encontra a massa de combustível inicial necessária para a missão e executa a simulação final.
        """
        logger.info("--- Iniciando Processo de Solução de Combustível da Missão ---")

        def objective_function(initial_fuel_guess: float) -> float:
            # logger.debug(f"Testando com combustível inicial = {initial_fuel_guess:.2f} kg...") # Descomente para depuração
            sim_results = self._run_single_simulation(
                initial_fuel_guess, chi_initial_mission, tank_type
            )
            consumed_fuel = sim_results.get("total_fuel_consumed_kg", float("inf"))
            difference = initial_fuel_guess - consumed_fuel
            # logger.debug(f"  -> Consumido: {consumed_fuel:.2f} kg | Diferença: {difference:.2f} kg") # Descomente para depuração
            return difference

        # Verificação de segurança: garantir que os sinais nos limites do intervalo são opostos
        try:
            val_a = objective_function(fuel_guess_bounds[0])
            val_b = objective_function(fuel_guess_bounds[1])
            if val_a * val_b >= 0:
                logger.error(
                    f"A função objetivo não muda de sinal no intervalo fornecido {fuel_guess_bounds}."
                )
                logger.error(f"Valor em {fuel_guess_bounds[0]} kg: {val_a:.2f}")
                logger.error(f"Valor em {fuel_guess_bounds[1]} kg: {val_b:.2f}")
                logger.error(
                    "Isso geralmente significa que o consumo real está fora do intervalo de busca. Tente ajustá-lo."
                )
                return
        except Exception as e:
            logger.error(f"ERRO inesperado ao testar os limites do intervalo: {e}")
            return

        try:
            logger.info(f"Buscando solução no intervalo {fuel_guess_bounds} kg...")
            optimal_fuel_mass = brentq(
                objective_function,
                a=fuel_guess_bounds[0],
                b=fuel_guess_bounds[1],
                xtol=0.1,
            )
            logger.info(
                f"Solução encontrada! Combustível necessário para a missão: {optimal_fuel_mass:.2f} kg"
            )
        except ValueError:
            # Este erro agora só deve ocorrer por outras razões, já que validamos os sinais
            logger.error(
                "A solução não convergiu. Verifique se a função de consumo é contínua."
            )
            return

        logger.info("Executando simulação final com os valores ótimos...")
        try:
            final_results = self._run_single_simulation(
                optimal_fuel_mass, chi_initial_mission, tank_type
            )
            # Verifica se o empuxo necessário foi atingido em cada fase
            for phase_detail in final_results.get("phase_details", []):
                if abs(phase_detail.get("Empuxo Obtido (kN)", 0) - phase_detail.get("Empuxo Requerido (kN)",
                                                                                    float("inf"))) > 0.01:
                    raise ValueError(f"Empuxo necessário não atingido na fase {phase_detail['Fase']}")
        except ValueError as e:
            logger.error(f"Falha na simulação final: {str(e)}")
            return
        # --- GERAÇÃO DOS DATAFRAMES E ARQUIVOS CSV ---
        full_df = pd.DataFrame(final_results['phase_details'])

        # Define as colunas para o relatório resumido
        summary_cols = [
            'Fase', 'Duração (min)', 'Empuxo Requerido (kN)', 'Empuxo Obtido (kN)',
            'Combustível Total (kg)', 'Emissão CO2 (kg)', 'Emissão H2O (kg)',
            'TSFC (kg/s/kN)', 'N2 (%)'
        ]
        summary_df = full_df[summary_cols]

        # Salva os arquivos CSV, aplicando o arredondamento na saída
        summary_df.to_csv('mission_summary.csv', index=False, float_format='%.3f')
        full_df.to_csv('mission_detailed.csv', index=False, float_format='%.4f')
        logger.info("Relatórios 'mission_summary.csv' e 'mission_detailed.csv' foram salvos.")

        self.results = {
            'Combustível Inicial (kg)': optimal_fuel_mass,
            'Fração de H2 da Missão (%)': chi_initial_mission * 100,
            'Tipo de Tanque': tank_type,
            'Combustível Total Consumido (kg)': final_results['total_fuel_consumed_kg'],
            'Emissão Total de CO2 (kg)': final_results['total_co2_emitted_kg'],
            'Emissão Total de H2O (kg)': final_results['total_h2o_emitted_kg'],
            'summary_df': summary_df,
            'detailed_df': full_df,
            'final_fuel_system_object': final_results['final_fuel_system_state']
        }

        logger.info("--- Simulação da Missão Concluída ---")
        logger.info(
            f"Combustível Total Consumido: {self.results['Combustível Total Consumido (kg)']:.2f} kg"
        )
        logger.info(
            f"Emissão Total de CO2: {self.results['Emissão Total de CO2 (kg)']:.2f} kg"
        )
        logger.info("Resultados detalhados por fase:")
        # Log do DataFrame convertido para string para uma melhor formatação no arquivo de log
        logger.info(f"\n{summary_df.round(3).to_string()}")

    def clear_mission(self):
        """
        Reinicia a missão limpando todas as fases de voo adicionadas.
        """
        self.flight_phases = []
        self.results = {}
        logger.info("Perfil da missão foi limpo.")

    def _run_single_simulation(
            self,
            initial_fuel_mass: float,
            chi_initial_mission: float,
            tank_type: Literal["TYPE_I", "TYPE_II", "TYPE_III", "TYPE_IV"],
    ) -> dict:
        """
        Executa uma única simulação completa da missão com uma massa de combustível inicial fornecida.
        Este méhtodo é robusto e não lança exceção em caso de falta de combustível; em vez disso,
        retorna um valor de consumo infinito para sinalizar a falha ao otimizador.
        """
        fuel_system = FuelSystem(initial_fuel_mass, chi_initial_mission, tank_type)
        phase_details = []
        totals = {'fuel': 0.0, 'co2': 0.0, 'h2o': 0.0}

        try:
            for phase in self.flight_phases:
                if phase["burn_strategy"] == "hydrogen_only":
                    engine_chi = 1.0
                elif phase["burn_strategy"] == "kerosene_only":
                    engine_chi = 0.0
                else:
                    engine_chi = chi_initial_mission

                self.engine.update_final_config({"hydrogen_fraction": engine_chi})
                self.engine.update_environment(
                    mach=phase["mach"],
                    altitude=phase["altitude_ft"],
                    percentage_of_rated_thrust=phase["thrust_percentage"],
                )

                thrust_required = self.engine._design_point["rated_thrust"] * phase["thrust_percentage"]
                thrust_obtained = self.engine.get_thrust()
                duration_sec = phase['duration_min'] * min2s

                # Consumo e emissões na fase
                fuel_consumed_phase = self.engine.get_fuel_consumption() * duration_sec
                emissions_flow = self.engine.get_emissions_flow()
                co2_emitted_phase = emissions_flow['co2_flow_kgs'] * duration_sec
                h2o_emitted_phase = emissions_flow['h2o_flow_kgs'] * duration_sec

                fuel_system.consume_fuel(fuel_consumed_phase, phase['burn_strategy'])

                # Decomposição de consumo e energia
                fuel_breakdown = calculate_fuel_consumption_breakdown(fuel_consumed_phase, chi_initial_mission,
                                                                      phase['burn_strategy'])
                energy_breakdown = calculate_energy_from_fuel(
                    fuel_consumed_phase, chi_initial_mission, phase['burn_strategy'],
                    self.engine.kerosene_PCI, self.engine.hydrogen_PCI
                )

                # Coleta de todos os dados para a fase
                phase_data = {
                    'Fase': phase['name'],
                    'Duração (min)': phase['duration_min'],
                    'Altitude (ft)': phase['altitude_ft'],
                    'Mach': phase['mach'],
                    'Estratégia de Queima': phase['burn_strategy'],
                    'Empuxo Requerido (kN)': thrust_required,
                    'Empuxo Obtido (kN)': thrust_obtained,
                    'Erro de Empuxo (%)': (
                                                      thrust_obtained - thrust_required) / thrust_required * 100 if thrust_required > 0 else 0.0,
                    'Combustível Total (kg)': fuel_consumed_phase,
                    'H2 Consumido (kg)': fuel_breakdown['h2_consumed_kg'],
                    'Querosene Consumido (kg)': fuel_breakdown['qav_consumed_kg'],
                    'Energia H2 (MJ)': energy_breakdown['energy_h2_kJ'] / 1000,
                    'Energia Querosene (MJ)': energy_breakdown['energy_qav_kJ'] / 1000,
                    'Emissão CO2 (kg)': co2_emitted_phase,
                    'Emissão H2O (kg)': h2o_emitted_phase,
                    'TSFC (kg/s/kN)': self.engine.get_tsfc(),
                    'N2 (%)': getattr(self.engine, 'N2_ratio', 1.0) * 100,
                    'N1 (%)': getattr(self.engine, 'N1_ratio', 1.0) * 100,
                    'f (razão comb/ar)': self.engine.fuel_to_air_ratio,
                    'chi (fração H2 no motor)': engine_chi,
                    'Vazão de Ar (kg/s)': self.engine.get_air_flow(),
                    'T04 (K)': self.engine.t04,
                    'BPR': self.engine.bpr if hasattr(self.engine, 'bpr') else None,
                    'Prf': self.engine.prf if hasattr(self.engine, 'prf') else None,
                    'Prc': self.engine.prc,
                }
                phase_details.append(phase_data)

                totals['fuel'] += fuel_consumed_phase
                totals['co2'] += co2_emitted_phase
                totals['h2o'] += h2o_emitted_phase

        except ValueError as e:
            # Se um ValueError ocorrer (falta de combustível), sinalizamos a falha
            # retornando um consumo total infinito.
            # logger.debug(f"  -> Falha na simulação: {e}")     # Todo: Descomente para depuração
            return {"total_fuel_consumed_kg": float("inf")}

        return {
            "total_fuel_consumed_kg": totals['fuel'],
            "total_co2_emitted_kg": totals['co2'],
            "total_h2o_emitted_kg": totals['h2o'],
            "phase_details": phase_details,
            "final_fuel_system_state": fuel_system
        }
