from typing import Literal, List, Dict, Union
from scipy.optimize import brentq
import pandas as pd

from src.systems import FuelSystem
# Importa as classes de sistema e de motor.
from src.turbofan import Turbofan
# from src.turboprop import Turboprop # Exemplo para uso futuro

# Cria uma dica de tipo (type hint) para o objeto do motor para um código mais limpo
EngineType = Union[Turbofan] # Todo: Adicionar Turboprop aqui depois: Union[Turbofan, Turboprop]


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
        if not hasattr(engine, '_design_point'):
            raise AttributeError("O objeto de motor fornecido deve ser calibrado e ter um '_design_point' salvo antes de criar uma missão.")

        self.engine = engine
        self.zero_fuel_weight = zero_fuel_weight
        self.flight_phases: List[Dict] = []
        self.results: Dict = {}
        print("MissionManager inicializado com sucesso.")

    def add_phase(
        self,
        name: str,
        duration_min: float,
        altitude_ft: float,
        mach: float,
        thrust_percentage: float,
        burn_strategy: Literal['proportional', 'hydrogen_only', 'kerosene_only'] = 'proportional'
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
        print(f"Fase '{name}' adicionada ao perfil da missão.")

    def solve_mission_fuel(
            self,
            chi_initial_mission: float = 0.0,
            tank_type: Literal['TYPE_I', 'TYPE_II', 'TYPE_III', 'TYPE_IV'] = 'TYPE_IV',
            fuel_guess_bounds: tuple = (1.0, 50000.0)  # Iniciar com 'a' > 0 para evitar divisão por zero
    ):
        """
        Encontra a massa de combustível inicial necessária para a missão e executa a simulação final.
        """
        print("\n--- Iniciando Processo de Solução de Combustível da Missão ---")

        def objective_function(initial_fuel_guess: float) -> float:
            # print(f"Testando com combustível inicial = {initial_fuel_guess:.2f} kg...") # Descomente para depuração
            sim_results = self._run_single_simulation(initial_fuel_guess, chi_initial_mission, tank_type)
            consumed_fuel = sim_results.get('total_fuel_consumed_kg', float('inf'))
            difference = initial_fuel_guess - consumed_fuel
            # print(f"  -> Consumido: {consumed_fuel:.2f} kg | Diferença: {difference:.2f} kg") # Descomente para depuração
            return difference

        # Verificação de segurança: garantir que os sinais nos limites do intervalo são opostos
        try:
            val_a = objective_function(fuel_guess_bounds[0])
            val_b = objective_function(fuel_guess_bounds[1])
            if val_a * val_b >= 0:
                print(f"ERRO: A função objetivo não muda de sinal no intervalo fornecido {fuel_guess_bounds}.")
                print(f"Valor em {fuel_guess_bounds[0]} kg: {val_a:.2f}")
                print(f"Valor em {fuel_guess_bounds[1]} kg: {val_b:.2f}")
                print("Isso geralmente significa que o consumo real está fora do intervalo de busca. Tente ajustá-lo.")
                return
        except Exception as e:
            print(f"ERRO inesperado ao testar os limites do intervalo: {e}")
            return

        try:
            print(f"Buscando solução no intervalo {fuel_guess_bounds} kg...")
            optimal_fuel_mass = brentq(
                objective_function,
                a=fuel_guess_bounds[0],
                b=fuel_guess_bounds[1],
                xtol=0.1
            )
            print(f"\nSolução encontrada! Combustível necessário para a missão: {optimal_fuel_mass:.2f} kg")
        except ValueError:
            # Este erro agora só deve ocorrer por outras razões, já que validamos os sinais
            print(f"ERRO: A solução não convergiu. Verifique se a função de consumo é contínua.")
            return

        print("\nExecutando simulação final com os valores ótimos...")
        final_results = self._run_single_simulation(optimal_fuel_mass, chi_initial_mission, tank_type)

        self.results = {
            'Combustível Inicial (kg)': optimal_fuel_mass,
            'Fração de H2 da Missão (%)': chi_initial_mission * 100,
            'Tipo de Tanque': tank_type,
            'Combustível Total Consumido (kg)': final_results['total_fuel_consumed_kg'],
            'Emissão Total de CO2 (kg)': final_results['total_co2_emitted_kg'],
            'Emissão Total de H2O (kg)': final_results['total_h2o_emitted_kg'],
            'Detalhes por Fase': pd.DataFrame(final_results['phase_details']),
            'final_fuel_system_object': final_results['final_fuel_system_state']
        }

        print("\n--- Simulação da Missão Concluída ---")
        print(f"Combustível Total Consumido: {self.results['Combustível Total Consumido (kg)']:.2f} kg")
        print(f"Emissão Total de CO2: {self.results['Emissão Total de CO2 (kg)']:.2f} kg")
        print("\nResultados detalhados por fase:")
        print(self.results['Detalhes por Fase'].round(2))

    def clear_mission(self):
        """
        Reinicia a missão limpando todas as fases de voo adicionadas.
        """
        self.flight_phases = []
        self.results = {}
        print("Perfil da missão foi limpo.")

    def _run_single_simulation(
            self,
            initial_fuel_mass: float,
            chi_initial_mission: float,
            tank_type: Literal['TYPE_I', 'TYPE_II', 'TYPE_III', 'TYPE_IV']
    ) -> dict:
        """
        Executa uma única simulação completa da missão com uma massa de combustível inicial fornecida.
        Este méthodo é robusto e não lança exceção em caso de falta de combustível; em vez disso,
        retorna um valor de consumo infinito para sinalizar a falha ao otimizador.
        """
        fuel_system = FuelSystem(initial_fuel_mass, chi_initial_mission, tank_type)

        phase_details = []
        total_fuel_consumed_kg = 0
        total_co2_emitted_kg = 0
        total_h2o_emitted_kg = 0

        try:
            for phase in self.flight_phases:
                if phase['burn_strategy'] == 'hydrogen_only':
                    engine_chi = 1.0
                elif phase['burn_strategy'] == 'kerosene_only':
                    engine_chi = 0.0
                else:
                    engine_chi = chi_initial_mission

                self.engine.update_final_config({'hydrogen_fraction': engine_chi})

                self.engine.update_environment(
                    mach=phase['mach'],
                    altitude=phase['altitude_ft'],
                    percentage_of_rated_thrust=phase['thrust_percentage'] * 100
                )

                duration_sec = phase['duration_min'] * 60
                fuel_flow_kgs = self.engine.get_fuel_consumption()
                emissions_flow = self.engine.get_emissions_flow()

                fuel_consumed_phase = fuel_flow_kgs * duration_sec
                co2_emitted_phase = emissions_flow['co2_flow_kgs'] * duration_sec
                h2o_emitted_phase = emissions_flow['h2o_flow_kgs'] * duration_sec

                fuel_system.consume_fuel(fuel_consumed_phase, phase['burn_strategy'])

                phase_details.append({
                    'Fase': phase['name'], 'Duração (min)': phase['duration_min'],
                    'Combustível Consumido (kg)': fuel_consumed_phase,
                    'Emissão de CO2 (kg)': co2_emitted_phase, 'Emissão de H2O (kg)': h2o_emitted_phase,
                    'TSFC (kg/s/kN)': self.engine.get_tsfc(),
                    'N2 (%)': getattr(self.engine, 'N2_ratio', None) * 100 if hasattr(self.engine,
                                                                                      'N2_ratio') else None,
                })

                total_fuel_consumed_kg += fuel_consumed_phase
                total_co2_emitted_kg += co2_emitted_phase
                total_h2o_emitted_kg += h2o_emitted_phase

        except ValueError as e:
            # Se um ValueError ocorrer (falta de combustível), sinalizamos a falha
            # retornando um consumo total infinito.
            # print(f"  -> Falha na simulação: {e}") # Todo: Descomente para depuração
            return {"total_fuel_consumed_kg": float('inf')}

        return {
            "total_fuel_consumed_kg": total_fuel_consumed_kg,
            "total_co2_emitted_kg": total_co2_emitted_kg,
            "total_h2o_emitted_kg": total_h2o_emitted_kg,
            "phase_details": phase_details,
            "final_fuel_system_state": fuel_system
        }