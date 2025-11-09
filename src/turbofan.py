import logging
import os
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
from numpy import ndarray
from scipy.optimize import minimize_scalar

from src.components.combustion_chamber import CombustionChamber
from src.components.compressor import Compressor
from src.components.inlet import Inlet
from src.components.nozzle import Nozzle
from src.components.turbine import Turbine
from utils.aux_tools import (
    atmosphere,
    ft2m,
    CO2_PER_KEROSENE_MASS,
    H2O_PER_KEROSENE_MASS,
    H2O_PER_HYDROGEN_MASS,
    SEA_LEVEL_PRESSURE,
    SEA_LEVEL_TEMPERATURE,
)
from utils.configs import DEFAULT_CONFIG_TURBOFAN
from utils.corrections import model_corrections

# --- Configuração do Logger ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Obtém o logger para este módulo
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Evita adicionar múltiplos handlers se o módulo for recarregado (útil em notebooks)
if not logger.handlers:
    # Cria um file handler que loga mensagens no arquivo
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, "turbofan.log"), mode="w", encoding="utf-8"
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


class Turbofan:

    def __init__(self, config_dict):
        """
        Inicializa o motor turbofan com base em um dicionário de configuração.

        Usa os valores padrão da classe para quaisquer chaves ausentes no
        dicionário fornecido.
        """
        # Cria a configuração final mesclando os padrões com os fornecidos
        self.final_config = DEFAULT_CONFIG_TURBOFAN.copy()
        self.final_config.update(config_dict)

        # --- Dados do ambiente ---
        self.mach = self.final_config["mach"]
        self.altitude = self.final_config["altitude"]
        self.delta_isa_temperature = self.final_config["delta_isa_temperature"]
        t_a_altitude, p_a_altitude, _, _ = atmosphere(
            self.altitude * ft2m, Tba=SEA_LEVEL_TEMPERATURE + self.delta_isa_temperature
        )
        self.t_a = self.final_config.get("t_a", t_a_altitude) or t_a_altitude
        self.p_a = (
                self.final_config.get("p_a", p_a_altitude / 1000) or p_a_altitude / 1000
        )  # Divide por 1000 para passar para kPa

        # --- Eficiências e Gammas ---
        self.eta_inlet = self.final_config["eta_inlet"]
        self.gamma_inlet = self.final_config["gamma_inlet"]
        self.eta_fan = self.final_config["eta_fan"]
        self.gamma_fan = self.final_config["gamma_fan"]
        self.eta_compressor = self.final_config["eta_compressor"]
        self.gamma_compressor = self.final_config["gamma_compressor"]
        self.eta_camara = self.final_config["eta_camara"]
        self.gamma_camara = self.final_config["gamma_camara"]
        self.eta_turbina_compressor = self.final_config["eta_turbina_compressor"]
        self.gamma_turbina_compressor = self.final_config["gamma_turbina_compressor"]
        self.eta_turbina_fan = self.final_config["eta_turbina_fan"]
        self.gamma_turbina_fan = self.final_config["gamma_turbina_fan"]
        self.eta_bocal_quente = self.final_config["eta_bocal_quente"]
        self.gamma_bocal_quente = self.final_config["gamma_bocal_quente"]
        self.eta_bocal_fan = self.final_config["eta_bocal_fan"]
        self.gamma_bocal_fan = self.final_config["gamma_bocal_fan"]

        # --- Dados operacionais ---
        self.bpr = self.final_config["bpr"]
        self.prf = self.final_config["prf"]
        self.pr_bst = self.final_config["pr_bst"]
        self.prc = self.final_config["prc"]
        self.hydrogen_fraction = self.final_config["hydrogen_fraction"]
        self.pressure_loss = self.final_config["pressure_loss"]
        self.kerosene_PCI = self.final_config["kerosene_PCI"]
        self.hydrogen_PCI = self.final_config["hydrogen_PCI"]
        self.mean_R_air = self.final_config["mean_R_air"]
        self.Cp = self.final_config["Cp"]
        self.t04_without_loss = self.final_config["T04"]
        self.t04 = self.final_config["T04"]
        self.sea_level_air_flow = None
        self.air_flow = None

        # Adiciona um atributo para armazenar os modelos de correção
        self._correction_models = {}
        self._initialize_correction_models()

    def _initialize_correction_models(self):
        """
        Inicializa os modelos polinomiais de correção adimensional.
        Cada modelo retorna a razão entre o valor em um dado N2 e o valor no ponto de projeto.
        """
        self._correction_models = model_corrections(is_turbofan=True)

    def update_final_config(self, config_dict: dict):
        """
        Atualiza a configuração do motor com novos valores fornecidos em um dicionário.
        Quaisquer chaves ausentes manterão seus valores atuais.
        """
        self.final_config.update(config_dict)

        # Reaplica os valores atualizados
        self.__init__(self.final_config)
        if hasattr(self, "_design_point"):
            self.set_sea_level_air_flow(self._design_point["sea_level_air_flow"])

    def save_design_point(self):
        """
        Salva os valores dos parâmetros no ponto de projeto após a calibração.
        """
        logger.debug("Salvando ponto de projeto do turbofan...")
        self._design_point = {
            "bpr": self.bpr,
            "prf": self.prf,
            "pr_bst": self.pr_bst,
            "prc": self.prc,
            "t04": self.t04_without_loss,
            "eta_fan": self.eta_fan,
            "eta_compressor": self.eta_compressor,
            "eta_camara": self.eta_camara,
            "eta_turbina_fan": self.eta_turbina_fan,
            "eta_turbina_compressor": self.eta_turbina_compressor,
            "eta_bocal_quente": self.eta_bocal_quente,
            "sea_level_air_flow": self.sea_level_air_flow,
            "rated_thrust": self.get_thrust(),
        }
        logger.debug(
            f"Ponto de projeto salvo com empuxo de {self._design_point['rated_thrust']:.2f} kN."
        )

    def update_from_N2(self, N2: float, N2_design: float = 1.0):
        """
        Atualiza os parâmetros operacionais do turbofan com base na rotação normalizada do eixo de alta (N2).

        Args:
            N2 (float): Rotação atual do eixo de alta pressão (normalizada)
            N2_design (float): Rotação de projeto do eixo de alta pressão (normalizada). Default = 1.0
        """
        if not hasattr(self, "_design_point"):
            raise ValueError(
                "Ponto de projeto não definido. Execute save_design_point() após a calibração."
            )

        models = self._correction_models
        N2_ratio = N2 / N2_design

        # Se estiver no ponto de projeto, não faz nada (todos coeficientes são 1)
        if abs(N2_ratio - 1.0) <= 1e-4:
            self.N2_ratio = 1.0
            self.N1_ratio = 1.0
            self.bpr = self._design_point["bpr"]
            self.prf = self._design_point["prf"]
            if self._design_point["pr_bst"]:
                self.pr_bst = self._design_point["pr_bst"]
            self.prc = self._design_point["prc"]
            self.t04 = self._design_point["t04"]
            self.eta_fan = self._design_point["eta_fan"]
            self.eta_compressor = self._design_point["eta_compressor"]
            self.eta_turbina_fan = self._design_point["eta_turbina_fan"]
            self.eta_turbina_compressor = self._design_point["eta_turbina_compressor"]
            self.eta_camara = self._design_point["eta_camara"]
            self.set_sea_level_air_flow(self._design_point["sea_level_air_flow"])
        else:
            N1_ratio = models["N1_from_N2"](N2_ratio)
            self.N1_ratio = N1_ratio
            self.N2_ratio = N2_ratio

            B_ratio = models["B_from_N1"](N1_ratio)
            self.bpr = self._design_point["bpr"] * B_ratio

            A = models["A_from_B_design"](self._design_point["bpr"])
            C = models["C_from_B_design"](self._design_point["bpr"])
            p_prf = np.poly1d([A, -4.3317e-2, C])
            self.prf = self._design_point["prf"] * p_prf(N1_ratio)

            if self._design_point["pr_bst"]:
                self.pr_bst = self._design_point["pr_bst"] * models["Pr_bst_from_N1"](
                    N1_ratio
                )
            self.prc = self._design_point["prc"] * models["Prc_from_N2"](N2_ratio)
            self.t04 = self._design_point["t04"] * models["T04_from_N2"](N2_ratio)

            self.eta_fan = self._design_point["eta_fan"] * models["eta_f_from_N1"](
                N1_ratio
            )
            self.eta_compressor = self._design_point["eta_compressor"] * models[
                "eta_c_from_N2"
            ](N2_ratio)
            self.eta_turbina_fan = self._design_point["eta_turbina_fan"] * models[
                "eta_tf_from_N1"
            ](N1_ratio)
            self.eta_turbina_compressor = self._design_point[
                                              "eta_turbina_compressor"
                                          ] * models["eta_t_from_N2"](N2_ratio)
            self.eta_camara = self._design_point["eta_camara"] * models[
                "eta_b_from_N2"
            ](N2_ratio)

            hot_air_flow_ratio = models["m_dot_H_from_N2"](N2_ratio)
            hot_air_flow = self._design_point["sea_level_air_flow"] * hot_air_flow_ratio
            self.set_sea_level_air_flow(hot_air_flow)

    def get_values_by_changing_param(
            self,
            value_name: str = "tsfc",
            param: Literal["N1", "N2"] = "N2",
            n2_range: np.ndarray = np.arange(0.5, 1.05, 0.1),
    ) -> pd.DataFrame:
        """
        Gera um pd.DataFrame variando um parâmetro (N1 ou N2) e retornando os valores de uma variável específica.

        Args:
            param (str): Parâmetro a ser variado ("N1" ou "N2").
            value_name (str): Nome da variável cujo valor será retornado.
        Returns:
            pd.DataFrame: DataFrame contendo os valores do parâmetro e da variável especificada.
        """
        if param not in ["N1", "N2"]:
            raise ValueError("Parâmetro inválido. Use 'N1' ou 'N2'.")
        if not hasattr(self, "_design_point"):
            raise ValueError(
                "Ponto de projeto não definido. Execute save_design_point() após a calibração."
            )

        y_values, n1_range = [], []
        for N2 in n2_range:
            self.update_from_N2(N2)
            n1_range.append(self.N1_ratio)
            if value_name == "tsfc":
                y_values.append(self.get_tsfc())
            elif value_name == "N1":
                y_values.append(self.N1_ratio)
            elif value_name == "thrust":
                y_values.append(self.get_thrust())
            elif value_name == "prc":
                y_values.append(self.prc)
            elif value_name == "pr":
                y_values.append(self.prc * self.prf)
            elif value_name == "u_core":
                y_values.append(self.u_core)
            elif value_name == "hot_air_flow":
                y_values.append(self.get_hot_air_flow())
            elif value_name == "t04":
                y_values.append(self.t04)
            elif value_name == "fuel_consumption":
                y_values.append(self.get_fuel_consumption())
            else:
                try:
                    y_values.append(getattr(self, value_name))
                except AttributeError:
                    raise ValueError(f"Nome de variável inválido: {value_name}")

        x_values = n2_range if param == "N2" else n1_range

        return pd.DataFrame({param: x_values, value_name: y_values}).round(4)

    def set_t04(self, t04: float):
        """
        Temperatura de entrada da turbina do compressor (saída da câmara de combustão): é um gargalo tecnológico.
        """
        self.t04 = t04
        self.update_turbofan_components()

    def set_delta_temperature(self, delta_temperature: float):
        """
        Define a variação de temperatura da atmosfera ISA e atualiza as condições do motor.
        Args:
            delta_temperature (float): Variação de temperatura em Kelvin.
        """
        self.update_final_config({"delta_temperature": delta_temperature})

    def update_turbofan_components(self):
        """
        Atualiza os componentes do turbofan com base nos parâmetros atuais.

        Esta função instancia e atualiza todos os componentes principais do motor:
        difusor (inlet), fan, compressor, câmara de combustão, turbinas, bocais e calcula
        as propriedades relevantes em cada estação do ciclo termodinâmico.
        """

        # 1. Difusor
        self.inlet = Inlet(
            self.t_a, self.p_a, self.mach, self.eta_inlet, self.gamma_inlet
        )
        self.t02, self.p02 = (
            self.inlet.get_total_temperature(),
            self.inlet.get_total_pressure(),
        )

        # 2. Fan
        self.fan = Compressor(
            self.t02, self.p02, self.prf, self.eta_fan, self.gamma_fan
        )
        self.t08, self.p08 = (
            self.fan.get_total_temperature(),
            self.fan.get_total_pressure(),
        )

        # 3. Compressor
        self.compressor = Compressor(
            self.t08, self.p08, self.prc, self.eta_compressor, self.gamma_compressor
        )
        self.t03, self.p03 = (
            self.compressor.get_total_temperature(),
            self.compressor.get_total_pressure(),
        )
        if self.pr_bst:
            self.p03 *= self.pr_bst

        # 4. Câmara de Combustão
        self.combustion_chamber = CombustionChamber(
            self.t03,
            self.p03,
            self.Cp,
            self.t04,
            self.eta_camara,
            self.gamma_camara,
            self.kerosene_PCI,
            self.hydrogen_PCI,
            self.hydrogen_fraction,
            self.pressure_loss,
        )
        self.p04 = self.combustion_chamber.get_total_pressure()
        # Calcula novamente t04 para levar em conta possível perda de pressão
        self.t04 = self.combustion_chamber.get_total_temperature_out()
        self.fuel_to_air_ratio = self.combustion_chamber.get_fuel_to_air_ratio()

        # 5. Turbina do compressor
        self.compressor_turbine = Turbine(
            self.t04,
            self.p04,
            self.t08,
            self.t03,
            self.eta_turbina_compressor,
            self.gamma_turbina_compressor,
        )
        self.t05, self.p05 = (
            self.compressor_turbine.get_total_temperature(),
            self.compressor_turbine.get_total_pressure(),
        )

        # 6. Turbina do fan
        bpr_design = (
            self._design_point["bpr"] if hasattr(self, "_design_point") else self.bpr
        )
        self.fan_turbine = Turbine(
            self.t05,
            self.p05,
            self.t02,
            self.t08,
            self.eta_turbina_fan,
            self.gamma_turbina_fan,
            bpr_design,
        )
        self.t06, self.p06 = (
            self.fan_turbine.get_total_temperature(),
            self.fan_turbine.get_total_pressure(),
        )

        # 7. Bocal dos gases quentes
        self.core_nozzle = Nozzle(
            self.t06,
            self.p06,
            self.p_a,
            self.eta_bocal_quente,
            self.gamma_bocal_quente,
            self.mean_R_air,
        )
        self.u_core = self.core_nozzle.get_exhaust_velocity()

        # 8. Bocal do fan
        self.fan_nozzle = Nozzle(
            self.t08,
            self.p08,
            self.p_a,
            self.eta_bocal_fan,
            self.gamma_bocal_fan,
            self.mean_R_air,
        )
        self.u_fan = self.fan_nozzle.get_exhaust_velocity()

        # 9. Velocidade de voo
        self.u_flight = self.get_flight_speed()

    def get_flight_speed(self):
        """
        Calcula a velocidade de voo baseada no Mach e nas propriedades do ar.
        """
        return self.mach * np.sqrt(self.gamma_inlet * self.mean_R_air * self.t_a)

    def get_specific_thrust(self):
        """
        Calcula o empuxo específico do motor turbofan.

        O empuxo específico é a força de empuxo gerada por unidade de vazão mássica de ar quente.
        Considera a contribuição do núcleo (core) e do fan, descontando a velocidade de voo.

        Returns:
            float: Empuxo específico em kN/(kg/s).
        """
        term1 = (1 + self.fuel_to_air_ratio) * self.u_core - self.u_flight
        term2 = self.bpr * (self.u_fan - self.u_flight)
        return (term1 + term2) / 1000  # converte para kN/(kg/s)

    def get_tsfc(self):
        specific_thrust = self.get_specific_thrust()
        if specific_thrust <= 0:
            return float("inf")
        return self.fuel_to_air_ratio / specific_thrust

    def set_sea_level_air_flow(self, sea_level_air_flow: float):
        """
        Define a vazão de ar ao nível do mar e atualiza a vazão corrigida para as condições atuais.

        Args:
            sea_level_air_flow (float): Vazão de ar ao nível do mar (kg/s).
        """
        self.sea_level_air_flow = sea_level_air_flow
        correction_factor = (SEA_LEVEL_TEMPERATURE / SEA_LEVEL_PRESSURE) * (
                self.p_a / self.t_a
        )
        self.air_flow = sea_level_air_flow * correction_factor
        self.update_turbofan_components()

    def get_sea_level_air_flow(self):
        """
        Retorna a vazão de ar ao nível do mar.

        Returns:
            float: Vazão de ar ao nível do mar (kg/s).

        Raises:
            ValueError: Se a vazão de ar não estiver definida.
        """
        if self.sea_level_air_flow is None:
            raise ValueError("Vazão de ar não definida.")
        return self.sea_level_air_flow

    def set_air_flow(self, air_flow: float):
        """
        Define a vazão de ar corrigida para as condições atuais e atualiza a vazão ao nível do mar.

        Args:
            air_flow (float): Vazão de ar corrigida (kg/s).
        """
        self.air_flow = air_flow
        correction_factor = (SEA_LEVEL_TEMPERATURE / SEA_LEVEL_PRESSURE) * (
                self.p_a / self.t_a
        )
        self.sea_level_air_flow = air_flow / correction_factor
        self.update_turbofan_components()

    def get_air_flow(self):
        """
        Retorna a vazão de ar corrigida para as condições atuais.

        Returns:
            float: Vazão de ar corrigida (kg/s).

        Raises:
            ValueError: Se a vazão de ar não estiver definida.
        """
        if self.air_flow is None:
            raise ValueError("Vazão de ar não definida.")
        return self.air_flow

    def get_hot_air_flow(self):
        """
        Retorna a vazão de ar quente (núcleo do motor).

        Returns:
            float: Vazão de ar quente (kg/s).
        """
        return self.get_air_flow() / (self.bpr + 1)

    def get_thrust(self):
        """
        Calcula o empuxo total do motor turbofan.

        Returns:
            float: Empuxo total (kN).
        """
        return self.get_specific_thrust() * self.get_hot_air_flow()

    def get_fuel_consumption(self):
        """
        Calcula o consumo total de combustível do motor turbofan.

        Returns:
            float: Consumo de combustível (kg/s).
        """
        return self.fuel_to_air_ratio * self.get_hot_air_flow()

    def get_emissions_flow(self) -> dict:
        """
        Calcula as vazões mássicas de emissões de CO2 e H2O.
        Lógica baseada na estequiometria do combustível.
        """
        total_fuel_flow = self.get_fuel_consumption()
        chi = self.hydrogen_fraction
        kerosene_flow = total_fuel_flow * (1 - chi)
        hydrogen_flow = total_fuel_flow * chi

        # Calcula a vazão de CO2 (gerado apenas pelo querosene)
        co2_flow = kerosene_flow * CO2_PER_KEROSENE_MASS

        # Calcula a vazão de H2O (gerado por ambos os combustíveis)
        h2o_from_kerosene = kerosene_flow * H2O_PER_KEROSENE_MASS
        h2o_from_hydrogen = hydrogen_flow * H2O_PER_HYDROGEN_MASS
        total_h2o_flow = h2o_from_kerosene + h2o_from_hydrogen

        return {"co2_flow_kgs": co2_flow, "h2o_flow_kgs": total_h2o_flow}

    # Print dos Outputs
    def print_outputs(self):
        """
        Imprime os resultados calculados do motor de forma organizada.
        Verifica se os componentes foram atualizados antes de imprimir.
        """
        if not hasattr(self, "inlet"):
            print(
                "\nAVISO: Os resultados não foram calculados. Execute 'update_turbofan_components()' primeiro."
            )
            return

        print("--- Resultados da Simulação do Motor ---")

        print("\n[ Estações do Motor ]")
        header = (
            f"{'Estação':<15} | {'Temp. Total (K)':<20} | {'Pressão Total (kPa)':<20}"
        )
        print(header)
        print("-" * len(header))
        print(f"{'2 (Inlet)':<15} | {self.t02:<20.3f} | {self.p02:<20.3f}")
        print(f"{'8 (Fan)':<15} | {self.t08:<20.3f} | {self.p08:<20.3f}")
        print(f"{'3 (Compressor)':<15} | {self.t03:<20.3f} | {self.p03:<20.3f}")
        print(f"{'4 (Câmara)':<15} | {self.t04:<20.3f} | {self.p04:<20.3f}")
        print(f"{'5 (Turbina Comp.)':<15} | {self.t05:<20.3f} | {self.p05:<20.3f}")
        print(f"{'6 (Turbina Fan)':<15} | {self.t06:<20.3f} | {self.p06:<20.3f}")

        print("\n[ Velocidades de Saída ]")
        print(f"{'Velocidade de Voo (u_0)':<32}: {self.u_flight:.3f} m/s")
        print(f"{'Velocidade Bocal Quente (u_core)':<32}: {self.u_core:.3f} m/s")
        print(f"{'Velocidade Bocal Frio (u_fan)':<32}: {self.u_fan:.3f} m/s")

        print("\n[ Performance Geral ]")
        print(f"{'Razão Combustível/Ar (f)':<32}: {self.fuel_to_air_ratio:.5f}")
        print(f"{'Empuxo Específico':<32}: {self.get_specific_thrust():.5f} kN.s/kg")
        print(f"{'Consumo Específico (TSFC)':<32}: {self.get_tsfc():.5f} kg/(kN.s)")

        if hasattr(self, "N2_ratio"):
            print(f"{'N2':<35}: {self.N2_ratio:.3f}")
        if hasattr(self, "N1_ratio"):
            print(f"{'N1':<35}: {self.N1_ratio:.3f}")

        if self.air_flow is not None:
            print(f"{'Empuxo Total':<32}: {self.get_thrust():.3f} kN")
            print(
                f"{'Consumo de Combustível':<32}: {self.get_fuel_consumption():.3f} kg/s"
            )

        print("\n" + "-" * 61)

    # Calibração do Turbofan
    def calibrate_turbofan(
            self,
            rated_thrust_kN: float,
            fuel_flow_kgs: float,
            t04_bounds: tuple = (1000, 3000),
            m_dot_bounds: tuple = (1, 2000),
    ) -> dict:
        """
        Calibra T04 primeiro para atingir o TSFC alvo (mantendo vazão fixa),
        e depois calibra a vazão mássica para atingir o empuxo alvo (T04 fixa).
        """
        if rated_thrust_kN <= 0:
            raise ValueError("O empuxo nominal (rated_thrust_kN) deve ser positivo.")

        # TSFC alvo (kg/s/kN)
        target_tsfc = fuel_flow_kgs / rated_thrust_kN

        # Inicializa vazão com um palpite razoável
        initial_m_dot = float(np.mean(m_dot_bounds))
        self.set_air_flow(initial_m_dot)

        # 1) Otimizar somente T04 para atingir o TSFC (m_dot fixo)
        def obj_t04(t04):
            self.set_t04(float(t04))
            return (self.get_tsfc() - target_tsfc) ** 2

        res_t04 = minimize_scalar(
            obj_t04, bounds=t04_bounds, method="bounded", options={"xatol": 1e-6}
        )
        if not getattr(res_t04, "success", True) and res_t04.status != 0:
            return {
                "success": False,
                "stage": "tsfc",
                "message": getattr(res_t04, "message", "Falha na otimização de T04"),
                "optimal_t04": None,
                "optimal_mass_flow_rate": None,
            }

        optimal_t04 = float(res_t04.x)
        self.set_t04(optimal_t04)
        self.t04_without_loss = optimal_t04

        # 2) Otimizar somente vazão mássica para atingir o empuxo (T04 fixo)
        def obj_mdot(m_dot):
            self.set_air_flow(float(m_dot))
            return ((self.get_thrust() - rated_thrust_kN) / rated_thrust_kN) ** 2

        res_mdot = minimize_scalar(
            obj_mdot, bounds=m_dot_bounds, method="bounded", options={"xatol": 1e-6}
        )
        if not getattr(res_mdot, "success", True) and res_mdot.status != 0:
            return {
                "success": False,
                "stage": "thrust",
                "message": getattr(res_mdot, "message", "Falha na otimização da vazão"),
                "optimal_t04": round(optimal_t04, 3),
                "optimal_mass_flow_rate": None,
            }

        optimal_m_dot = float(res_mdot.x)
        self.set_air_flow(optimal_m_dot)

        # Salva o ponto de projeto
        self.save_design_point()

        return {
            "success": True,
            "stage_sequence": ["tsfc_then_thrust"],
            "message_t04": getattr(res_t04, "message", "OK"),
            "message_mdot": getattr(res_mdot, "message", "OK"),
            "optimal_t04": round(optimal_t04, 3),
            "optimal_mass_flow_rate": round(optimal_m_dot, 3),
            "final_thrust_kN": round(self.get_thrust(), 1),
            "final_tsfc": round(self.get_tsfc(), 5),
        }

    def plot_calibration_result(
            self, target_tsfc: float,
            t04_range: ndarray = np.arange(1200, 1800, 1),
            title: str = "Análise da Relação TSFC vs. T04",
    ):
        original_t04 = self.t04
        tsfc_results = []
        logger.info(
            f"Calculando TSFC para visualização no intervalo T04 de {t04_range.min()}K a {t04_range.max()}K..."
        )
        for t04 in t04_range:
            self.set_t04(t04)
            tsfc_results.append(self.get_tsfc())

        df_plot = pd.DataFrame({"T04 (K)": t04_range, "TSFC (kg/s/kN)": tsfc_results})
        fig = px.line(
            df_plot,
            x="T04 (K)",
            y="TSFC (kg/s/kN)",
            title=title,
        )
        fig.add_hline(
            y=target_tsfc,
            line_dash="dash",
            annotation_text="TSFC Alvo",
            annotation_position="bottom right",
        )
        fig.update_layout(
            xaxis_title="T04 (K)",
            yaxis_title="TSFC (kg/(kN.s))",
            template="plotly",
        )
        # Restaurar o valor original de T04
        self.set_t04(original_t04)

        return fig

    def update_environment(
            self,
            mach: float = None,
            altitude: float = None,
            t_a: float = None,
            p_a: float = None,
            delta_temperature: float = None,
            percentage_of_rated_thrust: float | None = None,
    ):
        """
        Atualiza as condições de voo e encontra a rotação N2 para atingir um percentual do empuxo de projeto.

        Args:
            mach (float): Número de Mach.
            altitude (float): Altitude em pés [ft].
            t_a (float, optional): Temperatura ambiente em K. Se None, calcula a partir da altitude.
            p_a (float, optional): Pressão ambiente em kPa. Se None, calcula a partir da altitude.
            delta_temperature (float, optional): Variação de temperatura em relação à ISA.
            percentage_of_rated_thrust (float, optional): Percentual do empuxo de projeto a ser atingido (0.0 a 1.0).
                                                         Se fornecido, otimiza N2 para encontrar o empuxo alvo.
        """
        if not hasattr(self, "_design_point"):
            raise ValueError(
                "Ponto de projeto não definido. Execute save_design_point() após a calibração."
            )

        # 1. Atualiza as condições de ambiente e voo
        if mach is not None:
            self.mach = mach
        if altitude is not None:
            self.altitude = altitude
        if delta_temperature is not None:
            self.delta_isa_temperature = delta_temperature

        if altitude is not None or delta_temperature is not None:
            self.t_a, self.p_a, _, _ = atmosphere(
                self.altitude * ft2m,
                Tba=SEA_LEVEL_TEMPERATURE + self.delta_isa_temperature,
            )
            self.p_a /= 1000

        if t_a:
            self.t_a = t_a
        if p_a:
            self.p_a = p_a

        # 2. Se nenhum percentual de empuxo for fornecido, apenas atualiza os componentes e retorna.
        if percentage_of_rated_thrust is None:
            self.set_sea_level_air_flow(self.sea_level_air_flow)
            return

        target_thrust = self._design_point["rated_thrust"] * percentage_of_rated_thrust

        # 3. Define a função objetivo para o otimizador
        def objective_function(n2_ratio):
            self.update_from_N2(n2_ratio)
            return ((self.get_thrust() - target_thrust) / target_thrust) ** 2

        # 4. Executa a otimização para encontrar o N2
        result = minimize_scalar(
            objective_function,
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": 1e-6},
        )

        if not result.success:
            raise RuntimeError(f"Otimização de N2 para empuxo falhou: {result.message}")

        if result.fun > 0.01:
            result = minimize_scalar(objective_function, method="Golden", )

        if result.fun > 0.01 or result.x >= 0.99 or result.x <= 0.01:
            if result.fun > 0.01:
                logger.warning(f"Otimização de N2 não convergiu suficientemente. Erro final: {result.fun:.4f}")
            if result.fun > 0.01 or result.x <= 0.01:
                logger.warning("Revertendo N2 para 100% do valor de projeto.")
            self.update_from_N2(1.0)
            return

        self.update_from_N2(result.x)

    def get_thrust_envelope(self, mach: float, altitude_ft: float, delta_isa_temperature_K: float = 0.0) -> dict:
        """
        Calcula o envelope de empuxo (mínimo e máximo) para uma dada condição de voo.

        - O empuxo máximo é definido pela operação com N2 = 100%.
        - O empuxo mínimo (flight idle) é definido pela condição onde a pressão de
          saída do fan se iguala à pressão ambiente (P08 ≈ Pa), indicando que
          o fan não está mais provendo empuxo líquido significativo.

        Args:
            mach (float): Número de Mach de voo.
            altitude_ft (float): Altitude de voo em pés.
            delta_isa_temperature_K (float): Variação da temperatura em relação à ISA [K].

        Returns:
            dict: Dicionário contendo o empuxo mínimo e máximo em kN.
                  Ex: {'min_thrust_kN': 5.0, 'max_thrust_kN': 25.0}
        """
        if not hasattr(self, '_design_point'):
            raise ValueError("O motor deve ser calibrado e ter um ponto de projeto salvo.")

        # --- 1. Cálculo do Empuxo Máximo ---
        # Atualiza o ambiente para a condição de voo e define N2 para 100%
        self.update_environment(mach=mach, altitude=altitude_ft, delta_temperature=delta_isa_temperature_K)
        self.update_from_N2(1.0)
        max_thrust_kN = self.get_thrust()

        # --- 2. Cálculo do Empuxo Mínimo (Flight Idle) ---
        def objective_idle(n2_ratio):
            self.update_from_N2(n2_ratio)
            return abs(self.get_thrust())

        # Executa a otimização para encontrar o N2 de marcha lenta
        idle_result = minimize_scalar(
            objective_idle,
            bounds=(0.5, 1.0),  # Limites realistas para a operação do N2
            method='bounded',
            options={'xatol': 1e-6}
        )

        if not idle_result.success:
            logger.warning(
                f"Otimização para N2 de marcha lenta falhou. Usando o menor valor do intervalo como fallback.")
            n2_idle = 0.5
        else:
            if idle_result.fun > 0.01:
                logger.info(
                    f"Otimização para N2 de marcha lenta não convergiu suficientemente. Erro final: {idle_result.fun:.4f}")
            else:
                logger.info(f"Otimização para N2 de marcha lenta convergiu com sucesso!")
            n2_idle = idle_result.x

        # Define o motor para a condição de marcha lenta e obtém o empuxo
        self.update_from_N2(n2_idle)
        min_thrust_kN = self.get_thrust()

        return {
            'min_thrust_kN': min_thrust_kN,
            'max_thrust_kN': max_thrust_kN,
            'n2_for_min_thrust': n2_idle
        }


if __name__ == "__main__":
    from utils.configs import config_turbofan

    turbofan = Turbofan(config_turbofan)
    turbofan.set_air_flow(100)
    rated_thrust = 121.4  # kN
    fuel_flow = 1.293  # kg/s
    optimization_status = turbofan.calibrate_turbofan(rated_thrust, fuel_flow)
    # turbofan.update_environment(mach=0.0, percentage_of_rated_thrust=1)
    result = turbofan.get_thrust_envelope(0.789, 35e3)
    print(result)
