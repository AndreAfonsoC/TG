import numpy as np
import pandas as pd
import plotly.express as px
from numpy import ndarray
from scipy.optimize import minimize_scalar

from src.components.combustion_chamber import CombustionChamber
from src.components.compressor import Compressor
from src.components.inlet import Inlet
from src.components.nozzle import Nozzle
from src.components.turbine import Turbine, PowerTurbine
from utils.aux_tools import atmosphere, ft2m
from utils.corrections import model_corrections

SEA_LEVEL_TEMPERATURE = 288.15  # K
SEA_LEVEL_PRESSURE = 101.30  # kPa


class Turboprop:
    DEFAULT_CONFIG_DICT = {
        "mach": 0.0,
        "altitude": 0.0,  # em ft

        # Eficiências e Gammas
        "eta_inlet": 0.97,
        "gamma_inlet": 1.4,
        "eta_compressor": 0.85,
        "gamma_compressor": 1.37,
        "eta_camara": 1,
        "gamma_camara": 1.35,
        "eta_turbina_compressor": 0.9,
        "gamma_turbina_compressor": 1.33,
        "eta_turbina_livre": 0.9,
        "gamma_turbina_livre": 1.33,
        "eta_bocal_quente": 0.98,
        "gamma_bocal_quente": 1.36,

        # Dados operacionais
        "prc": 15.77,
        "pr_tl": 2.87,
        "hydrogen_fraction": 0.0,
        "pressure_loss_factor": 1.0,
        "kerosene_PCI": 45e3,  # kJ/kg
        "hydrogen_PCI": 120e3,  # kJ/kg
        "mean_R_air": 288.3,  # (m^2 / (s^2*K))
        "Cp": 1.11,  # (kJ / (kg*K))
        "Cp_tl": 1.16,  # (kJ / (kg*K))
        "T04": 1600,  # (K)

        # Dados da gearbox e hélice
        "gearbox_efficiency": 0.98,    # potência que chega na hélice / potência que sai da turbina
        "propeller_efficiency": 0.85,
        "max_gearbox_power": None,  # kW -> se não fornecido, pode ser considerado como 0.8 * Pot_th (termodinâmica)
        "ref_pot_th": 2456.49,  # kW -> se não fornecido, pode ser considerado como 0.8 * Pot_th (termodinâmica)
    }

    def __init__(self, config_dict):
        """
        Inicializa o motor turboprop com base em um dicionário de configuração.

        Usa os valores padrão da classe para quaisquer chaves ausentes no
        dicionário fornecido.
        """
        # Cria a configuração final mesclando os padrões com os fornecidos
        final_config = self.DEFAULT_CONFIG_DICT.copy()
        final_config.update(config_dict)

        # --- Dados do ambiente ---
        self.mach = final_config["mach"]
        self.altitude = final_config["altitude"]
        t_a_altitude, p_a_altitude, _, _ = atmosphere(self.altitude * ft2m)
        self.t_a = final_config.get("t_a", t_a_altitude) or t_a_altitude
        self.p_a = final_config.get("p_a", p_a_altitude / 1000) or p_a_altitude / 1000  # Divide por 1000 para passar para kPa

        # --- Eficiências e Gammas ---
        self.eta_inlet = final_config["eta_inlet"]
        self.gamma_inlet = final_config["gamma_inlet"]
        self.eta_compressor = final_config["eta_compressor"]
        self.gamma_compressor = final_config["gamma_compressor"]
        self.eta_camara = final_config["eta_camara"]
        self.gamma_camara = final_config["gamma_camara"]
        self.eta_turbina_compressor = final_config["eta_turbina_compressor"]
        self.gamma_turbina_compressor = final_config["gamma_turbina_compressor"]
        self.eta_turbina_livre = final_config["eta_turbina_livre"]
        self.gamma_turbina_livre = final_config["gamma_turbina_livre"]
        self.eta_bocal_quente = final_config["eta_bocal_quente"]
        self.gamma_bocal_quente = final_config["gamma_bocal_quente"]

        # --- Dados operacionais ---
        self.prc = final_config["prc"]
        self.pr_tl = final_config["pr_tl"]
        self.hydrogen_fraction = final_config["hydrogen_fraction"]
        self.pressure_loss_factor = final_config["pressure_loss_factor"]
        self.kerosene_PCI = final_config["kerosene_PCI"]
        self.hydrogen_PCI = final_config["hydrogen_PCI"]
        self.mean_R_air = final_config["mean_R_air"]
        self.Cp = final_config["Cp"]
        self.Cp_tl = final_config["Cp_tl"]
        self.t04 = final_config["T04"]
        self.sea_level_air_flow = None
        self.air_flow = None

        # --- Dados da gearbox e hélice ---
        self.gearbox_efficiency = final_config["gearbox_efficiency"]
        self.propeller_efficiency = final_config["propeller_efficiency"]
        self.max_gearbox_power = final_config["max_gearbox_power"]
        self.ref_pot_th = final_config["ref_pot_th"]

        # Adiciona um atributo para armazenar os modelos de correção
        self._correction_models = {}
        self._initialize_correction_models()

    def _initialize_correction_models(self):
        """
        Inicializa os modelos polinomiais de correção adimensional.
        Cada modelo retorna a razão entre o valor em um dado N2 e o valor no ponto de projeto.
        """
        self._correction_models = model_corrections(is_turboprop=False)

    def update_turboprop_components(self):
        # 1. Difusor
        self.inlet = Inlet(self.t_a, self.p_a, self.mach, self.eta_inlet, self.gamma_inlet)
        self.t02 = self.inlet.get_total_temperature()
        self.p02 = self.inlet.get_total_pressure()

        # 2. Compressor
        self.compressor = Compressor(self.t02, self.p02, self.prc, self.eta_compressor, self.gamma_compressor)
        self.t03 = self.compressor.get_total_temperature()
        self.p03 = self.compressor.get_total_pressure()

        # 3. Câmara de Combustão
        self.combustion_chamber = CombustionChamber(
            self.t03,
            self.p03,
            self.Cp,
            self.t04,
            self.eta_camara,
            self.kerosene_PCI,
            self.hydrogen_PCI,
            self.hydrogen_fraction,
            self.pressure_loss_factor,
        )
        self.p04 = self.combustion_chamber.get_total_pressure()
        self.fuel_to_air_ratio = self.combustion_chamber.get_fuel_to_air_ratio()

        # 4. Turbina do compressor
        self.compressor_turbine = Turbine(
            self.t04,
            self.p04,
            self.t02,
            self.t03,
            self.eta_turbina_compressor,
            self.gamma_turbina_compressor,
        )
        self.t05 = self.compressor_turbine.get_total_temperature()
        self.p05 = self.compressor_turbine.get_total_pressure()

        # 5. Turbina Livre
        self.power_turbine = PowerTurbine(
            self.t05,
            self.p05,
            self.pr_tl,
            self.eta_turbina_livre,
            self.gamma_turbina_livre,
            self.Cp_tl,
        )
        self.p06 = self.power_turbine.get_total_pressure()
        self.total_iso_temperature = self.power_turbine.get_total_isentropic_temperature()
        self.iso_work = self.power_turbine.get_isentropic_work()
        self.real_work = self.power_turbine.get_real_work()
        self.t06 = self.power_turbine.get_total_temperature()

        if self.air_flow is None:
            raise ValueError("Vazão de ar não definida.")
        else:
            self.pot_th = self.power_turbine.get_power(
                self.air_flow,
                self.fuel_to_air_ratio,
            )

        self.pot_tl = min(self.max_gearbox_power, self.pot_th) if self.max_gearbox_power is not None else self.pot_th
        self.pot_gear = self.gearbox_efficiency * self.pot_tl

        # 6. Bocal dos gases quentes
        self.core_nozzle = Nozzle(
            self.t06,
            self.p06,
            self.p_a,
            self.eta_bocal_quente,
            self.gamma_bocal_quente,
            self.mean_R_air,
        )
        self.u_core = self.core_nozzle.get_exhaust_velocity()

        # 9. Velocidade de voo
        self.u_flight = self.get_flight_speed()


    def calibrate_pot_th(self, pr_tl_min=1.0, pr_tl_max=10.0, step=0.001) -> None:
        """
        Varre o parâmetro pr_tl para encontrar o valor ótimo onde self.pot_th == self.ref_pot_th.
        Atualiza self.pr_tl com o valor encontrado.

        Args:
            pr_tl_min (float): valor mínimo de pr_tl
            pr_tl_max (float): valor máximo de pr_tl
            step (float): passo de variação de pr_tl
        """
        best_pr_tl = pr_tl_min
        min_diff = float('inf')
        pr_tl_range = np.arange(pr_tl_min, pr_tl_max + step, step)
        for pr_tl in pr_tl_range:
            self.set_pr_tl(pr_tl)
            self.update_turboprop_components()
            diff = abs(self.pot_th - self.ref_pot_th)
            if diff < min_diff:
                min_diff = diff
                best_pr_tl = pr_tl
            if min_diff < step:  # tolerância
                break
        self.set_pr_tl(best_pr_tl)
        self.update_turboprop_components()
        self.save_design_point()
        self._calibrate_pot_th_changing_n2()

    def _calibrate_pot_th_changing_n2(self, N2_min=0.05, N2_max=2.0, step=0.001):
        """
        Varre o parâmetro N2 para encontrar o valor ótimo onde self.pot_th == self.max_gearbox_power.
        Atualiza os parâmetros do turboprop com o N2 encontrado.
        """
        if self.max_gearbox_power is None:
            raise ValueError("max_gearbox_power não está definido.")

        best_N2 = N2_min
        min_diff = float('inf')
        N2_range = np.arange(N2_min, N2_max + step, step)
        for N2 in N2_range:
            self.update_from_N2(N2)
            diff = abs(self.pot_th - self.max_gearbox_power)
            if diff < min_diff:
                min_diff = diff
                best_N2 = N2
            if min_diff < step:  # tolerância
                break
        self.update_from_N2(best_N2)
        self.save_design_point()

    def save_design_point(self):
        """
        Salva os valores dos parâmetros no ponto de projeto após a calibração.
        """
        self._design_point = {
            'pr_tl': self.pr_tl,
            'prc': self.prc,
            't04': self.t04,
            'eta_fan': self.eta_fan,
            'eta_compressor': self.eta_compressor,
            'eta_camara': self.eta_camara,
            'eta_turbina_livre': self.eta_turbina_livre,
            'eta_turbina_compressor': self.eta_turbina_compressor,
            'eta_bocal_quente': self.eta_bocal_quente,
            'air_flow': self.air_flow,
        }

    def update_from_N2(self, N2: float, N2_design: float = 1.0):
        """
        Atualiza os parâmetros operacionais do turboprop com base na rotação normalizada do eixo de alta (N2).

        Args:
            N2 (float): Rotação atual do eixo de alta pressão (normalizada)
            N2_design (float): Rotação de projeto do eixo de alta pressão (normalizada). Default = 1.0
        """
        if not hasattr(self, '_design_point'):
            raise ValueError("Ponto de projeto não definido. Execute save_design_point() após a calibração.")

        models = self._correction_models
        N2_ratio = N2 / N2_design
        N1_ratio = models['N1_from_N2'](N2_ratio)
        self.N1_ratio = N1_ratio
        self.N2_ratio = N2_ratio

        # --- 1. Pressões e Temperaturas ---
        self.prc = self._design_point['prc'] * models['Prc_from_N2'](N2_ratio)
        self.t04 = self._design_point['t04'] * models['T04_from_N2'](N2_ratio)

        # --- 2. EficiÊncias ---
        self.eta_compressor = self._design_point['eta_compressor'] * models['eta_c_from_N2'](N2_ratio)
        self.eta_turbina_livre = self._design_point['eta_turbina_livre'] * models['eta_turbina_livre_from_N2'](N2_ratio)
        self.eta_turbina_compressor = self._design_point['eta_turbina_compressor'] * models['eta_t_from_N2'](N2_ratio)
        self.eta_camara = self._design_point['eta_camara'] * models['eta_b_from_N2'](N2_ratio)

        # --- 3. Razão de expansão ---
        self.pr_tl = self.pr_tl * models['pr_tl_from_N2'](N2_ratio)

        # Vazão mássica (nesse caso hot_air_flow é igual a air_flow)
        hot_air_flow_ratio = models['m_dot_H_from_N2'](N2_ratio)
        hot_air_flow = self._design_point['air_flow'] * hot_air_flow_ratio
        self.set_air_flow(hot_air_flow)

        # Atualiza os componentes do turboprop
        self.update_turboprop_components()

        print(f"Parâmetros atualizados para N2/N2_design = {N2_ratio:.4f}")


    # Velocidade de Voo
    def get_flight_speed(self):
        return self.mach * np.sqrt(self.gamma_inlet * self.mean_R_air * self.t_a)

    def set_air_flow(self, air_flow: float):
        self.air_flow = air_flow
        correction_factor = (SEA_LEVEL_TEMPERATURE / SEA_LEVEL_PRESSURE) * (self.p_a / self.t_a)
        self.sea_level_air_flow = air_flow / correction_factor
        self.update_turboprop_components()

    def get_tsfc(self):
        return self.fuel_to_air_ratio * self.air_flow / self.get_thrust()

    def get_bsfc(self):
        return self.fuel_to_air_ratio * self.air_flow / self.pot_tl

    def get_ebsfc(self):
        term1 = self.fuel_to_air_ratio * self.air_flow
        term2 = self.pot_tl + self.get_thrust() * self.get_flight_speed()

        return term1 / term2

    def get_nozzle_specific_thrust(self):
        """
        Calcula o empuxo específico do motor turboprop.

        O empuxo específico é a força de empuxo gerada por unidade de vazão mássica de ar quente.
        Considera a contribuição do núcleo (core), descontando a velocidade de voo.

        Returns:
            float: Empuxo específico em kN/(kg/s).

        """
        nozzle_specific_thrust_N = (1 + self.fuel_to_air_ratio) * self.u_core - self.u_flight
        nozzle_specific_thrust_kN = nozzle_specific_thrust_N / 1000

        return nozzle_specific_thrust_kN

    def get_nozzle_thrust(self):
        return self.get_nozzle_specific_thrust() * self.air_flow

    def get_air_flow(self):
        if self.air_flow is None:
            raise ValueError("Vazão de ar não definida.")
        else:
            return self.air_flow

    def get_propeler_thrust(self):
        thrust_N = self.propeller_efficiency * self.pot_gear / self.u_flight
        thurst_kN = thrust_N / 1000

        return thurst_kN

    def get_thrust(self, in_kN: bool = True):
        total_thrust_kN = self.get_nozzle_thrust() + self.get_propeler_thrust()

        if in_kN:
            return total_thrust_kN
        else:
            return total_thrust_kN * 1000

    def set_pr_tl(self, pr_tl):
        self.pr_tl = pr_tl

    def get_pr_tl(self):
        return self.pr_tl

    def get_total_pressure_after_tl(self):
        return self.p06

    def get_total_temperature_after_tl(self):
        return self.t06

