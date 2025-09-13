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
        "T04": 1600,  # (K)
    }

    def __init__(self, config_dict):
        """
        Inicializa o motor turbofan com base em um dicionário de configuração.

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
        self.t04 = final_config["T04"]
        self.sea_level_air_flow = None
        self.air_flow = None

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
            self.Cp,
        )
        self.t06 = self.power_turbine.get_total_temperature()
        self.p06 = self.power_turbine.get_total_pressure()


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

    # Velocidade de Voo
    def get_flight_speed(self):
        return self.mach * np.sqrt(self.gamma_inlet * self.mean_R_air * self.t_a)