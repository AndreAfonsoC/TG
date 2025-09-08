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
        "prc": 28.6 / 1.5,
        "hydrogen_fraction": 0.0,
        "pressure_loss_factor": 1.0,
        "kerosene_PCI": 45e3,  # kJ/kg
        "hydrogen_PCI": 120e3,  # kJ/kg
        "mean_R_air": 288.3,  # (m^2 / (s^2*K))
        "Cp": 1.11,  # (kJ / (kg*K))
        "T04": 1600,  # (K)
    }
