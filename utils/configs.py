config_ex23 = {
    "mach": 0.0,
    "t_a": 290.0,
    "p_a": 101.63,
    "bpr": 4.749,
    "prf": 1.69,
    "prc": 17.2,
    # "pr_bst": 1.0,
    "pressure_loss": 0.05,
    "T04": 1550,  # (K)

    # Eficiências e Gammas
    "eta_inlet": 0.97,
    "eta_fan": 0.93,
    "eta_compressor": 0.9,
    "eta_camara": 0.9995,
    "eta_turbina_compressor": 0.95,
    "eta_turbina_fan": 0.932,
    "eta_bocal_quente": 0.98,
    "eta_bocal_fan": 0.98,
    "gamma_inlet": 1.4,
    "gamma_fan": 1.4,
    "gamma_compressor": 1.37,
    "gamma_camara": 1.35,
    "gamma_turbina_compressor": 1.33,
    "gamma_turbina_fan": 1.33,
    "gamma_bocal_quente": 1.36,
    "gamma_bocal_fan": 1.4,

    # Dados operacionais
    "hydrogen_fraction": 0.0,
    "kerosene_PCI": 45e3,  # kJ/kg
    "hydrogen_PCI": 120e3,  # kJ/kg
    "mean_R_air": 288.3,  # (m^2 / (s^2*K))
    "Cp": 1.11,  # (kJ / (kg*K))
}

config_ex22 = {
    "mach": 0.85,
    "t_a": 216.7,
    "p_a": 18.75,

    # Eficiências e Gammas
    "eta_inlet": 0.97,
    "gamma_inlet": 1.4,
    "eta_fan": 0.85,
    "gamma_fan": 1.4,
    "eta_compressor": 0.85,
    "gamma_compressor": 1.37,
    "eta_camara": 1,
    "gamma_camara": 1.35,
    "eta_turbina_compressor": 0.9,
    "gamma_turbina_compressor": 1.33,
    "eta_turbina_fan": 0.9,
    "gamma_turbina_fan": 1.33,
    "eta_bocal_quente": 0.98,
    "gamma_bocal_quente": 1.36,
    "eta_bocal_fan": 0.98,
    "gamma_bocal_fan": 1.4,

    # Dados operacionais
    "bpr": 5,
    "prf": 1.5,
    "prc": 20,
    "hydrogen_fraction": 0.0,
    "pressure_loss": 0.0,
    "kerosene_PCI": 45e3,  # kJ/kg
    "hydrogen_PCI": 120e3,  # kJ/kg
    "mean_R_air": 288.3,  # (m^2 / (s^2*K))
    "Cp": 1.11,  # (kJ / (kg*K))
    "T04": 1600,  # (K)
}

config_turbofan = {
    "mach": 0.0,
    "altitude": 0.0,
    "t_a": 288.15,
    "p_a": 101.33,

    # Eficiências e Gammas
    "eta_inlet": 0.97,
    "gamma_inlet": 1.4,
    "eta_fan": 0.85,
    "gamma_fan": 1.4,
    "eta_compressor": 0.85,
    "gamma_compressor": 1.37,
    "eta_camara": 1,
    "gamma_camara": 1.35,
    "eta_turbina_compressor": 0.9,
    "gamma_turbina_compressor": 1.33,
    "eta_turbina_fan": 0.9,
    "gamma_turbina_fan": 1.33,
    "eta_bocal_quente": 0.98,
    "gamma_bocal_quente": 1.36,
    "eta_bocal_fan": 0.98,
    "gamma_bocal_fan": 1.4,

    # Dados operacionais
    "bpr": 5.0,
    "prf": 1.5,
    "prc": 19.0867,
    "hydrogen_fraction": 0.0,
    "pressure_loss": 0.0,
    "kerosene_PCI": 45e3,  # kJ/kg
    "hydrogen_PCI": 120e3,  # kJ/kg
    "mean_R_air": 288.3,  # (m^2 / (s^2*K))
    "Cp": 1.11,  # (kJ / (kg*K))
    "T04": 1750,  # (K)
}

config_teste1 = {
    "mach": 0.0,
    "t_a": 288.15,
    "p_a": 101.63,

    # Eficiências e Gammas
    "eta_inlet": 0.97,
    "gamma_inlet": 1.4,
    "eta_fan": 0.92,
    "gamma_fan": 1.4,
    "eta_compressor": 0.9,
    "gamma_compressor": 1.37,
    "eta_camara": 1,
    "gamma_camara": 1.35,
    "eta_turbina_compressor": 0.93,
    "gamma_turbina_compressor": 1.33,
    "eta_turbina_fan": 0.93,
    "gamma_turbina_fan": 1.33,
    "eta_bocal_quente": 0.98,
    "gamma_bocal_quente": 1.36,
    "eta_bocal_fan": 0.98,
    "gamma_bocal_fan": 1.4,

    # Dados operacionais
    "bpr": 10.5,
    "prf": 1.5,
    "prc": 25.5,
    "hydrogen_fraction": 0.0,
    "pressure_loss": 0.0,
    "kerosene_PCI": 45e3,  # kJ/kg
    "hydrogen_PCI": 120e3,  # kJ/kg
    "mean_R_air": 288.3,  # (m^2 / (s^2*K))
    "Cp": 1.11,  # (kJ / (kg*K))
    "T04": 1750,  # (K)
}

config_teste2 = {
    "mach": 0.0,
    "t_a": 288.2,
    "p_a": 101.63,

    # Eficiências e Gammas
    "eta_inlet": 0.97,
    "gamma_inlet": 1.4,
    "eta_fan": 1,
    "gamma_fan": 1.4,
    "eta_compressor": 1,
    "gamma_compressor": 1.37,
    "eta_camara": 1,
    "gamma_camara": 1.35,
    "eta_turbina_compressor": 1,
    "gamma_turbina_compressor": 1.33,
    "eta_turbina_fan": 1,
    "gamma_turbina_fan": 1.33,
    "eta_bocal_quente": 0.98,
    "gamma_bocal_quente": 1.36,
    "eta_bocal_fan": 0.98,
    "gamma_bocal_fan": 1.4,

    # Dados operacionais
    "bpr": 5,
    "prf": 1.5,
    "prc": 10,
    "hydrogen_fraction": 0.0,
    "pressure_loss": 0.0,
    "kerosene_PCI": 45e3,  # kJ/kg
    "hydrogen_PCI": 120e3,  # kJ/kg
    "mean_R_air": 288.3,  # (m^2 / (s^2*K))
    "Cp": 1.11,  # (kJ / (kg*K))
    "T04": 1500,  # (K)
}

config_ex71 = {
    "mach": 0.0,
    "t_a": 288.15,
    "p_a": 101.3,

    # Eficiências e Gammas
    "eta_inlet": 0.85,
    "gamma_inlet": 1.4,
    "eta_compressor": 0.75,
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
    "pr_tl": 2.8778,
    "hydrogen_fraction": 0.0,
    "pressure_loss": 0.0,
    "kerosene_PCI": 45e3,  # kJ/kg
    "hydrogen_PCI": 120e3,  # kJ/kg
    "mean_R_air": 288.3,  # (m^2 / (s^2*K))
    "Cp": 1.11,  # (kJ / (kg*K))
    "Cp_tl": 1.16,  # (kJ / (kg*K))
    "T04": 1600,  # (K)

    # Dados da gearbox e hélice
    "gearbox_efficiency": 0.98,  # potência que chega na hélice / potência que sai da turbina
    "propeller_efficiency": 0.85,
    "max_gearbox_power": 2050,  # kW -> se não fornecido, pode ser considerado como 0.8 * Pot_th (termodinâmica)
    "ref_pot_th": 2457,  # kW -> se não fornecido, pode ser considerado como 0.8 * Pot_th (termodinâmica)
}

config_ex72 = {
        "mach": 0.45,
        "t_a": 246.55,
        "p_a": 41,
    }


config_ex73 = {
    "mach": 0.45,
    "t_a": 246.55,
    "p_a": 41.0,

    # Eficiências e Gammas
    "eta_inlet": 0.97,
    "gamma_inlet": 1.4,
    "eta_fan": 0.9,
    "gamma_fan": 1.4,
    "eta_compressor": 0.75,
    "gamma_compressor": 1.37,
    "eta_camara": 0.9995,
    "gamma_camara": 1.35,
    "eta_turbina_compressor": 0.9,
    "gamma_turbina_compressor": 1.33,
    "eta_turbina_fan": 0.9,
    "gamma_turbina_fan": 1.33,
    "eta_bocal_quente": 0.98,
    "gamma_bocal_quente": 1.36,
    "eta_bocal_fan": 0.98,
    "gamma_bocal_fan": 1.4,

    # Dados operacionais
    "bpr": 4.2,
    "prf": 1.5,
    "prc": 8.33,
    "hydrogen_fraction": 0.0,
    "pressure_loss": 0.0,
    "kerosene_PCI": 45e3,  # kJ/kg
    "hydrogen_PCI": 120e3,  # kJ/kg
    "mean_R_air": 288.3,  # (m^2 / (s^2*K))
    "Cp": 1.11,  # (kJ / (kg*K))
    "T04": 1550,  # (K)
}
