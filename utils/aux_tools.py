import numpy as np

# Conversão de unidades
ft2m = 0.3048
kt2ms = 0.514444
lb2N = 4.44822
min2s = 60.0

# Condições ao nível do mar padrão
SEA_LEVEL_TEMPERATURE = 288.15  # K
SEA_LEVEL_PRESSURE = 101.30  # kPa
SEA_LEVEL_GRAVITY = 9.80665  # m/s²

# ==============================================================================
# FATORES DE EMISSÃO (BASEADO EM ESTEQUIOMETRIA)
# ==============================================================================
# Todo: conferir esses valores com fontes confiáveis
# Reação do Querosene (aproximado como C12H23):
# C12H23 + 17.75 O2 -> 12 CO2 + 11.5 H2O
# Massa molar C12H23 ≈ 167 g/mol | Massa molar CO2 = 44 g/mol | Massa molar H2O = 18 g/mol
# Fator CO2 = (12 * 44) / 167 ≈ 3.15
# Fator H2O = (11.5 * 18) / 167 ≈ 1.24
CO2_PER_KEROSENE_MASS = 3.15  # kg de CO2 por kg de querosene queimado
H2O_PER_KEROSENE_MASS = 1.24  # kg de H2O por kg de querosene queimado

# Reação do Hidrogênio:
# 2 H2 + O2 -> 2 H2O
# Massa molar H2 ≈ 2 g/mol | Massa molar H2O = 18 g/mol
# Fator H2O = (2 * 18) / (2 * 2) ≈ 8.94
H2O_PER_HYDROGEN_MASS = 8.94   # kg de H2O por kg de hidrogênio queimado

#==============================================

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
    Tb = Tba-Tstdb[0]
    # air viscosity
    mu0 = 18.27e-6 # [Pa s]
    T0 = 291.15 # [K]
    C = 120 # [K]

    # geopotential altitude
    H = r*z/(r+z)

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
    T = Tstdb[i]+A[i]*(H-Ht[i])+Tb

    # Calculating pressure
    if A[i] == 0:
        p = pb[i]*np.exp(-g0*(H-Ht[i])/R/(Tstdb[i]+Tb))
    else:
        p = pb[i]*(T/(Tstdb[i]+Tb))**(-g0/A[i]/R)

    # Calculating density
    rho = p/R/T

    # Calculating viscosity with Sutherland's Formula
    mu=mu0*(T0+C)/(T+C)*(T/T0)**(1.5)

    return T,p,rho,mu