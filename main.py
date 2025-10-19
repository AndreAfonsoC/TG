import pandas as pd

from src.mission import MissionManager
from src.turbofan import Turbofan

# Configura o pandas para exibir todas as colunas e formatar os números
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.options.display.float_format = "{:,.4f}".format

# --- 1. CONFIGURAÇÃO E CALIBRAÇÃO DO MOTOR ---

print("=" * 80)
print("FASE DE CONFIGURAÇÃO: Calibrando o motor para o ponto de projeto...")
print("=" * 80)

# Parâmetros de calibração para um motor similar ao CFM56-7B em condição de decolagem (Take-Off)
# Dados baseados em documentação de referência para motores dessa classe.
config_calibracao = {
    "mach": 0.0,
    "altitude": 0,  # Nível do mar
    "hydrogen_fraction": 0.0,  # Calibração feita com 100% querosene
}

# Cria a instância do motor
meu_motor = Turbofan(config_calibracao)

# Dados alvo para a calibração (Empuxo e consumo em Take-Off)
# Valores típicos para um motor da classe do CFM56-7B
empuxo_de_projeto_kN = 121.4  # (equivalente a ~27300 lbf)
consumo_decolagem_kgs = 1.293  # (consumo de combustível em kg/s)

# Executa a calibração para encontrar T04 e a vazão de ar que correspondem ao desempenho de projeto
# Este méthodo já chama 'save_design_point()' internamente no final.
resultado_calibracao = meu_motor.calibrate_turbofan(
    rated_thrust_kN=empuxo_de_projeto_kN,
    fuel_flow_kgs=consumo_decolagem_kgs,
    t04_bounds=(1400, 1800),
    m_dot_bounds=(300, 600),
)

if not resultado_calibracao["success"]:
    raise RuntimeError(
        "A calibração do motor falhou. Verifique os parâmetros de entrada."
    )

print("\nMotor calibrado com sucesso!")
print(f"  - T04 de Projeto: {resultado_calibracao['optimal_t04']:.2f} K")
print(
    f"  - Vazão de Ar de Projeto (nível do mar): {resultado_calibracao['optimal_mass_flow_rate']:.2f} kg/s"
)
print(
    f"  - Empuxo de Projeto verificado: {resultado_calibracao['final_thrust_kN']:.2f} kN"
)
print(
    f"  - TSFC de Projeto verificado: {resultado_calibracao['final_tsfc']:.5f} kg/(s*kN)"
)

# --- 2. DEFINIÇÃO DA AERONAVE E MISSÃO ---

print("\n" + "=" * 80)
print("FASE DE DEFINIÇÃO: Construindo o perfil da missão...")
print("=" * 80)

# Peso da aeronave sem combustível (Zero Fuel Weight). Valor típico para um B737-800.
zero_fuel_weight_kg = 61500

# Cria o gerenciador da missão
missao = MissionManager(engine=meu_motor, zero_fuel_weight=zero_fuel_weight_kg)

# Dados da missão baseados na Tabela 2.1 (GRU -> FOR)
perfil_de_voo = [
    {
        "name": "Taxi (Saída)",
        "duration_min": 1,
        "altitude_ft": 0,
        "mach": 0.0,
        "thrust_percentage": 100,
    },
    {
        "name": "Decolagem",
        "duration_min": 1,
        "altitude_ft": 0,
        "mach": 0.2,
        "thrust_percentage": 100,
    },
    {
        "name": "Subida 1",
        "duration_min": 8,
        "altitude_ft": 5830,
        "mach": 0.298,
        "thrust_percentage": 1,
    },
    {
        "name": "Subida 2",
        "duration_min": 8,
        "altitude_ft": 17500,
        "mach": 0.494,
        "thrust_percentage": 1,
    },
    {
        "name": "Subida 3",
        "duration_min": 8,
        "altitude_ft": 29170,
        "mach": 0.691,
        "thrust_percentage": 1,
    },
    {
        "name": "Cruzeiro",
        "duration_min": 10,
        "altitude_ft": 35000,
        "mach": 0.789,
        "thrust_percentage": 1,
    },
    {
        "name": "Loiter",
        "duration_min": 10,
        "altitude_ft": 15000,
        "mach": 0.4,
        "thrust_percentage": 1,
    },
    {
        "name": "Descida 1",
        "duration_min": 8,
        "altitude_ft": 29170,
        "mach": 0.691,
        "thrust_percentage": 1,
    },
    {
        "name": "Descida 2",
        "duration_min": 8,
        "altitude_ft": 17500,
        "mach": 0.494,
        "thrust_percentage": 1,
    },
    {
        "name": "Descida 3",
        "duration_min": 8,
        "altitude_ft": 5830,
        "mach": 0.298,
        "thrust_percentage": 1,
    },
    {
        "name": "Pouso",
        "duration_min": 1,
        "altitude_ft": 0,
        "mach": 0.2,
        "thrust_percentage": 1,
    },
    {
        "name": "Taxi (Chegada)",
        "duration_min": 5,
        "altitude_ft": 0,
        "mach": 0.0,
        "thrust_percentage": 7,
    },
]

# Adiciona cada fase ao gerenciador da missão
# Para este exemplo, usamos a mesma estratégia de queima 'proporcional' para todas as fases.
# Você pode customizar a 'burn_strategy' para cada fase individualmente.
for fase in perfil_de_voo:
    missao.add_phase(
        name=fase["name"],
        duration_min=fase["duration_min"],
        altitude_ft=fase["altitude_ft"],
        mach=fase["mach"],
        thrust_percentage=fase["thrust_percentage"],
        burn_strategy="kerosene_only",  # Altere aqui se desejar (ex: 'hydrogen_only', 'kerosene_only')
    )

# --- 3. EXECUÇÃO E ANÁLISE DA SIMULAÇÃO ---

print("\n" + "=" * 80)
print("FASE DE EXECUÇÃO: Solucionando a missão...")
print("=" * 80)

# Define a fração de hidrogênio e o tipo de tanque para a missão a ser simulada
fracao_h2_missao = 0.0  # Exemplo: 30% da massa de combustível é hidrogênio
tipo_de_tanque = "TYPE_IV"

# Executa a simulação. Este méthodo irá iterar até encontrar o combustível necessário.
missao.solve_mission_fuel(
    chi_initial_mission=fracao_h2_missao,
    tank_type=tipo_de_tanque,
    fuel_guess_bounds=(76, 50e3),  # Intervalo de busca para a massa de combustível
)

print("\n" + "=" * 80)
print("ANÁLISE FINAL: Resultados consolidados da missão")
print("=" * 80)

if missao.results:
    # Acessa o objeto FuelSystem final diretamente a partir dos resultados
    fs_final_obj = missao.results["final_fuel_system_object"]

    # Imprime um resumo dos resultados totais
    print(
        f"Combustível Total Necessário: {missao.results['Combustível Inicial (kg)']:.2f} kg"
    )
    print(f"  - Hidrogênio Inicial: {fs_final_obj.hydrogen_mass_initial:.2f} kg")
    print(f"  - Querosene Inicial: {fs_final_obj.kerosene_mass_initial:.2f} kg")

    tanque_peso = fs_final_obj.get_tank_weight()
    print(f"Peso do Tanque de H2 ({fs_final_obj.tank_type}): {tanque_peso:.2f} kg")

    # Calcula o peso total de decolagem (Take-Off Weight)
    tow = missao.zero_fuel_weight + fs_final_obj.get_total_weight_at_takeoff()
    print(f"Peso Total de Decolagem (TOW): {tow:,.2f} kg")

    print("\n--- EMISSÕES TOTAIS DA MISSÃO ---")
    print(f"CO2: {missao.results['Emissão Total de CO2 (kg)']:.2f} kg")
    print(f"H2O: {missao.results['Emissão Total de H2O (kg)']:.2f} kg")

    print("\n--- COMBUSTÍVEL NÃO UTILIZADO (RESERVA) ---")
    print(f"Hidrogênio Remanescente: {fs_final_obj.get_remaining_hydrogen():.2f} kg")
    print(f"Querosene Remanescente: {fs_final_obj.get_remaining_kerosene():.2f} kg")

else:
    print("A simulação não foi concluída com sucesso ou não foi executada.")
