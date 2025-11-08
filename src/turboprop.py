import logging
import os
import numpy as np
from scipy.optimize import minimize_scalar

from src.components.combustion_chamber import CombustionChamber
from src.components.compressor import Compressor
from src.components.inlet import Inlet
from src.components.nozzle import Nozzle
from src.components.turbine import Turbine, PowerTurbine
from utils.aux_tools import atmosphere, ft2m, CO2_PER_KEROSENE_MASS, H2O_PER_KEROSENE_MASS, H2O_PER_HYDROGEN_MASS
from utils.corrections import model_corrections
from utils.configs import DEFAULT_CONFIG_TURBOPROP

# --- Configuração do Logger (Padrão do Projeto) ---
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(LOG_DIR, "turboprop.log"), mode='w', encoding='utf-8')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

SEA_LEVEL_TEMPERATURE = 288.15  # K
SEA_LEVEL_PRESSURE = 101.30  # kPa


class Turboprop:

    def __init__(self, config_dict: dict):
        """
        Inicializa o motor turboprop com base em um dicionário de configuração.

        Usa os valores padrão da classe para quaisquer chaves ausentes no
        dicionário fornecido.
        """
        # Cria a configuração final mesclando os padrões com os fornecidos
        self.final_config = DEFAULT_CONFIG_TURBOPROP.copy()
        self.final_config.update(config_dict)

        # --- Dados do ambiente ---
        self.mach = self.final_config["mach"]
        self.altitude = self.final_config["altitude"]
        t_a_altitude, p_a_altitude, _, _ = atmosphere(self.altitude * ft2m)
        self.t_a = self.final_config.get("t_a", t_a_altitude) or t_a_altitude
        self.p_a = self.final_config.get("p_a",
                                         p_a_altitude / 1000) or p_a_altitude / 1000  # Divide por 1000 para passar para kPa

        # --- Eficiências e Gammas ---
        self.eta_inlet = self.final_config["eta_inlet"]
        self.gamma_inlet = self.final_config["gamma_inlet"]
        self.eta_compressor = self.final_config["eta_compressor"]
        self.gamma_compressor = self.final_config["gamma_compressor"]
        self.eta_camara = self.final_config["eta_camara"]
        self.gamma_camara = self.final_config["gamma_camara"]
        self.eta_turbina_compressor = self.final_config["eta_turbina_compressor"]
        self.gamma_turbina_compressor = self.final_config["gamma_turbina_compressor"]
        self.eta_turbina_livre = self.final_config["eta_turbina_livre"]
        self.gamma_turbina_livre = self.final_config["gamma_turbina_livre"]
        self.eta_bocal_quente = self.final_config["eta_bocal_quente"]
        self.gamma_bocal_quente = self.final_config["gamma_bocal_quente"]

        # --- Dados operacionais ---
        self.bpr = 0.0  # bypass ratio é zero para turboprop
        self.prc = self.final_config["prc"]
        self.pr_tl = self.final_config["pr_tl"]
        self.hydrogen_fraction = self.final_config["hydrogen_fraction"]
        self.pressure_loss = self.final_config["pressure_loss"]
        self.kerosene_PCI = self.final_config["kerosene_PCI"]
        self.hydrogen_PCI = self.final_config["hydrogen_PCI"]
        self.mean_R_air = self.final_config["mean_R_air"]
        self.Cp = self.final_config["Cp"]
        self.Cp_tl = self.final_config["Cp_tl"]
        self.t04 = self.final_config["T04"]
        self.sea_level_air_flow = None
        self.air_flow = None

        # --- Dados da gearbox e hélice ---
        self.gearbox_efficiency = self.final_config["gearbox_efficiency"]
        self.propeller_efficiency = self.final_config["propeller_efficiency"]
        self.max_gearbox_power = self.final_config["max_gearbox_power"]
        self.ref_pot_th = self.final_config["ref_pot_th"]

        # --- Constantes Físicas Adicionais ---
        # Velocidade de corte para linearização do empuxo estático (Mach 0.1 ~ 34 m/s SLS)
        # Usada para evitar divisão por zero em get_propeler_thrust
        self.MACH_THRESHOLD_STATIC = 0.1

        # Adiciona um atributo para armazenar os modelos de correção
        self._correction_models = {}
        self._initialize_correction_models()

    def _initialize_correction_models(self):
        """
        Inicializa os modelos polinomiais de correção adimensional.
        Cada modelo retorna a razão entre o valor em um dado N2 e o valor no ponto de projeto.
        """
        self._correction_models = model_corrections(is_turbofan=False)

    def update_final_config(self, config_dict: dict):
        """
        Atualiza a configuração do motor com novos valores fornecidos em um dicionário.
        Quaisquer chaves ausentes manterão seus valores atuais.
        """
        self.final_config.update(config_dict)

        # Reaplica os valores atualizados
        self.__init__(self.final_config)
        if hasattr(self, '_design_point'):
            self.set_sea_level_air_flow(self._design_point['sea_level_air_flow'])

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
            self.gamma_camara,
            self.kerosene_PCI,
            self.hydrogen_PCI,
            self.hydrogen_fraction,
            self.pressure_loss,
        )
        self.p04 = self.combustion_chamber.get_total_pressure()
        self.t04 = self.combustion_chamber.get_total_temperature_out()
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
            self.turbine_airflow = self.power_turbine.get_total_air_flow(self.air_flow, self.fuel_to_air_ratio)
            self.pot_th = self.power_turbine.get_power(self.air_flow, self.fuel_to_air_ratio)

        self.pot_tl = self.pot_th
        self.pot_gear = self.gearbox_efficiency * self.pot_tl

        # 6. Bocal dos gases quentesf
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

    def calibrate_pot_th(self, pr_tl_min: float = 1.0, pr_tl_max: float = 10.0) -> None:
        """
        Calibra o parâmetro pr_tl para que self.pot_th seja igual a self.ref_pot_th
        usando um otimizador escalar.

        Este méthodo encontra o valor ótimo de 'pr_tl' que minimiza a diferença
        absoluta entre a potência termodinâmica calculada e a de referência.

        Args:
            pr_tl_min (float): Limite inferior da busca para pr_tl.
            pr_tl_max (float): Limite superior da busca para pr_tl.
        """

        # 1. Definir a função objetivo (a ser minimizada)
        def objective_function(pr_tl: float) -> float:
            """
            Calcula a diferença absoluta entre a potência calculada e a de referência
            para um determinado valor de pr_tl.
            """
            self.set_pr_tl(pr_tl)
            self.update_turboprop_components()
            return abs(self.pot_th - self.ref_pot_th)

        # 2. Executar o otimizador
        result = minimize_scalar(
            objective_function,
            bounds=(pr_tl_min, pr_tl_max),
            method='bounded'
        )

        # 3. Verificar o sucesso e aplicar o resultado
        if not result.success:
            raise RuntimeError(f"A calibração de pot_th falhou: {result.message}")

        if result.fun > 0.01:
            result = minimize_scalar(objective_function, method="Golden")

        best_pr_tl = result.x

        # 4. Atualizar o estado final do objeto com o valor ótimo encontrado
        self.set_pr_tl(best_pr_tl)
        self.update_turboprop_components()
        self.save_design_point()
        logger.info(f"Calibração de pot_th concluída. pr_tl ótimo: {result.x:.4f}")
        self._calibrate_pot_th_changing_n2()

    def _calibrate_pot_th_changing_n2(self, N2_min: float = 0.00, N2_max: float = 1.0) -> None:
        """
        Ajusta a rotação N2 para que a potência termodinâmica do motor atinja
        o limite da caixa de redução (flat-rating).

        Utiliza um otimizador para encontrar o valor de N2 que minimiza a diferença
        entre self.pot_th e self.max_gearbox_power.

        Args:
            N2_min (float): Limite inferior da busca para N2 (ex: 50%).
            N2_max (float): Limite superior da busca para N2 (ex: 100%).
        """
        if self.max_gearbox_power is None:
            raise ValueError("O atributo 'max_gearbox_power' não foi definido.")

        # 1. Função-objetivo que calcula o erro a ser minimizado.
        def objective_function(N2: float) -> float:
            """
            Calcula o erro absoluto entre a potência do motor e o limite da
            gearbox para uma dada rotação N2.
            """
            self.update_from_N2(N2)
            return abs(self.pot_th - self.max_gearbox_power)

        # 2. Executa o otimizador para encontrar o N2 ótimo.
        result = minimize_scalar(
            objective_function,
            bounds=(N2_min, N2_max),
            method='bounded'
        )

        # 3. Valida o resultado e lança um erro se a calibração falhar.
        if not result.success:
            raise RuntimeError(f"A calibração de N2 falhou: {result.message}")

        # 4. Atualiza o estado final do objeto com o valor ótimo e salva.
        self.update_from_N2(result.x)
        logger.info(f"Calibração de N2 para max_gearbox_power concluída. N2 ótimo: {result.x:.4f}")

    def save_design_point(self):
        """
        Salva os valores dos parâmetros no ponto de projeto após a calibração.
        """
        logger.info("Salvando ponto de projeto do turboprop...")
        self._design_point = {
            'pr_tl': self.pr_tl,
            'prc': self.prc,
            't04': self.t04,
            'eta_compressor': self.eta_compressor,
            'eta_camara': self.eta_camara,
            'eta_turbina_livre': self.eta_turbina_livre,
            'eta_turbina_compressor': self.eta_turbina_compressor,
            'eta_bocal_quente': self.eta_bocal_quente,
            'sea_level_air_flow': self.sea_level_air_flow,
            # Salva o empuxo "rated" como o empuxo estático total no ponto de projeto
            # para servir de referência para input de porcentagem.
            'rated_thrust': self.get_thrust(in_kN=True),
            'rated_power': self.max_gearbox_power if self.max_gearbox_power else self.ref_pot_th
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

        # Permite N2_ratio == 1.0 recalcular para garantir consistência se o ambiente mudou
        # if abs(N2_ratio - 1.0) <= 1e-4: return

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
        self.pr_tl = self._design_point['pr_tl'] * models['pr_tl_from_N2'](N2_ratio)

        # Vazão mássica (nesse caso hot_air_flow é igual a air_flow)
        hot_air_flow_ratio = models['m_dot_H_from_N2'](N2_ratio)
        self.set_sea_level_air_flow(self._design_point['sea_level_air_flow'] * hot_air_flow_ratio)

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

    # Velocidade de Voo
    def get_flight_speed(self):
        return self.mach * np.sqrt(self.gamma_inlet * self.mean_R_air * self.t_a)

    def set_air_flow(self, air_flow: float):
        self.air_flow = air_flow
        correction_factor = (SEA_LEVEL_TEMPERATURE / SEA_LEVEL_PRESSURE) * (self.p_a / self.t_a)
        self.sea_level_air_flow = air_flow / correction_factor
        self.update_turboprop_components()

    def set_sea_level_air_flow(self, sea_level_air_flow: float):
        self.sea_level_air_flow = sea_level_air_flow

        correction_factor = (SEA_LEVEL_TEMPERATURE / SEA_LEVEL_PRESSURE) * (self.p_a / self.t_a)
        self.air_flow = sea_level_air_flow * correction_factor
        self.update_turboprop_components()

    def get_fuel_consumption(self):
        return self.fuel_to_air_ratio * self.air_flow

    def get_tsfc(self):
        thrust = self.get_thrust(in_kN=True)
        if thrust <= 1e-6: return float('inf')  # Evita divisão por zero
        return self.get_fuel_consumption() / thrust

    def get_bsfc(self):
        if self.pot_tl <= 1e-6: return float('inf')
        return self.get_fuel_consumption() / self.pot_tl

    def get_ebsfc(self):
        equiv_power = self.pot_tl + self.get_nozzle_thrust() * self.get_flight_speed() / 1000  # kW
        if equiv_power <= 1e-6: return float('inf')
        return self.get_fuel_consumption() / equiv_power

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

    def get_propeler_thrust(self, in_kN: bool = True) -> float:
        """
        Calcula o empuxo gerado pela hélice, com linearização para baixas velocidades.
        Evita divisão por zero quando Mach tende a 0.
        """
        # Usa a maior velocidade entre a atual e a de corte (Mach 0.1)
        effective_mach = max(self.mach, self.MACH_THRESHOLD_STATIC)
        effective_velocity = effective_mach * np.sqrt(self.gamma_inlet * self.mean_R_air * self.t_a)

        # Empuxo = (Eficiência * Potência) / Velocidade
        # Potência em kW (kJ/s), Velocidade em m/s -> Resultado em kN
        thrust_kN = (self.propeller_efficiency * self.pot_gear) / effective_velocity

        if in_kN:
            return thrust_kN
        else:
            return thrust_kN * 1000

    def get_thrust(self, in_kN: bool = True):
        # Soma o empuxo da hélice (já tratado para Mach 0) e o do bocal
        total_thrust_kN = self.get_nozzle_thrust() + self.get_propeler_thrust(in_kN=True)

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

    def get_emissions_flow(self) -> dict:
        """
        Calcula as vazões mássicas de emissões de CO2 e H2O.
        Lógica baseada na estequiometria do combustível.
        """
        total_fuel_flow = self.get_fuel_consumption()
        chi = self.hydrogen_fraction
        kerosene_flow = total_fuel_flow * (1 - chi)
        hydrogen_flow = total_fuel_flow * chi

        co2_flow = kerosene_flow * CO2_PER_KEROSENE_MASS
        h2o_from_kerosene = kerosene_flow * H2O_PER_KEROSENE_MASS
        h2o_from_hydrogen = hydrogen_flow * H2O_PER_HYDROGEN_MASS
        total_h2o_flow = h2o_from_kerosene + h2o_from_hydrogen

        return {'co2_flow_kgs': co2_flow, 'h2o_flow_kgs': total_h2o_flow}

    def print_config(self):
        """
        Imprime as características de configuração do motor de forma organizada.
        """
        print("\n--- Configuração do Motor Turboprop ---")
        max_key_len = max(len(key) for key in self.final_config.keys())

        # Categorias para melhor organização visual
        categories = {
            "Condições de Voo e Ambiente": ["mach", "altitude", "t_a", "p_a"],
            "Eficiências": [k for k in self.final_config if k.startswith("eta_")],
            "Gammas": [k for k in self.final_config if k.startswith("gamma_")],
            "Dados Operacionais": ["prc", "pr_tl", "hydrogen_fraction", "pressure_loss", "kerosene_PCI", "hydrogen_PCI",
                                   "mean_R_air", "Cp", "Cp_tl", "T04"],
            "Dados da Gearbox e Hélice": ["gearbox_efficiency", "propeller_efficiency", "max_gearbox_power",
                                          "ref_pot_th"]
        }

        for category, keys in categories.items():
            print(f"\n[{category}]")
            for key in keys:
                value = self.final_config.get(key)
                if isinstance(value, (int, float)):
                    print(f"{key:<{max_key_len}}: {value:.3f}")
                else:
                    print(f"{key:<{max_key_len}}: {value}")
        print("\n" + "-" * (max_key_len + 10))

    def print_outputs(self):
        """
        Imprime os resultados calculados do motor de forma organizada.
        Verifica se os componentes foram atualizados antes de imprimir.
        """
        if not hasattr(self, 'inlet'):
            print("\nAVISO: Os resultados não foram calculados. Execute 'update_turboprop_components()' primeiro.")
            return

        print("--- Resultados da Simulação do Motor ---")

        print("\n[ Estações do Motor ]")
        header = f"{'Estação':<20} | {'Temp. Total (K)':<20} | {'Pressão Total (kPa)':<20}"
        print(header)
        print("-" * len(header))
        print(f"{'2 (Inlet)':<20} | {self.t02:<20.3f} | {self.p02:<20.3f}")
        print(f"{'3 (Compressor)':<20} | {self.t03:<20.3f} | {self.p03:<20.3f}")
        print(f"{'4 (Câmara)':<20} | {self.t04:<20.3f} | {self.p04:<20.3f}")
        print(f"{'5 (Turbina Comp.)':<20} | {self.t05:<20.3f} | {self.p05:<20.3f}")
        print(f"{'6 (Turbina Livre)':<20} | {self.t06:<20.3f} | {self.p06:<20.3f}")

        print("\n[ Dados da Turbina Livre ]")
        print(f"{'Razão de expansão na tl':<35}: {self.pr_tl:.3f}")
        if hasattr(self, 'N2_ratio'):
            print(f"{'N2':<35}: {self.N2_ratio:.3f}")
        print(f"{'Pressão total na saída da tl':<35}: {self.p06:.2f} kPa")
        print(f"{'Temperatura Isentrópica Total':<35}: {self.total_iso_temperature:.2f} K")
        print(f"{'Temperatura Total':<35}: {self.t06:.2f} K")
        print(f"{'Cp_tl':<35}: {self.Cp_tl:.2f}")
        print(f"{'Trabalho Isentrópico':<35}: {self.iso_work:.2f} kJ/kg")
        print(f"{'Trabalho Real':<35}: {self.real_work:.2f} kJ/kg")
        print(f"{'Vazão na turbina':<35}: {self.turbine_airflow:.2f} kg/s")
        print(f"{'Potência na Turbina Livre':<35}: {self.pot_tl:.2f} kW")
        print(f"{'Potência na Gearbox':<35}: {self.pot_gear:.2f} kW")

        print("\n[ Empuxo ]")
        print(f"{'Empuxo específico do Bocal':<35}: {self.get_nozzle_specific_thrust():.3f} kN.s/kg")
        print(f"{'Empuxo do Bocal':<35}: {self.get_nozzle_thrust():.3f} kN")
        if self.u_flight > 0:
            print(f"{'Empuxo da Hélice':<35}: {self.get_propeler_thrust():.3f} kN")
            print(f"{'Empuxo Total':<35}: {self.get_thrust():.3f} kN")
            print(f"{'Consumo Esp. / empuxo (TSFC)':<35}: {self.get_tsfc():.5f} kg/(s*kN)")

        print("\n[ Performance Geral ]")
        print(f"{'Razão Combustível/Ar (f)':<35}: {self.fuel_to_air_ratio:.5f}")
        print(f"{'Velocidade Bocal Quente (u_core)':<32}: {self.u_core:.3f} m/s")
        print(f"{'Velocidade de Voo (u_0)':<32}: {self.u_flight:.3f} m/s")
        print(f"{'Consumo Esp. no eixo (BSFC)':<35}: {self.get_bsfc() * 10 ** 5:.2f} kg/(s*kW) * 10^-5")
        print(f"{'Consumo Esp. Equivalente (EBSFC)':<35}: {self.get_ebsfc() * 10 ** 5:.2f} kg/(s*kW) * 10^-5")
        print(f"{'Consumo de Combustível':<35}: {self.get_fuel_consumption():.3f} kg/s")
        print("\n" + "-" * 61)


if __name__ == "__main__":
    from utils.configs import config_ex71

    turboprop = Turboprop(config_ex71)
    turboprop.set_air_flow(8.49)
    turboprop.calibrate_pot_th()
    # turboprop.update_final_config(config_ex72)
    # turboprop.update_from_N2(0.85)
    turboprop.print_outputs()