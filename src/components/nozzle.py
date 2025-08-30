import math

class Nozzle:
    """
    Representa um bocal de exaustão de um motor a jato.

    Esta classe genérica pode ser usada para calcular a velocidade de saída
    tanto para o bocal de gases quentes (core) quanto para o bocal do fan (bypass),
    dadas as condições de entrada apropriadas.
    """

    def __init__(
            self,
            t0_in: float,
            p0_in: float,
            p_a: float,
            eta_n: float,
            gamma_n: float,
            mean_r_air: float = 288.3,
    ):
        """
        Inicializa o objeto Nozzle.

        Args:
            t0_in (float): Temperatura total na entrada do bocal [K].
                              (Será T06 para o bocal quente, T08 para o bocal do fan).
            p0_in (float): Pressão total na entrada do bocal [Pa].
                              (Será P06 para o bocal quente, P08 para o bocal do fan).
            p_a (float): Pressão ambiente estática [Pa].
            eta_n (float): Eficiência adiabática do bocal (adimensional).
            gamma_n (float): Razão de calores específicos do fluido no bocal (adimensional).
            mean_r_air (float, optional): Constante específica do gás [J/(kg·K)]. Padrão para o ar.
        """
        self.t0_in = t0_in
        self.p0_in = p0_in
        self.p_a = p_a
        self.eta_n = eta_n
        self.gamma_n = gamma_n
        self.mean_r_air = mean_r_air

    def get_exhaust_velocity(self) -> float:
        """
        Calcula a velocidade de exaustão dos gases na saída do bocal.

        A fórmula genérica implementada é:
        u = sqrt( (2*eta*gamma*mean_r_air*t0_in)/(gamma-1) * [1 - (p_a/p0_in)^((gamma-1)/gamma)] )

        Returns:
            float: A velocidade de saída do bocal [m/s].
        """
        # --- Validações para evitar erros matemáticos ---
        if self.gamma_n == 1:
            raise ValueError("A razão de calores específicos (gamma_n) não pode ser 1.")
        if self.p0_in == 0:
            raise ValueError("A pressão total de entrada (p0_in) não pode ser zero.")
        if self.p_a / self.p0_in > 1:
            # Se a pressão ambiente for maior, não há expansão; a fórmula daria erro.
            return 0.0

        # --- Cálculo por partes para clareza ---
        # Termo 1: (2 * eta * gamma * mean_r_air * T0) / (gamma - 1)
        term1 = (2 * self.eta_n * self.gamma_n * self.mean_r_air * self.t0_in) / (self.gamma_n - 1)

        # Expoente: (gamma - 1) / gamma
        exponent = (self.gamma_n - 1) / self.gamma_n

        # Termo 2: [1 - (Pa / p0_in)^exponent]
        pressure_ratio_term = (self.p_a / self.p0_in) ** exponent
        term2 = 1 - pressure_ratio_term

        # Cálculo final
        velocity_squared = term1 * term2

        return math.sqrt(velocity_squared)


