class Inlet:
    """
    Representa a seção de entrada (inlet/difusor) de um motor.

    Calcula a temperatura e a pressão totais na saída do difusor com base nas condições de voo.
    """

    def __init__(
        self,
        t_a: float,
        p_a: float,
        mach: float,
        eta_d: float,
        gamma_d: float,
    ):
        """
        Args:
            t_a (float): Temperatura ambiente estática [K].
            p_a (float): Pressão ambiente estática [Pa].
            mach (float): Número de mach do voo.
            eta_d (float): Eficiência isentrópica do difusor (adimensional).
            gamma_d (float): Razão de calores específicos do fluido (adimensional, ~1.4 para o ar).
        """
        self.t_a = t_a
        self.p_a = p_a
        self.mach = mach
        self.eta_d = eta_d
        self.gamma_d = gamma_d

        # Calculando Temperatura e Pressão totais
        self._t02 = self.get_total_temperature()
        self._p02 = self.get_total_pressure()

    def get_total_temperature(self) -> float:
        """
        Calcula a temperatura total (de estagnação) na saída do difusor (t02).
        """
        term = 1 + ((self.gamma_d - 1) / 2) * (self.mach**2)
        t02 = self.t_a * term

        return t02

    def get_total_pressure(self) -> float:
        """
        Calcula a pressão total (de estagnação) na saída do difusor (p02).
        """
        # Usa o t02 já calculado no __init__
        t02 = self._t02

        # Garante que t_a não seja zero para evitar divisão por zero
        if self.t_a == 0:
            raise ValueError("Temperatura ambiente estática não pode ser zero.")

        ratio_temp = t02 / self.t_a
        exponent = self.gamma_d / (self.gamma_d - 1)
        base = 1 + self.eta_d * (ratio_temp - 1)

        p02 = self.p_a * (base**exponent)

        return p02
