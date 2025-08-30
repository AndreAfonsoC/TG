class Compressor:
    """
    Representa a seção do compressor de um motor a jato.

    Calcula a temperatura e a pressão totais na saída do compressor
    com base nas condições de entrada e nos parâmetros de desempenho do compressor.
    """

    def __init__(
        self,
        t0_in: float,
        p0_in: float,
        pr: float,
        eta: float,
        gamma: float,
    ):
        """
        Inicializa o objeto Compressor.

        Args:
            t0_in (float): Temperatura total na entrada do compressor [K].
            p0_in (float): Pressão total na entrada do compressor [Pa].
            pr (float): Razão de aumento de pressão do compressor (adimensional).
            eta (float): Eficiência adiabática do compressor (adimensional).
            gamma (float): Razão de calores específicos do fluido (adimensional, ~1.4 para o ar).
        """
        self.t0_in = t0_in
        self.p0_in = p0_in
        self.pr = pr
        self.eta = eta
        self.gamma = gamma

        # Pré-calcula os valores de saída para eficiência
        self._T0_out = self.get_total_temperature()
        self._P0_out = self.get_total_pressure()

    def get_total_pressure(self) -> float:
        """
        Calcula a pressão total na saída do compressor (P0_out).
        """
        P0_out = self.p0_in * self.pr

        return P0_out

    def get_total_temperature(self) -> float:
        """
        Calcula a temperatura total na saída do compressor (T0_out)
        """
        # Garante que a eficiência não seja zero para evitar divisão por zero
        if self.eta == 0:
            raise ValueError("A eficiência do compressor (eta) не poate fi zero.")

        exponent = (self.gamma - 1) / self.gamma
        pressure_term = self.pr**exponent

        T0_out = self.t0_in * (1 + (1 / self.eta) * (pressure_term - 1))

        return T0_out
