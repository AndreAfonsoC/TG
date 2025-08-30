class Turbine:
    """
    Representa a seção do elemento turbina, seja do fan, seja do compressor.
    """

    def __init__(
            self,
            t0_in: float,
            p0_in: float,
            t0_in_compressor: float,
            t0_out_compressor: float,
            eta_t: float,
            gamma_t: float,
            bpr: float = 0.0,
    ):
        """
        Inicializa o objeto Turbine.

        Args:
            t0_in (float): Temperatura total na entrada da turbina [K].
            p0_in (float): Pressão total na entrada da turbina [Pa].
            t0_in_compressor (float): Temperatura total na entrada do compressor dessa turbina [K].
            t0_out_compressor (float): Temperatura total na saída do compressor dessa turbina [K].
            bpr (float): Razão de passagem (bypass ratio) do fan (adimensional).
                Padrão para 0.0, ou seja, não se trata de um fan. Se bpr > 0 se trata de um fan.
            eta_t (float): Eficiência adiabática da turbina (adimensional).
            gamma_t (float): Razão de calores específicos dos gases quentes (adimensional, ~1.33).
        """
        self.t0_in = t0_in
        self.p0_in = p0_in
        self.t0_out_compressor = t0_out_compressor
        self.t0_in_compressor = t0_in_compressor
        self.bpr = bpr
        self.eta_t = eta_t
        self.gamma_t = gamma_t

        # Pré-calcula os valores de saída para eficiência
        self._t0_out = self.get_total_temperature()
        self._p0_out = self.get_total_pressure()

    def get_total_temperature(self) -> float:
        """
        Calcula a temperatura total na saída da turbina (t0_out).
        """
        t0_out = self.t0_in - (self.bpr + 1) * (
                self.t0_out_compressor - self.t0_in_compressor
        )
        return t0_out

    def get_total_pressure(self) -> float:
        """
        Calcula a pressão total na saída da turbina (p0_out).
        """
        # Garante que a eficiência e a temperatura não sejam zero para evitar divisões
        if self.eta_t == 0:
            raise ValueError("A eficiência da turbina (eta_t) não pode ser zero.")
        if self.t0_in == 0:
            raise ValueError(
                "A temperatura de entrada da turbina (t0_in) não pode ser zero."
            )

        # Usa o valor de t0_out pré-calculado
        t0_out = self._t0_out

        temp_ratio = t0_out / self.t0_in
        exponent = self.gamma_t / (self.gamma_t - 1)

        base = 1 - (1 / self.eta_t) * (1 - temp_ratio)

        p0_out = self.p0_in * (base ** exponent)

        return p0_out
