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

class PowerTurbine:
    def __init__(
            self,
            t0_in: float,
            p0_in: float,
            pr_tl: float,
            eta_tl: float,
            gamma_tl: float,
            cp_tl: float,
    ):
        """
        Inicializa o objeto PowerTurbine.

        Args:
            t0_in (float): Temperatura total na entrada da turbina [K].
            p0_in (float): Pressão total na entrada da turbina [kPa].
            pr_tl (float): Razão de expansão da turbina livre (adimensional).
            eta_tl (float): Eficiência adiabática da turbina (adimensional).
            gamma_tl (float): Razão de calores específicos dos gases quentes (adimensional, ~1.33).
        """
        self.t0_in = t0_in
        self.p0_in = p0_in
        self.pr_tl = pr_tl
        self.eta_tl = eta_tl
        self.gamma_tl = gamma_tl
        self.gamma_tl = gamma_tl
        self.cp_tl = cp_tl

        # Pré-calcula os valores de saída para eficiência
        self._t0_out = self.get_total_temperature()
        self._p0_out = self.get_total_pressure()

    def get_total_pressure(self) -> float:
        """
        Calcula a pressão total na saída da turbina (p0_out).
        """
        return self.p0_in / self.pr_tl

    def get_total_isentropic_temperature(self) -> float:
        """
        Calcula a temperatura total isentrópica na saída da turbina (t0_out_isentropic).
        """
        base = self.t0_in * (1 / self.pr_tl)
        exponent = (self.gamma_tl - 1) / self.gamma_tl
        return base ** exponent

    def get_isentropic_work(self) -> float:
        """
        Calcula o trabalho isentrópico por unidade de massa da turbina livre (w_isentropic).
        """
        t0_out_isentropic = self.get_total_isentropic_temperature()
        return self.cp_tl * (self.t0_in - t0_out_isentropic)

    def get_real_work(self) -> float:
        """
        Calcula o trabalho real por unidade de massa da turbina livre (w_tl).
        """
        w_isentropic = self.get_isentropic_work()
        return self.eta_tl * w_isentropic

    def get_total_temperature(self) -> float:
        """
        Calcula a temperatura total na saída da turbina (t0_out).
        """
        w_tl = self.get_real_work()
        return self.t0_in - (w_tl / self.cp_tl)

    def get_power(self, air_flow: float, fuel_to_air_ratio: float) -> float:
        """
        Calcula a potência fornecida pela turbina livre (pot_tl).

        Args:
            air_flow (float): Vazão mássica de ar através da turbina [kg/s].
            fuel_to_air_ratio (float): Razão combustível / ar da câmara de combustão  [adimensional].

        Returns:
            float: Potência fornecida pela turbina livre [W].
        """
        w_tl = self.get_real_work()
        return w_tl * air_flow * (1 + fuel_to_air_ratio)
