class CombustionChamber:
    def __init__(
            self,
            t0_in: float,
            p0_in: float,
            cp: float,
            t0_out_without_loss: float,
            eta: float = 1.0,
            gamma: float = 1.35,
            kerosene_pci: float = 45e3,
            hydrogen_pci: float = 120e3,
            chi: float = 0.0,
            pressure_loss: float = 0.0,
    ):
        """
        Inicializa a câmara de combustão com os parâmetros dados.

        Args:
            t0_in (float): Temperatura total de entrada (K).
            p0_in (float): Pressão total de entrada (Pa).
            cp (float): calor específico à pressão constante no combustor (kJ/(kg.K))
            t0_out_without_loss (float): Temperatura total de saída desprezando perda de pressão (K).
            eta (float): Eficiência da câmara de combustão.
            kerosene_pci (float, optional): Poder calorífico inferior do querosene (kJ/kg). Padrão para 45e3.
            hydrogen_pci (float, optional): Poder calorífico inferior do hidrogênio (kJ/kg). Padrão para 120e3.
            chi (float, optional): fração de hidrogênio na mistura de combustível (adimensional).
            pressure_loss (float, optional): Perda de pressão percentual na câmara de combustão (adimensional).
                                                    0.0 significa sem perdas. Padrão para 0.0.
        """
        self.t0_in = t0_in
        self.p0_in = p0_in
        self.eta = eta
        self.gamma = gamma
        self.cp = cp
        self.pressure_loss = pressure_loss
        self.p0_out = self.p0_in * (1 - self.pressure_loss)
        self.t0_out = t0_out_without_loss * (self.p0_out / self.p0_in) ** ((self.gamma - 1) / self.gamma)
        self.kerosene_pci = kerosene_pci
        self.hydrogen_pci = hydrogen_pci
        self.chi = chi

    def set_total_out_temperature(self, t0_out):
        """
        Define a temperatura total de saída da câmara de combustão.
        """
        self.t0_out = t0_out

    def set_hydrogen_fraction(self, chi):
        """
        Define a fração de hidrogênio na mistura de combustível.
        """
        self.chi = chi

    def get_total_pressure(self) -> float:
        """
        Retorna a pressão total na saída do câmara de combustão (P0_out).
        """
        return self.p0_out

    def get_total_temperature_out(self) -> float:
        """
        Returna a temperatura total na saída do câmara de combustão (t0_out).
        Já considera a perda de pressão.
        """
        return self.t0_out

    def get_fuel_to_air_ratio(self):
        """
        Calcula a razão combustível ar (m_dot fuel / m_dot air) na câmara de combustão
        """
        temp_ratio = self.t0_out / self.t0_in
        term = self.eta * (self.chi * self.hydrogen_pci + (1 - self.chi) * self.kerosene_pci)
        cp_t03 = self.cp * self.t0_in

        return (temp_ratio - 1) / (term / cp_t03 - temp_ratio)

    def get_energy_contribution(self):
        """
        Calcula a contribuição de energia de cada combustível (hidrogênio e querosene)
        com base na fração de massa de hidrogênio (chi).

        Returns:
            dict: Um dicionário contendo a energia por kg de combustível e a contribuição
                  percentual de cada componente.
        """
        # Energia liberada por cada combustível, por kg da mistura de combustível
        energy_from_hydrogen = self.chi * self.hydrogen_pci
        energy_from_kerosene = (1 - self.chi) * self.kerosene_pci

        total_energy = energy_from_hydrogen + energy_from_kerosene

        # Evita divisão por zero caso os poderes caloríficos sejam nulos
        if total_energy == 0.0:
            hydrogen_percentage = 0.0
            kerosene_percentage = 0.0
        else:
            hydrogen_percentage = (energy_from_hydrogen / total_energy) * 100
            kerosene_percentage = (energy_from_kerosene / total_energy) * 100

        return {
            'energia_hidrogenio_kJ_por_kg_combustivel': energy_from_hydrogen,
            'energia_querosene_kJ_por_kg_combustivel': energy_from_kerosene,
            'pci_resultante_kJ_por_kg_combustivel': total_energy,
            'percentual_energia_hidrogenio': hydrogen_percentage,
            'percentual_energia_querosene': kerosene_percentage,
        }
