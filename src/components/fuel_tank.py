class FuelTank:
    """
    Modela um sistema de tanque de combustível de hidrogênio do tipo IV.

    Esta classe calcula as massas individuais de hidrogênio, querosene,
    o tanque de armazenamento, e o peso total do sistema de hidrogênio,
    com base na massa total de combustível e na fração de hidrogênio (chi).
    """
    # Relação de massa do sistema (tanque + H2) para a massa de H2,
    # derivado de 1 / 0.057 (5.7% wt) para tanques tipo IV[cite: 134, 135].
    SYSTEM_TO_H2_MASS_RATIO = 1 / 0.057  # Aproximadamente 17.54

    def __init__(self, total_fuel_mass: float, chi: float):
        """
        Inicializa o objeto FuelTank.

        Args:
            total_fuel_mass (float): A massa total da mistura de combustível (H2 + querosene) [kg].
            chi (float): A fração mássica de hidrogênio na mistura de combustível (valor de 0.0 a 1.0).
        """
        if not 0.0 <= chi <= 1.0:
            raise ValueError("A fração de hidrogênio 'chi' deve estar entre 0 e 1.")
        if total_fuel_mass < 0:
            raise ValueError("A massa total de combustível não pode ser negativa.")

        self.total_fuel_mass = total_fuel_mass
        self.chi = chi

        # --- Pré-cálculo dos componentes de massa para eficiência ---
        self._hydrogen_mass = self.chi * self.total_fuel_mass
        self._kerosene_mass = self.total_fuel_mass - self._hydrogen_mass

    def get_hydrogen_weight(self) -> float:
        """Retorna a massa apenas do hidrogênio armazenado."""
        return self._hydrogen_mass

    def get_kerosene_weight(self) -> float:
        """Retorna a massa apenas do querosene."""
        return self._kerosene_mass

    def get_tank_weight(self) -> float:
        """
        Retorna a massa apenas do tanque de hidrogênio vazio.

        Cálculo: m_tanque = m_sistema - m_H2
                      = (ratio * m_H2) - m_H2
                      = (ratio - 1) * m_H2
        """
        tank_to_h2_ratio = self.SYSTEM_TO_H2_MASS_RATIO - 1.0
        return tank_to_h2_ratio * self._hydrogen_mass

    def get_system_weight(self) -> float:
        """
        Retorna o peso total do sistema de hidrogênio (tanque + hidrogênio).

        Fórmula: m_sistema = (m_sistema / m_H2) * m_H2
        """
        return self.SYSTEM_TO_H2_MASS_RATIO * self._hydrogen_mass

