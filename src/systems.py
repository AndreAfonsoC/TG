from typing import Literal


class FuelSystem:
    """
    Gerencia a massa de combustível (hidrogênio e querosene) e o peso do sistema
    de armazenamento de H2 para uma missão, baseado em um tanque de compósito tipo IV
    ou outros tipos especificados.
    """

    # Relações de massa do sistema (tanque + H2) para massa de H2
    TANK_MASS_RATIOS = {
        "TYPE_I": 1 / 0.017,  # 58.82
        "TYPE_II": 1 / 0.021,  # 47.62
        "TYPE_III": 1 / 0.042,  # 23.81
        "TYPE_IV": 1 / 0.057,  # 17.54
    }

    def __init__(
            self,
            initial_total_fuel: float,
            chi_initial_mission: float,
            tank_type: Literal["TYPE_I", "TYPE_II", "TYPE_III", "TYPE_IV"] = "TYPE_IV",
    ):
        """
        Inicializa o sistema de combustível com as massas iniciais.

        Args:
            initial_total_fuel (float): Massa total de combustível (H2+QAV) no início da missão [kg].
            chi_initial_mission (float): Fração mássica inicial de hidrogênio (valor de 0.0 a 1.0).
            tank_type (str): Tipo de tanque de H2, conforme Tabela 1.2 da monografia. O padrão é 'TYPE_IV'.
        """
        if not 0.0 <= chi_initial_mission <= 1.0:
            raise ValueError(
                "A fração de hidrogênio 'chi_initial_mission' deve estar entre 0.0 e 1.0."
            )
        if initial_total_fuel < 0:
            raise ValueError("A massa inicial de combustível não pode ser negativa.")
        if tank_type not in self.TANK_MASS_RATIOS:
            raise ValueError(
                f"Tipo de tanque inválido. Escolha um entre: {list(self.TANK_MASS_RATIOS.keys())}"
            )

        self.chi_initial_mission = chi_initial_mission
        self.tank_type = tank_type
        self.system_to_h2_ratio = self.TANK_MASS_RATIOS[tank_type]

        # Massas iniciais (não mudam durante a simulação)
        self.hydrogen_mass_initial = initial_total_fuel * self.chi_initial_mission
        self.kerosene_mass_initial = initial_total_fuel * (1 - self.chi_initial_mission)

        # Massas restantes (serão atualizadas a cada fase)
        self.hydrogen_mass_remaining = self.hydrogen_mass_initial
        self.kerosene_mass_remaining = self.kerosene_mass_initial

    def get_tank_weight(self) -> float:
        """
        Calcula e retorna o peso do tanque de hidrogênio (vazio).
        O peso do tanque é fixo e baseado na quantidade inicial de H2.
        """
        tank_only_ratio = self.system_to_h2_ratio - 1.0
        return tank_only_ratio * self.hydrogen_mass_initial

    def get_total_weight_at_takeoff(self) -> float:
        """
        Retorna o peso total do sistema de combustível no início da missão.
        Isso inclui o querosene, o hidrogênio e o tanque de H2.
        """
        return (
                self.kerosene_mass_initial
                + self.hydrogen_mass_initial
                + self.get_tank_weight()
        )

    def consume_fuel(
            self,
            consumed_fuel_mass: float,
            chi: float,
            burn_strategy: Literal["proportional", "hydrogen_only", "kerosene_only"],
    ) -> None:
        """
        Atualiza as massas de combustível restantes com base na estratégia de queima da fase.

        Args:
            consumed_fuel_mass (float): Massa total de combustível consumida na fase [kg].
            burn_strategy (str): A estratégia de queima ('proportional', 'hydrogen_only', 'kerosene_only').

        Raises:
            ValueError: Se não houver combustível suficiente para a estratégia de queima solicitada.
        """
        if consumed_fuel_mass < 0:
            raise ValueError("Massa de combustível consumida não pode ser negativa.")

        if burn_strategy == "hydrogen_only":
            if self.hydrogen_mass_remaining < consumed_fuel_mass:
                raise ValueError(
                    f"Consumo de H2 ({consumed_fuel_mass:.2f} kg) excede o restante ({self.hydrogen_mass_remaining:.2f} kg)."
                )
            self.hydrogen_mass_remaining -= consumed_fuel_mass

        elif burn_strategy == "kerosene_only":
            if self.kerosene_mass_remaining < consumed_fuel_mass:
                raise ValueError(
                    f"Consumo de querosene ({consumed_fuel_mass:.2f} kg) excede o restante ({self.kerosene_mass_remaining:.2f} kg)."
                )
            self.kerosene_mass_remaining -= consumed_fuel_mass

        elif burn_strategy == "proportional":
            if (
                    chi == 0 and consumed_fuel_mass > 0
            ):  # Caso de queima só de querosene
                self.consume_fuel(consumed_fuel_mass, chi, "kerosene_only")
                return
            if (
                    chi == 1 and consumed_fuel_mass > 0
            ):  # Caso de queima só de H2
                self.consume_fuel(consumed_fuel_mass, chi, "hydrogen_only")
                return

            h2_to_consume = consumed_fuel_mass * chi
            kerosene_to_consume = consumed_fuel_mass * (1 - chi)

            if self.hydrogen_mass_remaining < h2_to_consume:
                raise ValueError(
                    f"Consumo proporcional de H2 ({h2_to_consume:.2f} kg) excede o restante ({self.hydrogen_mass_remaining:.2f} kg)."
                )
            if self.kerosene_mass_remaining < kerosene_to_consume:
                raise ValueError(
                    f"Consumo proporcional de querosene ({kerosene_to_consume:.2f} kg) excede o restante ({self.kerosene_mass_remaining:.2f} kg)."
                )

            self.hydrogen_mass_remaining -= h2_to_consume
            self.kerosene_mass_remaining -= kerosene_to_consume

    def get_remaining_hydrogen(self) -> float:
        """Retorna a massa de hidrogênio restante."""
        return self.hydrogen_mass_remaining

    def get_remaining_kerosene(self) -> float:
        """Retorna a massa de querosene restante."""
        return self.kerosene_mass_remaining

    def get_remaining_total_fuel(self) -> float:
        """Retorna a massa total de combustível (H2 + QAV) restante."""
        return self.hydrogen_mass_remaining + self.kerosene_mass_remaining
