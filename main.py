"""main"""

# pylint: disable=import-error


import numpy as np
from scipy import integrate
from rsms import hello
from rsms import PotK
from rsms import Uij
from rsms import Bms

NA = 6.02214076e23  # Avogadro constant, 1/mol
K = 1.380649e-23  # Boltzmann constant, J/K
E = 1.602176634e-19  # elementary charge, C
EPSILON_0 = 8.8541878128e-12  # vacuum electric permittivity, F/m => C^2/J/m
E2_COEF = E * E / (4 * np.pi * EPSILON_0) * 1e10 / K  # K*A


def test_uij():
    """test_uij"""
    sigma = 2.0
    b_hs = 2.0 / 3.0 * np.pi * sigma**3
    lj = Uij(
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0]],
        [[PotK("R6R12", [-4 * sigma**6, 4 * sigma**12])]],
    )
    tb_ref = [
        [0.625, -5.7578],
        [0.75, -4.1759],
        [1.0, -2.5381],
        [1.2, -1.8359],
        [1.3, -1.5841],
        [1.4, -1.3758],
        [1.5, -1.2009],
        [2.0, -0.6276],
        [2.5, -0.3126],
        [5.0, 0.2433],
        [10.0, 0.4609],
    ]
    for tb in tb_ref:
        t_ref = tb[0]
        b_ref = tb[1]
        result = integrate.quad(
            lambda r: (
                np.exp(
                    -lj.ur_mean(r, 1e-6, 1, 1, 2)
                    / t_ref  # pylint: disable=cell-var-from-loop
                )
                - 1.0
            )
            * r
            * r,
            0.0,
            np.inf,
            full_output=True,
        )
        b_cal = round(-2 * np.pi * result[0] / b_hs, 4)
        assert b_cal == b_ref, f"Error in T*={t_ref}, b_ref={b_ref}, b_cal={b_cal}"


def test_bms():
    """test_bms"""
    lj6 = -((0.37122 * 10) ** 6) * 1e3 / NA / K  # (kJ/mol)**(1/6)*nm -> K*A**6
    lj12 = (0.3428 * 10) ** 12 * 1e3 / NA / K  # (kJ/mol)**(1/12)*nm -> K*A**12
    e = {"o": -0.8476, "h": 0.4238}
    spce = Bms(
        [
            [0.0, 0.0, 0.0],  # Oxygen
            [1.0, 0.0, 0.0],  # Hydrogen
            [-1.0 / 3.0, np.sqrt(8.0) / 3.0, 0.0],  # Hydrogen
        ],
        [
            [0.0, 0.0, 0.0],  # Oxygen
            [1.0, 0.0, 0.0],  # Hydrogen
            [-1.0 / 3.0, np.sqrt(8.0) / 3.0, 0.0],  # Hydrogen
        ],
        [
            [
                PotK("R1R6R12", [e["o"] * e["o"] * E2_COEF, lj6, lj12]),
                PotK("R1", [e["o"] * e["h"] * E2_COEF]),
                PotK("R1", [e["o"] * e["h"] * E2_COEF]),
            ],
            [
                PotK("R1", [e["h"] * e["o"] * E2_COEF]),
                PotK("R1", [e["h"] * e["h"] * E2_COEF]),
                PotK("R1", [e["h"] * e["h"] * E2_COEF]),
            ],
            [
                PotK("R1", [e["h"] * e["o"] * E2_COEF]),
                PotK("R1", [e["h"] * e["h"] * E2_COEF]),
                PotK("R1", [e["h"] * e["h"] * E2_COEF]),
            ],
        ],
        573.0,
    )
    tb_ref = [
        [373, -1856],
        [423, -907],
        [473, -529],
        [523, -345],
        [573, -242],
        [623, -179],
        [673, -137],
        [723, -108],
        [773, -87],
    ]
    for tb in tb_ref:
        t_ref = tb[0]
        b_ref = tb[1]
        b_cal = spce.calc_b(t_ref)
        print(
            f"[test_bms] T = {t_ref:.1f},",
            f"b_ref = {b_ref:.1f},",
            f"b_cal = {b_cal*(NA*1e-24):.2f}",
            f"diff = {100*(b_cal*(NA*1e-24)/b_ref-1):.2f}%",
        )


def main():
    """main"""
    print(hello())
    test_uij()
    test_bms()


if __name__ == "__main__":
    main()
