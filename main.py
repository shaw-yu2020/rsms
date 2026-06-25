"""main"""

# pylint: disable=import-error


import math
from scipy import integrate
from rsms import hello
from rsms import PotK
from rsms import Uij
from rsms import Bms
from rsms import Cms

NA = 6.02214076e23  # Avogadro constant, 1/mol
K = 1.380649e-23  # Boltzmann constant, J/K
E = 1.602176634e-19  # elementary charge, C
EPSILON_0 = 8.8541878128e-12  # vacuum electric permittivity, F/m => C^2/J/m
E2_COEF = E * E / (4 * math.pi * EPSILON_0) * 1e10 / K  # K*A


LJ6 = -((0.37122 * 10) ** 6) * 1e3 / NA / K  # (kJ/mol)**(1/6)*nm -> K*A**6
LJ12 = (0.3428 * 10) ** 12 * 1e3 / NA / K  # (kJ/mol)**(1/12)*nm -> K*A**12
EO, EH = -0.8476, 0.4238
SPCE_XYZ = [
    [0.0, 0.0, 0.0],  # Oxygen
    [1.0, 0.0, 0.0],  # Hydrogen
    [-1.0 / 3.0, math.sqrt(8.0) / 3.0, 0.0],  # Hydrogen
]
SPCE_POT = [
    [
        PotK("R1R6R12", [EO * EO * E2_COEF, LJ6, LJ12]),
        PotK("R1", [EO * EH * E2_COEF]),
        PotK("R1", [EO * EH * E2_COEF]),
    ],
    [
        PotK("R1", [EH * EO * E2_COEF]),
        PotK("R1", [EH * EH * E2_COEF]),
        PotK("R1", [EH * EH * E2_COEF]),
    ],
    [
        PotK("R1", [EH * EO * E2_COEF]),
        PotK("R1", [EH * EH * E2_COEF]),
        PotK("R1", [EH * EH * E2_COEF]),
    ],
]


def test_uij():
    """test_uij"""
    sigma = 2.0
    b_hs = 2.0 / 3.0 * math.pi * sigma**3
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
                math.exp(
                    -lj.ur_mean(r, 1e-6, 1, 1, 2)
                    / t_ref  # pylint: disable=cell-var-from-loop
                )
                - 1.0
            )
            * r
            * r,
            0.0,
            math.inf,
            full_output=True,
        )
        b_cal = round(-2 * math.pi * result[0] / b_hs, 4)
        assert b_cal == b_ref, f"Error in T*={t_ref}, b_ref={b_ref}, b_cal={b_cal}"


def test_bms():
    """test_bms"""
    spce = Bms(SPCE_XYZ, SPCE_XYZ, SPCE_POT, 573.0)
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


def test_cms():
    """test_cms"""
    spce = Cms(SPCE_XYZ, SPCE_XYZ, SPCE_XYZ, SPCE_POT, SPCE_POT, SPCE_POT, 573.0)
    tb_ref = [
        [373, -10700000],
        [423, -1080000],
        [473, -140000],
        [523, -8660],
        [573, 11100],
        [623, 11900],
        [673, 9740],
        [723, 7520],
        [773, 5780],
    ]
    for tb in tb_ref:
        t_ref = tb[0]
        b_ref = tb[1]
        b_cal = spce.calc_c(t_ref)
        print(
            f"[test_bms] T = {t_ref:.1f},",
            f"b_ref = {b_ref:.1f},",
            f"b_cal = {b_cal*(NA*1e-24)**2:.1f}",
            f"diff = {b_cal*(NA*1e-24)**2-b_ref:.2f}%",
        )


def main():
    """main"""
    print(hello())
    test_uij()
    test_bms()
    test_cms()


if __name__ == "__main__":
    main()
