"""main"""

# pylint: disable=import-error


import numpy as np
from scipy import integrate
from rsms import hello
from rsms import PotK
from rsms import Uij


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


def main():
    """main"""
    print(hello())
    test_uij()


if __name__ == "__main__":
    main()
