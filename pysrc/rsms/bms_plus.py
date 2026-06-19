"""ms_plus"""

# pylint: disable=import-error,no-name-in-module
# pylint: disable=too-many-arguments,too-many-positional-arguments


import numpy as np
from scipy.interpolate import pade
from rsms.rsms import Bms


class BmsPlus:  # pylint: disable=too-few-public-methods
    """BmsPlus"""

    def __init__(self, mol1, mol2, potk, t_ref, u_min=-np.inf, tol=1e-3, num=20):
        taylor = Bms(mol1, mol2, potk).run(t_ref, u_min, tol, num)
        print(f"[BmsPlus.__init__] taylor_init = {taylor}")
        while True:
            if taylor[-1] != 0.0:
                break
            taylor.pop()
        print(f"[BmsPlus.__init__] taylor_last = {taylor}")
        self.__p, self.__q = pade(taylor, 1)
        self.__t_ref = t_ref

    def calc_b(self, t_kelvin):
        """calc_b"""
        p = self.__p(self.__t_ref / t_kelvin - 1.0)
        q = self.__q(self.__t_ref / t_kelvin - 1.0)
        return p / q
