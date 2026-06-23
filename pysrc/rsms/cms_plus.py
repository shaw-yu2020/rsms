"""cms_plus"""

# pylint: disable=import-error,no-name-in-module
# pylint: disable=too-many-arguments,too-many-positional-arguments


from scipy.interpolate import pade
from rsms.rsms import Cms


class CmsPlus:  # pylint: disable=too-few-public-methods
    """CmsPlus"""

    def __init__(
        self, mol1, mol2, mol3, potk12, potk13, potk23, t_ref, tol=9e-4, num=30
    ):
        taylor = Cms(mol1, mol2, mol3, potk12, potk13, potk23).run(t_ref, tol, num)
        print(f"[CmsPlus.__init__] taylor_init = {taylor}")
        while True:
            if taylor[-1] != 0.0:
                break
            taylor.pop()
        print(f"[CmsPlus.__init__] taylor_last = {taylor}")
        self.__p, self.__q = pade(taylor, 1)
        self.__t_ref = t_ref

    def calc_c(self, t_kelvin):
        """calc_c"""
        p = self.__p(self.__t_ref / t_kelvin - 1.0)
        q = self.__q(self.__t_ref / t_kelvin - 1.0)
        return p / q
