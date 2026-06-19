"""uij_plus"""

# pylint: disable=import-error,no-name-in-module
# pylint: disable=too-many-arguments,too-many-positional-arguments


from rsms.rsms import Uij


class UijPlus:
    """UijPlus"""

    def __init__(self, moli, molj, potk):
        self.__uij = Uij(moli, molj, potk)

    def uij(self, r, phi1, theta1, phi2, theta2, psi2):
        """uij"""
        self.__uij.u12(r, phi1, theta1, phi2, theta2, psi2)

    def ur_mean(self, r, dev_tol=3e-5, num_inner=6**5, num_outer=6**5, num_print=2**5):
        """ur_mean"""
        return self.__uij.u_r(r, dev_tol, num_inner, num_outer, num_print)
