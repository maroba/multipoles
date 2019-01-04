import sys, os
sys.path.insert(0, os.path.abspath('..'))
import unittest
import numpy as np
from multipoles.expansion import MultipoleExpansion


class TestMultipoleExpansion(unittest.TestCase):

    def test_gaussian_monopole_at_center(self):

        x, y, z = [np.linspace(-5, 5, 101)]*3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        sigma = 1.5
        rho = gaussian((X, Y, Z), (0, 0, 0), sigma)
        mpe = MultipoleExpansion(charge_dist={'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=2)

        self.assertAlmostEqual(1, mpe.total_charge, places=4)
        np.testing.assert_array_almost_equal(mpe.center_of_charge, (0, 0, 0))

        self.assertAlmostEqual(0.1, mpe.multipole_contribs((10, 0, 0))[0], places=4)

    def test_gaussian_monopole_at_off_center(self):

        x, y, z = [np.linspace(-5, 5, 101)]*3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        sigma = 1.5
        rho = gaussian((X, Y, Z), (1, 0, 0), sigma)
        mpe = MultipoleExpansion(charge_dist={'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=2)

        self.assertAlmostEqual(0.1, mpe.multipole_contribs((11, 0, 0))[0], places=4)

    def test_gaussian_dipole_at_center(self):

        x, y, z = [np.linspace(-5, 5, 101)]*3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        sigma = 1.5
        rho = gaussian((X, Y, Z), (1, 0, 0), sigma) - gaussian((X, Y, Z), (-1, 0, 0), sigma)
        mpe = MultipoleExpansion(charge_dist={'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=3)

        self.assertAlmostEqual(0, mpe.total_charge, places=4)

        # dipole term for point charges obtained from hand calculation: q1,-1 = -sqrt(2), q10 = 0
        #  ==> phi_1(10e_x) = 0.02, but octupole also contributes to total phi!

        phi_l = mpe.multipole_contribs((10, 0, 0))
        np.testing.assert_array_almost_equal((0, 0.02, 0, 0), phi_l, decimal=3)

        self.assertAlmostEqual(mpe.eval((10, 0, 0), 3), 1/9. - 1/11., places=3)
        self.assertAlmostEqual(mpe(10, 0, 0), 1 / 9. - 1 / 11., places=3)


    def test_dipole_of_point_charges(self):

        mpe = MultipoleExpansion({
            'discrete': True,
            'charges': [
                {'q': 1, 'xyz': (1, 0, 0)},
                {'q': -1, 'xyz': (-1, 0, 0)},
            ]
        }, l_max=3)

        np.testing.assert_array_almost_equal((0, 0.02, 0, 0), mpe.multipole_contribs((10, 0, 0)), decimal=3)



def gaussian(XYZ, xyz0, sigma):
    g = np.ones_like(XYZ[0])
    for k in range(3):
        g *= np.exp(-(XYZ[k]-xyz0[k])**2/sigma**2)
    g *= (sigma**2*np.pi)**-1.5
    return g


if __name__ == "__main__":
    unittest.main()