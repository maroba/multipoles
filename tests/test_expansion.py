import os
import sys

sys.path.insert(0, os.path.abspath('..'))
import unittest
import numpy as np
from multipoles.expansion import (
    MultipoleExpansion, InvalidChargeDistributionException, InvalidExpansionException
)


class TestMultipoleExpansion(unittest.TestCase):

    def test_gaussian_monopole_at_center(self):
        x, y, z = [np.linspace(-5, 5, 51)] * 3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        sigma = 1.5
        rho = gaussian((X, Y, Z), (0, 0, 0), sigma)
        mpe = MultipoleExpansion(charge_dist={'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=2)

        self.assertAlmostEqual(1, mpe.total_charge, places=4)
        np.testing.assert_array_almost_equal(mpe.center_of_charge, (0, 0, 0))

        self.assertAlmostEqual(0.1, mpe.multipole_contribs((10, 0, 0))[0], places=4)

    def test_gaussian_monopole_at_off_center(self):
        x, y, z = [np.linspace(-5, 5, 51)] * 3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        sigma = 1.5
        rho = gaussian((X, Y, Z), (1, 0, 0), sigma)
        mpe = MultipoleExpansion(charge_dist={'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=2)

        self.assertAlmostEqual(0.1, mpe.multipole_contribs((11, 0, 0))[0], places=4)

    def test_gaussian_dipole_at_center(self):
        x, y, z = [np.linspace(-5, 5, 51)] * 3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        sigma = 1.5
        rho = gaussian((X, Y, Z), (1, 0, 0), sigma) - gaussian((X, Y, Z), (-1, 0, 0), sigma)
        mpe = MultipoleExpansion(charge_dist={'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=3)

        self.assertAlmostEqual(0, mpe.total_charge, places=4)

        # dipole term for point charges obtained from hand calculation: q1,-1 = -sqrt(2), q10 = 0
        #  ==> phi_1(10e_x) = 0.02, but octupole also contributes to total phi!

        phi_l = mpe.multipole_contribs((10, 0, 0))
        np.testing.assert_array_almost_equal((0, 0.02, 0, 0), phi_l, decimal=3)

        self.assertAlmostEqual(mpe.eval((10, 0, 0), 3), 1 / 9. - 1 / 11., places=3)
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

    def test_surrounding_point_charges(self):
        mpe = MultipoleExpansion({
            'discrete': True,
            'charges': [
                {'q': 1, 'xyz': (0, 4, 0)},
                {'q': 1, 'xyz': (0, -4, 0)},
                {'q': 1, 'xyz': (4, 0, 0)},
                {'q': 1, 'xyz': (-4, 0, 0)},
            ]
        }, l_max=3, interior=True)

        # Should add to 1 at the center
        np.testing.assert_array_almost_equal(1, mpe(0, 0, 0), decimal=3)

    def test_balanced_guassian_dipole(self):
        x, y, z = [np.linspace(-6, 6, 51)] * 3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        sigma = 1
        rho = gaussian((X, Y, Z), (3, 3, 3), sigma) - gaussian((X, Y, Z), (-3, -3, -3), sigma)
        mpe = MultipoleExpansion(charge_dist={'discrete': False, 'rho': rho,
                                              'xyz': (X, Y, Z)}, l_max=3, interior=True)

        self.assertAlmostEqual(0, mpe.total_charge, places=4)

        # Should cancel out at the center
        np.testing.assert_array_almost_equal(0, mpe(0, 0, 0), decimal=3)

    def test_explicit_interior_expansion_call(self):
        charges = {
            'discrete': True,
            'charges': [
                {'q': 1, 'xyz': (10, 0, 0)},
                {'q': -1, 'xyz': (-10, 0, 0)},
            ]
        }
        mpe_interior = MultipoleExpansion(charges, l_max=3, interior=True)
        mpe_not_exterior = MultipoleExpansion(charges, l_max=3, exterior=False)

        np.testing.assert_array_almost_equal(mpe_interior.multipole_contribs((0, 1, 1)),
                                             mpe_not_exterior.multipole_contribs((0, 1, 1)), decimal=10)

    def test_explicit_exterior_expansion_call(self):
        charges = {
            'discrete': True,
            'charges': [
                {'q': 1, 'xyz': (1, 0, 0)},
                {'q': -1, 'xyz': (-1, 0, 0)},
            ]
        }
        mpe_exterior = MultipoleExpansion(charges, l_max=3, exterior=True)
        mpe_not_interior = MultipoleExpansion(charges, l_max=3, interior=False)

        np.testing.assert_array_almost_equal(mpe_exterior.multipole_contribs((10, 0, 0)),
                                             mpe_not_interior.multipole_contribs((10, 0, 0)), decimal=10)

    def test_expansion_type_is_exclusive(self):
        self.assertRaises(InvalidExpansionException, lambda:
        MultipoleExpansion({}, 2, interior=True, exterior=True))
        self.assertRaises(InvalidExpansionException, lambda:
        MultipoleExpansion({}, 2, interior=False, exterior=False))

    def test_explict_expansion_type(self):
        self.assertRaises(InvalidChargeDistributionException, lambda:
        MultipoleExpansion({}, 2, interior=False, exterior=True))
        self.assertRaises(InvalidChargeDistributionException, lambda:
        MultipoleExpansion({}, 2, interior=True, exterior=False))

    def test_charge_dist_without_discrete_should_raise(self):
        charge_dist = {
            'charges': [
                {'q': 1, 'xyz': (1, 0, 0)},
                {'q': -1, 'xyz': (-1, 0, 0)},
            ]
        }

        self.assertRaises(InvalidChargeDistributionException, lambda: MultipoleExpansion(charge_dist, 2))

    def test_charge_dist_without_charges_should_raise(self):
        charge_dist = {
            'discrete': True,
        }

        self.assertRaises(InvalidChargeDistributionException, lambda: MultipoleExpansion(charge_dist, 2))

    def test_discrete_charge_dist_with_rho_should_raise(self):
        charge_dist = {
            'discrete': True,
            'rho': None,
            'xyz': None
        }

        self.assertRaises(InvalidChargeDistributionException, lambda: MultipoleExpansion(charge_dist, 2))

    def test_charges_without_list_should_raise(self):
        charge_dist = {
            'discrete': False,
            'charges':
                {'q': 1, 'xyz': (1, 0, 0)},
        }

        self.assertRaises(InvalidChargeDistributionException, lambda: MultipoleExpansion(charge_dist, 2))

    def test_charges_without_q_should_raise(self):
        charge_dist = {
            'discrete': False,
            'charges': [
                {'qa': 1, 'xyz': (1, 0, 0)}],
        }

        self.assertRaises(InvalidChargeDistributionException, lambda: MultipoleExpansion(charge_dist, 2))

    def test_continuous_charge_dist_with_charges_should_raise(self):
        charge_dist = {
            'discrete': False,
            'charges': [
                {'q': 1, 'xyz': (1, 0, 0)},
                {'q': -1, 'xyz': (-1, 0, 0)},
            ]
        }

        self.assertRaises(InvalidChargeDistributionException, lambda: MultipoleExpansion(charge_dist, 2))

    def test_gaussian_at_center_evaluate_with_slices(self):
        x, y, z = [np.linspace(-5, 5, 51)] * 3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        sigma = 1.5
        rho = gaussian((X, Y, Z), (0, 0, 0), sigma)
        mpe = MultipoleExpansion(charge_dist={'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=3)

        actual = np.zeros_like(rho)
        actual[:, :, :] = mpe[:, :, :]

        self.assertEqual(3, actual.ndim)
        self.assertAlmostEqual(actual[0, 25, 25], 1 / 5., delta=5)

    def test_gaussian_at_center_evaluate_with_mask(self):
        x, y, z = [np.linspace(-5, 5, 51)] * 3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        sigma = 1.5
        rho = gaussian((X, Y, Z), (0, 0, 0), sigma)
        mpe = MultipoleExpansion(charge_dist={'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=3)

        mask = np.ones_like(rho, dtype=bool)
        mask[1:-1, 1:-1, 1:-1] = False
        actual = np.zeros_like(rho)
        actual[mask] = mpe[mask]

        self.assertEqual(3, actual.ndim)
        self.assertEqual(actual.shape, rho.shape)
        self.assertAlmostEqual(actual[0, 25, 25], 1 / 5., delta=5)
        self.assertTrue(np.all(actual[1:-1, 1:-1, 1:-1] == 0))
        self.assertFalse(np.all(actual[mask] == 0))

    def test_issue_7(self):
        # Phi(x,y,z) gives different answers if entire system is shifted #7
        charge_dist = {'discrete': True, 'charges': [{'q': 1, 'xyz': (0, 0, 0.5)}, {'q': -1, 'xyz': (0, 0, -1)}]}
        l_max = 3
        Phi = MultipoleExpansion(charge_dist, l_max)
        phi_1 = Phi(0, 0, 0)

        charge_dist = {'discrete': True, 'charges': [{'q': 1, 'xyz': (5, 5, 5.5)}, {'q': -1, 'xyz': (5, 5, 4)}]}
        l_max = 3
        Phi = MultipoleExpansion(charge_dist, l_max)
        phi_2 = Phi(5, 5, 5)

        self.assertAlmostEqual(phi_1, phi_2)


def gaussian(XYZ, xyz0, sigma):
    g = np.ones_like(XYZ[0])
    for k in range(3):
        g *= np.exp(-(XYZ[k] - xyz0[k]) ** 2 / sigma ** 2)
    g *= (sigma ** 2 * np.pi) ** -1.5
    return g


if __name__ == "__main__":
    unittest.main()
