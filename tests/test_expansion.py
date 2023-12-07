import os
import sys

sys.path.insert(0, os.path.abspath('..'))
import unittest
import numpy as np
import numpy.testing as npt
from scipy.special import erf

from multipoles.expansion import (
    MultipoleExpansion, InvalidChargeDistributionException, InvalidExpansionException, cartesian_to_spherical
)


class TestMultipoleExpansion(unittest.TestCase):

    def test_gaussian_monopole_at_center(self):
        x, y, z = [np.linspace(-7, 7, 51)] * 3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        sigma = 1.5
        rho = gaussian((X, Y, Z), (0, 0, 0), sigma)
        mpe = MultipoleExpansion(charge_dist={'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=4)

        self.assertAlmostEqual(1, mpe.total_charge, places=4)
        np.testing.assert_array_almost_equal(mpe.center_of_charge, (0, 0, 0))

        self.assertAlmostEqual(0.1, mpe._multipole_contribs((10, 0, 0))[0], places=4)

        R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
        b = np.ones_like(R, dtype=bool)
        b[1:-1, 1:-1, 1:-1] = False

        r = 20
        self.assertAlmostEqual(
            erf(r / sigma) / r,
            mpe(r, 0, 0), places=5
        )
        self.assertAlmostEqual(
            erf(r / sigma) / r,
            mpe(0, r, 0), places=5
        )
        self.assertAlmostEqual(
            erf(r / sigma) / r,
            mpe(0, 0, r), places=5
        )

    def test_gaussian_monopole_at_off_center(self):
        x, y, z = [np.linspace(-5, 5, 51)] * 3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        sigma = 1.5
        rho = gaussian((X, Y, Z), (1, 0, 0), sigma)
        mpe = MultipoleExpansion(charge_dist={'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=8)

        def dist(xyz1, xyz2):
            return np.sqrt(sum((np.array(xyz1) - np.array(xyz2)) ** 2))

        P = (20, 0, 0)
        r = dist(P, (1, 0, 0))
        npt.assert_almost_equal(
            erf(r / sigma) / r,
            mpe(*P), decimal=5
        )

        P = (0, 20, 0)
        r = dist(P, (1, 0, 0))
        npt.assert_almost_equal(
            erf(r / sigma) / r,
            mpe(*P), decimal=5
        )

        P = (0, 0, 20)
        r = dist(P, (1, 0, 0))
        npt.assert_almost_equal(
            erf(r / sigma) / r,
            mpe(*P), decimal=5
        )

        P = (20, 20, 20)
        r = dist(P, (1, 0, 0))
        npt.assert_almost_equal(
            erf(r / sigma) / r,
            mpe(*P), decimal=5
        )

    def test_gaussian_dipole_at_center(self):
        x, y, z = [np.linspace(-5, 5, 51)] * 3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        sigma = 1.5
        rho = gaussian((X, Y, Z), (1, 0, 0), sigma) - gaussian((X, Y, Z), (-1, 0, 0), sigma)
        mpe = MultipoleExpansion(charge_dist={'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=3)

        self.assertAlmostEqual(0, mpe.total_charge, places=4)

        # dipole term for point charges obtained from hand calculation: q1,-1 = -sqrt(2), q10 = 0
        #  ==> phi_1(10e_x) = 0.02, but octupole also contributes to total phi!

        phi_l = mpe._multipole_contribs((10, 0, 0))
        np.testing.assert_array_almost_equal((0, 0.02, 0, 0), phi_l, decimal=3)

        self.assertAlmostEqual(mpe._eval((10, 0, 0), 3), 1 / 9. - 1 / 11., places=3)
        self.assertAlmostEqual(mpe(10, 0, 0), 1 / 9. - 1 / 11., places=3)

    def test_dipole_of_point_charges(self):
        mpe = MultipoleExpansion({
            'discrete': True,
            'charges': [
                {'q': 1, 'xyz': (1, 0, 0)},
                {'q': -1, 'xyz': (-1, 0, 0)},
            ]
        }, l_max=3)

        np.testing.assert_array_almost_equal((0, 0.02, 0, 0), mpe._multipole_contribs((10, 0, 0)), decimal=3)

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

        np.testing.assert_array_almost_equal(mpe_interior._multipole_contribs((0, 1, 1)),
                                             mpe_not_exterior._multipole_contribs((0, 1, 1)), decimal=10)

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

        np.testing.assert_array_almost_equal(mpe_exterior._multipole_contribs((10, 0, 0)),
                                             mpe_not_interior._multipole_contribs((10, 0, 0)), decimal=10)

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

        actual = mpe[:, :, :]
        self.assertEqual(3, actual.ndim)
        self.assertAlmostEqual(actual[0, 25, 25], 1 / 5., delta=5)

        actual = mpe[:, :, 0]
        self.assertEqual(2, actual.ndim)
        npt.assert_array_equal((51, 51), actual.shape)

    def test_gaussian_at_center_evaluate_with_1d_array(self):
        x, y, z = [np.linspace(-5, 5, 51)] * 3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        sigma = 1.5
        rho = gaussian((X, Y, Z), (0, 0, 0), sigma)
        mpe = MultipoleExpansion(charge_dist={'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=3)

        where_to_eval = np.linspace(100, 200, 10)
        expected = [mpe(x, 0, 0) for x in where_to_eval]

        # evaluate by three 1D arrays
        actual = mpe(where_to_eval, [0] * len(where_to_eval), [0] * len(where_to_eval))
        npt.assert_allclose(expected, actual)

        # evaluate by one 1D array, should automatically expand numbers to arrays
        actual = mpe(where_to_eval, 0, 0)
        npt.assert_allclose(expected, actual)

        # evaluate by two 1D array, should automatically expand number to arrays
        actual = mpe(where_to_eval, where_to_eval, 0)
        expected = [mpe(x, x, 0) for x in where_to_eval]
        npt.assert_allclose(expected, actual)

    def test_gaussian_dipole_at_center(self):
        x, y, z = [np.linspace(-5, 5, 51)] * 3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        sigma = 1.5
        rho = gaussian((X, Y, Z), (1, 0, 0), sigma) - gaussian((X, Y, Z), (-1, 0, 0), sigma)
        mpe = MultipoleExpansion(charge_dist={'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=3)

        self.assertAlmostEqual(0, mpe.total_charge, places=4)

        # dipole term for point charges obtained from hand calculation: q1,-1 = -sqrt(2), q10 = 0
        #  ==> phi_1(10e_x) = 0.02, but octupole also contributes to total phi!

        phi_l = mpe._multipole_contribs((10, 0, 0))
        np.testing.assert_array_almost_equal((0, 0.02, 0, 0), phi_l, decimal=3)

        self.assertAlmostEqual(mpe._eval((10, 0, 0), 3), 1 / 9. - 1 / 11., places=3)
        self.assertAlmostEqual(mpe(10, 0, 0), 1 / 9. - 1 / 11., places=3)

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

    def test_internal_coordinates_spherical(self):
        x, y, z = [np.linspace(-5, 5, 51)] * 3
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        sigma = 1.5
        rho = gaussian((X, Y, Z), (0, 0, 0), sigma)
        mpe = MultipoleExpansion(charge_dist={'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=3)

        R, Phi, Theta = mpe.internal_coords_spherical
        self.assertAlmostEqual(np.max(R), np.sqrt(3) * 5)
        self.assertAlmostEqual(np.min(Phi), -np.pi)
        self.assertAlmostEqual(np.max(Theta), np.pi)


class TestCartesianToSpherical(unittest.TestCase):

    def test_works(self):
        pi = np.pi

        actual = cartesian_to_spherical(1, 0, 0)
        expected = (1, 0, pi / 2)
        npt.assert_array_equal(expected, actual)

        actual = cartesian_to_spherical(0, 1, 0)
        expected = (1, pi / 2, pi / 2)
        npt.assert_array_equal(expected, actual)

        actual = cartesian_to_spherical(0, -1, 0)
        expected = (1, -pi / 2, pi / 2)
        npt.assert_array_equal(expected, actual)


def gaussian(XYZ, xyz0, sigma):
    g = np.ones_like(XYZ[0])
    for k in range(3):
        g *= np.exp(-(XYZ[k] - xyz0[k]) ** 2 / sigma ** 2)
    g *= (sigma ** 2 * np.pi) ** -1.5
    return g


if __name__ == "__main__":
    unittest.main()
