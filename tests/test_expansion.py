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

    def test_call_respects_l_max(self):
        mpe = MultipoleExpansion(DIPOLE_Z, l_max=3)

        # total charge is zero, so the monopole contribution vanishes
        self.assertAlmostEqual(0, mpe(0, 0, 5, l_max=0))
        self.assertAlmostEqual(sum(mpe._multipole_contribs((0, 0, 5))[:2]), mpe(0, 0, 5, l_max=1))
        self.assertRaises(ValueError, lambda: mpe(0, 0, 5, l_max=4))

    def test_exterior_false_means_interior(self):
        mpe = MultipoleExpansion(DIPOLE_Z, l_max=2, exterior=False)
        self.assertTrue(mpe.interior)
        self.assertFalse(mpe.exterior)

        mpe = MultipoleExpansion(DIPOLE_Z, l_max=2, interior=False)
        self.assertTrue(mpe.exterior)
        self.assertFalse(mpe.interior)

    def test_getitem_matches_call_for_interior_expansion(self):
        X, Y, Z = grid(-6, 6, 25)
        rho = gaussian((X, Y, Z), (3, 3, 3), 1) - gaussian((X, Y, Z), (-3, -3, -3), 1)
        mpe = MultipoleExpansion({'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=3, interior=True)

        idx = (12, 13, 12)
        self.assertAlmostEqual(mpe(X[idx], Y[idx], Z[idx]), mpe[idx])
        npt.assert_allclose(mpe[:, :, 12], mpe(X[:, :, 12], Y[:, :, 12], Z[:, :, 12]))

    def test_getitem_does_not_modify_internal_coords(self):
        X, Y, Z = grid(-5, 5, 21)
        rho = gaussian((X, Y, Z), (0, 0, 0), 1.5)
        mpe = MultipoleExpansion({'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=2)
        R = mpe.internal_coords_spherical[0].copy()

        mpe[:, :, :]

        npt.assert_array_equal(R, mpe.internal_coords_spherical[0])

    def test_getitem_with_single_index(self):
        X, Y, Z = grid(-5, 5, 21)
        rho = gaussian((X, Y, Z), (0, 0, 0), 1.5)
        mpe = MultipoleExpansion({'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=2)

        self.assertEqual((21, 21), mpe[0].shape)

    def test_getitem_for_discrete_charges_raises(self):
        mpe = MultipoleExpansion(DIPOLE_Z, l_max=2)
        self.assertRaises(TypeError, lambda: mpe[0, 0, 0])

    def test_zero_total_charge_raises(self):
        self.assertRaises(InvalidChargeDistributionException, lambda: MultipoleExpansion(
            {'discrete': True, 'charges': [{'q': 0, 'xyz': (0, 0, 0)}]}, 1))

        X, Y, Z = grid(-1, 1, 5)
        self.assertRaises(InvalidChargeDistributionException, lambda: MultipoleExpansion(
            {'discrete': False, 'rho': np.zeros_like(X), 'xyz': (X, Y, Z)}, 1))

    def test_interior_expansion_with_charge_at_center_raises(self):
        self.assertRaises(InvalidExpansionException, lambda: MultipoleExpansion(
            {'discrete': True, 'charges': [{'q': 1, 'xyz': (0, 0, 0)}]}, 1, interior=True))

        X, Y, Z = grid(-5, 5, 21)
        rho = gaussian((X, Y, Z), (2, 0, 0), 1) + gaussian((X, Y, Z), (-2, 0, 0), 1)
        self.assertRaises(InvalidExpansionException, lambda: MultipoleExpansion(
            {'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, 1, interior=True))

    def test_xy_indexed_meshgrid_raises(self):
        x = np.linspace(-5, 5, 21)
        X, Y, Z = np.meshgrid(x, x, x)
        rho = gaussian((X, Y, Z), (0, 0, 0), 1.5)
        self.assertRaises(InvalidChargeDistributionException, lambda: MultipoleExpansion(
            {'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, 1))

    def test_mismatched_grid_shapes_raise(self):
        X, Y, Z = grid(-5, 5, 21)
        self.assertRaises(InvalidChargeDistributionException, lambda: MultipoleExpansion(
            {'discrete': False, 'rho': np.ones((5, 5, 5)), 'xyz': (X, Y, Z)}, 1))

    def test_descending_grid(self):
        x = np.linspace(-7, 7, 51)
        X, Y, Z = np.meshgrid(x[::-1], x, x, indexing='ij')
        rho = gaussian((X, Y, Z), (0, 0, 0), 1.5)
        mpe = MultipoleExpansion({'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=2)

        self.assertAlmostEqual(1, mpe.total_charge, places=4)
        self.assertAlmostEqual(erf(20 / 1.5) / 20, mpe(20, 0, 0), places=5)

    def test_eval_with_multidimensional_arrays(self):
        mpe = MultipoleExpansion(DIPOLE_Z, l_max=2)
        x = np.array([[1., 2.], [3., 4.]])
        z = 5 * np.ones((2, 2))

        actual = mpe(x, 0, z)

        self.assertEqual((2, 2), actual.shape)
        npt.assert_allclose([[mpe(1, 0, 5), mpe(2, 0, 5)], [mpe(3, 0, 5), mpe(4, 0, 5)]], actual)

    def test_eval_with_0d_arrays(self):
        mpe = MultipoleExpansion(DIPOLE_Z, l_max=2)
        self.assertAlmostEqual(mpe(1, 0, 5), mpe(np.array(1.), np.array(0.), np.array(5.)))

    def test_eval_with_mismatched_array_lengths_raises(self):
        mpe = MultipoleExpansion(DIPOLE_Z, l_max=2)
        self.assertRaises(ValueError, lambda: mpe([1, 2], [1, 2, 3], 0))


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

    def test_origin(self):
        npt.assert_array_equal((0, 0, np.pi / 2), cartesian_to_spherical(0, 0, 0))

        R, Phi, Theta = cartesian_to_spherical(np.zeros(2), np.zeros(2), np.array([0., 1.]))
        npt.assert_array_equal((0, 1), R)
        npt.assert_array_equal((0, 0), Phi)
        npt.assert_array_equal((np.pi / 2, 0), Theta)

    def test_does_not_change_numpy_error_settings(self):
        old = np.geterr()
        cartesian_to_spherical(np.zeros(2), np.zeros(2), np.zeros(2))
        self.assertEqual(old, np.geterr())


DIPOLE_Z = {'discrete': True, 'charges': [{'q': 1, 'xyz': (0, 0, 1)}, {'q': -1, 'xyz': (0, 0, -1)}]}


def grid(lo, hi, n):
    x = np.linspace(lo, hi, n)
    return np.meshgrid(x, x, x, indexing='ij')


def gaussian(XYZ, xyz0, sigma):
    g = np.ones_like(XYZ[0])
    for k in range(3):
        g *= np.exp(-(XYZ[k] - xyz0[k]) ** 2 / sigma ** 2)
    g *= (sigma ** 2 * np.pi) ** -1.5
    return g


if __name__ == "__main__":
    unittest.main()
