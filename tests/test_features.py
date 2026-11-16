"""Tests of the constructors, evaluation helpers, moments, transformations and I/O."""

import pickle
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from multipoles import (
    EPSILON_0,
    GRAVITATIONAL_CONSTANT,
    Evaluation,
    ExpansionValidityWarning,
    InvalidChargeDistributionException,
    InvalidExpansionException,
    MultipoleExpansion,
)
from multipoles._grid import simpson_weights, trapezoid_weights

# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------


def random_charges(n=8, seed=1, scale=1.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n), scale * rng.normal(size=(n, 3))


def coulomb_potential(q, xyz, points):
    points = np.atleast_2d(points)
    return np.array([np.sum(q / np.linalg.norm(p - xyz, axis=1)) for p in points])


def coulomb_field(q, xyz, points):
    points = np.atleast_2d(points)
    return np.array(
        [np.sum(q[:, None] * (p - xyz) / np.linalg.norm(p - xyz, axis=1)[:, None] ** 3, axis=0) for p in points]
    )


def points_on_sphere(n, radius, center, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return center + radius * v / np.linalg.norm(v, axis=1)[:, None]


def gaussian_density(axes, x0, sigma):
    X, Y, Z = np.meshgrid(*axes, indexing="ij", sparse=True)
    return np.exp(-((X - x0[0]) ** 2 + (Y - x0[1]) ** 2 + (Z - x0[2]) ** 2) / sigma**2) * (
        sigma**2 * np.pi
    ) ** -1.5


def finite_difference_hessian(f, p, h=1e-4):
    E = np.eye(3)
    return np.array(
        [
            [
                (f(*(p + h * a + h * b)) - f(*(p + h * a - h * b)) - f(*(p - h * a + h * b)) + f(*(p - h * a - h * b)))
                / (4 * h * h)
                for b in E
            ]
            for a in E
        ]
    )


# ----------------------------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------------------------


class TestFromCharges:
    def test_matches_dict_api(self):
        q, xyz = random_charges()
        a = MultipoleExpansion.from_charges(q, xyz, 6)
        b = MultipoleExpansion(
            {"discrete": True, "charges": [{"q": qi, "xyz": xi} for qi, xi in zip(q, xyz)]}, 6
        )
        for key, value in b.multipole_moments.items():
            assert a.multipole_moments[key] == pytest.approx(value, abs=1e-13)
        npt.assert_allclose(a.center, b.center)

    def test_scalar_charge_and_single_point(self):
        mpe = MultipoleExpansion.from_charges(2.0, (1, 2, 3), 0)
        assert mpe.total_charge == 2
        npt.assert_allclose(mpe.center, (1, 2, 3))

        mpe = MultipoleExpansion.from_charges(1.0, [(1, 0, 0), (-1, 0, 0)], 2)
        assert mpe.total_charge == 2

    @pytest.mark.parametrize(
        "q, xyz",
        [
            ([1, 2], [(0, 0, 0)] * 3),  # mismatched lengths
            ([1], [(0, 0)]),  # 2D positions
            ([1, np.nan], [(0, 0, 0), (1, 0, 0)]),  # not finite
            ([], np.zeros((0, 3))),  # empty
            ([0, 0], [(0, 0, 0), (1, 0, 0)]),  # zero absolute charge
        ],
    )
    def test_invalid_input_raises(self, q, xyz):
        with pytest.raises(InvalidChargeDistributionException):
            MultipoleExpansion.from_charges(q, xyz, 2)

    def test_empty_dict_charges_has_clear_message(self):
        with pytest.raises(InvalidChargeDistributionException, match="no charges"):
            MultipoleExpansion({"discrete": True, "charges": []}, 2)

    def test_repr(self):
        q, xyz = random_charges(3)
        assert repr(MultipoleExpansion.from_charges(q, xyz, 2, center=(0, 0, 0))) == (
            "MultipoleExpansion(3 point charges, l_max=2, exterior, center=[0.0, 0.0, 0.0])"
        )


class TestFromDensity:
    axes = (np.linspace(-6, 6, 41),) * 3

    def test_axes_meshgrid_and_sparse_meshgrid_agree(self):
        rho = gaussian_density(self.axes, (0.5, 0, -0.5), 1.2)
        rho = np.broadcast_to(rho, (41, 41, 41))
        a = MultipoleExpansion.from_density(rho, self.axes, 4)
        b = MultipoleExpansion.from_density(rho, np.meshgrid(*self.axes, indexing="ij"), 4)
        c = MultipoleExpansion.from_density(rho, np.meshgrid(*self.axes, indexing="ij", sparse=True), 4)
        d = MultipoleExpansion({"discrete": False, "rho": rho, "xyz": self.axes}, 4)
        for other in (b, c, d):
            npt.assert_allclose(a.multipole_moments_array, other.multipole_moments_array, atol=1e-14)
        assert a.total_charge == pytest.approx(1, abs=1e-6)

    def test_unevenly_spaced_grid(self):
        # finer in the middle, coarser outside
        x = np.concatenate([np.linspace(-10, -4, 13)[:-1], np.linspace(-4, 4, 81), np.linspace(4, 10, 13)[1:]])
        axes = (x, x, x)
        rho = gaussian_density(axes, (0.3, 0, 0), 1.0)
        mpe = MultipoleExpansion.from_density(rho, axes, 4)

        assert mpe.total_charge == pytest.approx(1, abs=1e-5)
        npt.assert_allclose(mpe.center_of_charge, (0.3, 0, 0), atol=1e-6)
        npt.assert_allclose(mpe.shifted((0, 0, 0)).dipole_moment, (0.3, 0, 0), atol=1e-6)
        assert mpe.dxyz is None and mpe.dvol is None
        P = (25.0, 3.0, -4.0)
        r = np.linalg.norm(np.array(P) - (0.3, 0, 0))
        assert mpe(*P) == pytest.approx(1 / r, rel=1e-5)

    def test_grid_properties(self):
        rho = gaussian_density(self.axes, (0, 0, 0), 1.2)
        mpe = MultipoleExpansion.from_density(rho, self.axes, 2)
        assert mpe.dxyz == pytest.approx((0.3, 0.3, 0.3))
        assert mpe.dvol == pytest.approx(0.027)
        for a, b in zip(mpe.grid_axes, self.axes):
            npt.assert_array_equal(a, b)
        assert mpe.external_coords[0].shape == (41, 41, 41)
        assert mpe.rho.shape == (41, 41, 41)

    @pytest.mark.parametrize(
        "rho, grid, match",
        [
            (np.ones((4, 4)), (np.arange(4.0),) * 3, "3D"),
            (np.ones((4, 4, 4)), (np.arange(5.0),) * 3, "do not match"),
            (np.ones((4, 4, 4)), (np.array([0, 1, 1, 2.0]),) + (np.arange(4.0),) * 2, "zero"),
            (np.ones((4, 4, 4)), (np.array([0, 2, 1, 3.0]),) + (np.arange(4.0),) * 2, "increasing"),
            (np.ones((4, 4, 4)), (np.arange(4.0),) * 2, "three"),
            (np.ones((4, 4, 4)) * 1j, (np.arange(4.0),) * 3, "real"),
            (np.ones((1, 4, 4)), (np.arange(1.0),) + (np.arange(4.0),) * 2, "at least 2"),
        ],
    )
    def test_invalid_grid_raises(self, rho, grid, match):
        with pytest.raises(InvalidChargeDistributionException, match=match):
            MultipoleExpansion.from_density(rho, grid, 2)

    def test_non_rectilinear_coordinates_raise(self):
        X, Y, Z = np.meshgrid(*self.axes, indexing="ij")
        X = X + 0.01 * Y
        with pytest.raises(InvalidChargeDistributionException, match="rectilinear"):
            MultipoleExpansion.from_density(np.ones(X.shape), (X, Y, Z), 2)

    def test_grid_methods_need_density(self):
        mpe = MultipoleExpansion.from_charges([1, -1], [(0, 0, 1), (0, 0, -1)], 2)
        for method in (lambda: mpe[0], mpe.field_on_grid, mpe.boundary_values, lambda: mpe.rho):
            with pytest.raises(TypeError):
                method()


class TestSimpsonWeights:
    @pytest.mark.parametrize("n", range(2, 12))
    def test_match_scipy(self, n):
        scipy_integrate = pytest.importorskip("scipy.integrate")
        if not hasattr(scipy_integrate, "simpson"):
            pytest.skip("SciPy too old")
        rng = np.random.default_rng(n)
        f = rng.normal(size=n)
        for x in (np.linspace(0, 3, n), np.cumsum(rng.random(n) + 0.1)):
            assert simpson_weights(x) @ f == pytest.approx(scipy_integrate.simpson(f, x=x), abs=1e-13)
            # descending axes give the same, positive weights
            assert simpson_weights(x[::-1]) @ f == pytest.approx(-scipy_integrate.simpson(f, x=x[::-1]), abs=1e-13)

    def test_exact_for_quadratics_on_uneven_grids(self):
        x = np.cumsum(np.random.default_rng(0).random(9) + 0.1)
        assert simpson_weights(x) @ x**2 == pytest.approx((x[-1] ** 3 - x[0] ** 3) / 3)
        assert trapezoid_weights(x) @ x == pytest.approx((x[-1] ** 2 - x[0] ** 2) / 2)


class TestLMax:
    q, xyz = [1, -1], [(0, 0, 1), (0, 0, -1)]

    def test_integral_float_is_accepted(self):
        mpe = MultipoleExpansion.from_charges(self.q, self.xyz, 2.0)
        assert mpe.l_max == 2 and isinstance(mpe.l_max, int)
        assert mpe(0, 0, 10) == pytest.approx(0.02, abs=1e-5)

    @pytest.mark.parametrize("l_max", [-1, 2.5, "two", None, True])
    def test_invalid_raises(self, l_max):
        with pytest.raises(ValueError):
            MultipoleExpansion.from_charges(self.q, self.xyz, l_max)

    @pytest.mark.parametrize("l_max", [-1, 2.5, 3])
    def test_invalid_in_call_raises(self, l_max):
        mpe = MultipoleExpansion.from_charges(self.q, self.xyz, 2)
        with pytest.raises(ValueError):
            mpe(0, 0, 10, l_max=l_max)

    def test_tol_without_auto_raises(self):
        with pytest.raises(ValueError, match="auto"):
            MultipoleExpansion.from_charges(self.q, self.xyz, 2, tol=1e-3)


# ----------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------


class TestPointsArray:
    q, xyz = random_charges()
    mpe = MultipoleExpansion.from_charges(q, xyz, 12)
    points = np.array([[9.0, -7, 8], [0, 0, 15], [12, 3, -4]])

    def test_potential(self):
        npt.assert_allclose(self.mpe(self.points), coulomb_potential(self.q, self.xyz, self.points), rtol=1e-9)
        npt.assert_allclose(self.mpe(self.points), self.mpe(*self.points.T))
        assert np.ndim(self.mpe(self.points[0])) == 0
        assert self.mpe(self.points.reshape(3, 1, 3)).shape == (3, 1)

    def test_field_and_hessian_have_components_last(self):
        E = self.mpe.field(self.points)
        assert E.shape == (3, 3)
        npt.assert_allclose(E, coulomb_field(self.q, self.xyz, self.points), rtol=1e-6)
        npt.assert_allclose(E, self.mpe.field(*self.points.T).T)
        assert self.mpe.hessian(self.points).shape == (3, 3, 3)
        npt.assert_allclose(self.mpe.hessian(self.points)[1], self.mpe.hessian(*self.points[1]))

    def test_is_valid_and_error_bound(self):
        assert self.mpe.is_valid(self.points).shape == (3,)
        assert self.mpe.error_bound(self.points).shape == (3,)

    @pytest.mark.parametrize("args", [(np.ones(4),), (np.ones((2, 2)),), (1, 2), (1, 2, 3, 4)])
    def test_wrong_shapes_raise(self, args):
        with pytest.raises((ValueError, TypeError)):
            self.mpe(*args)


class TestHessian:
    def test_exterior_matches_finite_differences(self):
        q, xyz = random_charges()
        mpe = MultipoleExpansion.from_charges(q, xyz, 10)
        for p in [np.array([9.0, -7, 8]), np.array([0, 0, 12.0])]:
            npt.assert_allclose(mpe.hessian(*p), finite_difference_hessian(mpe, p), rtol=1e-4, atol=1e-9)

    def test_interior_matches_finite_differences(self):
        q, xyz = random_charges(scale=10)
        mpe = MultipoleExpansion.from_charges(q, xyz, 25, interior=True, center=(0, 0, 0))
        p = np.array([0.3, -0.2, 0.4])
        npt.assert_allclose(mpe.hessian(*p), finite_difference_hessian(mpe, p), rtol=1e-4, atol=1e-9)

    def test_is_symmetric_and_traceless(self):
        q, xyz = random_charges()
        H = MultipoleExpansion.from_charges(q, xyz, 8).hessian(np.linspace(10, 20, 4), 3, 1)
        assert H.shape == (3, 3, 4)
        npt.assert_allclose(H, np.swapaxes(H, 0, 1))
        npt.assert_allclose(np.trace(H), 0, atol=1e-12)  # Laplace equation outside the charges

    def test_point_charge(self):
        mpe = MultipoleExpansion.from_charges(1, (0, 0, 0), 0)
        x = np.array([3.0, 0, 4])
        expected = (3 * np.outer(x, x) - 25 * np.eye(3)) / 5**5
        npt.assert_allclose(mpe.hessian(*x), expected, atol=1e-15)

    def test_low_order_interior_is_zero(self):
        mpe = MultipoleExpansion.from_charges(1, (5, 0, 0), 1, interior=True, center=(0, 0, 0))
        npt.assert_array_equal(mpe.hessian(0.1, 0, 0), np.zeros((3, 3)))


class TestEvaluate:
    def test_combines_potential_field_and_hessian(self):
        q, xyz = random_charges()
        mpe = MultipoleExpansion.from_charges(q, xyz, 8)
        x = np.linspace(10, 20, 5)

        result = mpe.evaluate(x, 2, -3, hessian=True)
        assert isinstance(result, Evaluation)
        npt.assert_allclose(result.potential, mpe(x, 2, -3))
        npt.assert_allclose(result.field, mpe.field(x, 2, -3))
        npt.assert_allclose(result.hessian, mpe.hessian(x, 2, -3))

        result = mpe.evaluate(x, 2, -3, field=False)
        assert result.field is None and result.hessian is None

    def test_respects_l_max(self):
        mpe = MultipoleExpansion.from_charges([1, -1], [(0, 0, 1), (0, 0, -1)], 4)
        result = mpe.evaluate(0, 0, 10, l_max=1)
        assert result.potential == pytest.approx(0.02)
        npt.assert_allclose(result.field, (0, 0, 0.004), atol=1e-15)


class TestContributions:
    def test_sum_is_potential(self):
        q, xyz = random_charges()
        mpe = MultipoleExpansion.from_charges(q, xyz, 6)
        x = np.linspace(10, 20, 5)
        contribs = mpe.contributions(x, 0, 3)
        assert contribs.shape == (7, 5)
        npt.assert_allclose(contribs.sum(axis=0), mpe(x, 0, 3))
        npt.assert_allclose(mpe.contributions(x, 0, 3, l_max=2), contribs[:3])

    def test_monopole(self):
        mpe = MultipoleExpansion.from_charges(2, (0, 0, 0), 3)
        npt.assert_allclose(mpe.contributions(0, 0, 4), (0.5, 0, 0, 0), atol=1e-16)


class TestGridEvaluation:
    axes = (np.linspace(-8, 8, 33),) * 3
    mpe = MultipoleExpansion.from_density(gaussian_density(axes, (0.5, 0, 0), 1.2), axes, 4)

    def test_field_on_grid(self):
        X, Y, Z = self.mpe.external_coords
        E = self.mpe.field_on_grid(np.s_[0, :, ::4])
        assert E.shape == (3, 33, 9)
        npt.assert_allclose(E, self.mpe.field(X[0, :, ::4], Y[0, :, ::4], Z[0, :, ::4]))

    def test_boundary_values(self):
        bc = self.mpe.boundary_values()
        assert bc.shape == (33, 33, 33)
        assert np.all(bc[1:-1, 1:-1, 1:-1] == 0)
        assert bc[0, 16, 16] == pytest.approx(self.mpe[0, 16, 16])
        assert bc[16, 16, -1] == pytest.approx(1 / np.hypot(8, 0.5), rel=1e-6)

        bc2 = self.mpe.boundary_values(width=2)
        npt.assert_allclose(bc2[1, 5, 5], self.mpe[1, 5, 5])
        assert bc2[2, 5, 5] == 0
        with pytest.raises(ValueError):
            self.mpe.boundary_values(width=0)

    def test_boundary_values_warn_for_small_grid(self):
        axes = (np.linspace(-3, 3, 21),) * 3
        mpe = MultipoleExpansion.from_density(gaussian_density(axes, (0, 0, 0), 1.2), axes, 2)
        with pytest.warns(ExpansionValidityWarning) as record:
            mpe.boundary_values()
        assert record[0].filename == __file__


# ----------------------------------------------------------------------------------------
# Units
# ----------------------------------------------------------------------------------------


class TestPrefactor:
    def test_named_prefactors(self):
        si = MultipoleExpansion.from_charges(1, (0, 0, 0), 2, prefactor="SI")
        assert si.prefactor == pytest.approx(8.987551792e9, rel=1e-9)
        assert si(2, 0, 0) == pytest.approx(1 / (4 * np.pi * EPSILON_0 * 2))

        gravity = MultipoleExpansion.from_charges(5.972e24, (0, 0, 0), 2, prefactor="gravity")
        r = 6.371e6
        assert gravity(r, 0, 0) == pytest.approx(-GRAVITATIONAL_CONSTANT * 5.972e24 / r)
        # the acceleration points to the center, about 9.8 m/s^2 on the surface
        g = gravity.field(r, 0, 0)
        assert g[0] == pytest.approx(-9.82, abs=0.01)

    def test_scales_everything(self):
        q, xyz = random_charges()
        a = MultipoleExpansion.from_charges(q, xyz, 6)
        b = MultipoleExpansion.from_charges(q, xyz, 6, prefactor=2.5)
        P = (10.0, -12, 8)
        assert b(*P) == pytest.approx(2.5 * a(*P))
        npt.assert_allclose(b.field(*P), 2.5 * a.field(*P))
        npt.assert_allclose(b.hessian(*P), 2.5 * a.hessian(*P))
        npt.assert_allclose(b.contributions(*P), 2.5 * a.contributions(*P))
        assert b.error_bound(*P) == pytest.approx(2.5 * a.error_bound(*P))
        assert b.energy([1], [P]) == pytest.approx(2.5 * a(*P))
        # the moments describe the distribution, they are not scaled
        npt.assert_array_equal(a.multipole_moments_array, b.multipole_moments_array)
        assert a.total_charge == b.total_charge

    @pytest.mark.parametrize("prefactor", ["cgs", 0, np.inf])
    def test_invalid_raises(self, prefactor):
        with pytest.raises(ValueError):
            MultipoleExpansion.from_charges(1, (0, 0, 0), 2, prefactor=prefactor)


# ----------------------------------------------------------------------------------------
# Accuracy control
# ----------------------------------------------------------------------------------------


class TestErrorBound:
    @pytest.mark.parametrize("l_max", [0, 2, 5, 10])
    def test_bounds_the_exterior_error(self, l_max):
        q, xyz = random_charges(10, seed=3)
        mpe = MultipoleExpansion.from_charges(q, xyz, 10)
        rng = np.random.default_rng(l_max)
        radii = rng.uniform(1.01, 5, 300) * mpe.validity_radius
        points = mpe.center + radii[:, None] * points_on_sphere(300, 1, 0)
        error = np.abs(mpe(points, l_max=l_max) - coulomb_potential(q, xyz, points))
        bound = mpe.error_bound(points, l_max=l_max)
        assert np.all(error <= bound)
        assert np.median(bound / error) < 100  # and is not too pessimistic

    def test_bounds_the_interior_error(self):
        q, xyz = random_charges(10, seed=4, scale=10)
        xyz[np.linalg.norm(xyz, axis=1) < 5] *= 3
        mpe = MultipoleExpansion.from_charges(q, xyz, 8, interior=True, center=(0, 0, 0))
        radii = np.random.default_rng(0).uniform(0.05, 0.99, 300) * mpe.validity_radius
        points = radii[:, None] * points_on_sphere(300, 1, 0)
        error = np.abs(mpe(points) - coulomb_potential(q, xyz, points))
        assert np.all(error <= mpe.error_bound(points))

    def test_infinite_where_invalid(self):
        mpe = MultipoleExpansion.from_charges([1, -1], [(0, 0, 1), (0, 0, -1)], 4)
        npt.assert_array_equal(mpe.error_bound([0, 0], 0, [0.5, 2])[0], np.inf)
        assert np.isfinite(mpe.error_bound(0, 0, 2))

    def test_continuous(self):
        axes = (np.linspace(-8, 8, 41),) * 3
        mpe = MultipoleExpansion.from_density(gaussian_density(axes, (1, 0, 0), 1.0), axes, 3)
        exact = 1 / np.linalg.norm(np.array([9.0, 5, 3]) - (1, 0, 0))  # outside the gaussian
        assert abs(mpe(9, 5, 3) - exact) <= mpe.error_bound(9, 5, 3)

    def test_from_moments_needs_abs_charge(self):
        mpe = MultipoleExpansion.from_moments({(0, 0): 1})
        with pytest.raises(ValueError):
            mpe.error_bound(1, 0, 0)
        mpe = MultipoleExpansion.from_moments({(0, 0): 1, (1, 0): 0}, validity_radius=1, abs_charge=1)
        assert mpe.error_bound(2, 0, 0) == pytest.approx(0.25)


class TestAutoLMax:
    def test_meets_tolerance(self):
        q, xyz = random_charges(10, seed=5)
        ref = MultipoleExpansion.from_charges(q, xyz, 0)
        radius = 3 * ref.validity_radius
        for tol in (1e-3, 1e-6, 1e-9):
            mpe = MultipoleExpansion.from_charges(q, xyz, "auto", tol=tol, radius=radius)
            points = points_on_sphere(200, radius, mpe.center)
            error = np.abs(mpe(points) - coulomb_potential(q, xyz, points))
            assert np.max(error) <= tol * np.sum(np.abs(q)) / radius
            # and one order less would not satisfy the bound
            if mpe.l_max > 0:
                bound = mpe.error_bound(points[:1], l_max=mpe.l_max - 1)[0]
                assert bound > tol * np.sum(np.abs(q)) / radius

    def test_density_defaults_to_grid_boundary(self):
        axes = (np.linspace(-8, 8, 41),) * 3
        rho = gaussian_density(axes, (1, 0, 0), 1.0) - gaussian_density(axes, (-1, 0, 0), 1.0)
        mpe = MultipoleExpansion.from_density(rho, axes, "auto", tol=1e-4)
        assert 1 <= mpe.l_max <= 20
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExpansionValidityWarning)
            mpe.boundary_values()

    def test_needs_radius(self):
        with pytest.raises(ValueError, match="radius"):
            MultipoleExpansion.from_charges([1, -1], [(0, 0, 1), (0, 0, -1)], "auto")

    def test_radius_inside_charges_raises(self):
        with pytest.raises(ValueError, match="does not converge"):
            MultipoleExpansion.from_charges([1, -1], [(0, 0, 1), (0, 0, -1)], "auto", radius=0.5)

    def test_unreachable_tolerance_raises(self):
        with pytest.raises(ValueError, match="l_max >"):
            MultipoleExpansion.from_charges([1, -1], [(0, 0, 1), (0, 0, -1)], "auto", radius=1.001, tol=1e-12)

    def test_interior(self):
        q, xyz = random_charges(10, seed=6, scale=10)
        xyz[np.linalg.norm(xyz, axis=1) < 5] *= 3
        mpe = MultipoleExpansion.from_charges(q, xyz, "auto", interior=True, center=(0, 0, 0), radius=2, tol=1e-8)
        points = points_on_sphere(100, 2, 0)
        scale = np.sum(np.abs(q) / np.linalg.norm(xyz, axis=1))
        assert np.max(np.abs(mpe(points) - coulomb_potential(q, xyz, points))) <= 1e-8 * scale


# ----------------------------------------------------------------------------------------
# Moments
# ----------------------------------------------------------------------------------------


def traceless_power(v, l):
    """(2l-1)!! times the traceless part of the l-fold outer product of v, for l <= 3."""
    d = np.eye(3)
    r2 = v @ v
    if l == 0:
        return np.array(1.0)
    if l == 1:
        return v
    if l == 2:
        return 3 * np.outer(v, v) - r2 * d
    return 15 * np.einsum("i,j,k->ijk", v, v, v) - 3 * r2 * (
        np.einsum("i,jk->ijk", v, d) + np.einsum("j,ik->ijk", v, d) + np.einsum("k,ij->ijk", v, d)
    )


class TestCartesianMoment:
    q, xyz = random_charges()
    mpe = MultipoleExpansion.from_charges(q, xyz, 4)

    @pytest.mark.parametrize("l", [0, 1, 2, 3])
    def test_matches_definition(self, l):
        x = self.xyz - self.mpe.center
        expected = sum(qi * traceless_power(xi, l) for qi, xi in zip(self.q, x))
        npt.assert_allclose(self.mpe.cartesian_moment(l), expected, atol=1e-12)

    def test_reproduces_contribution(self):
        x = np.array([7.0, -3, 5]) - self.mpe.center
        r = np.linalg.norm(x)
        M = self.mpe.cartesian_moment(4)
        assert M.shape == (3, 3, 3, 3)
        phi_4 = np.einsum("ijkl,i,j,k,l", M, x, x, x, x) / (24 * r**9)
        assert phi_4 == pytest.approx(self.mpe.contributions(*(x + self.mpe.center))[4], rel=1e-10)
        npt.assert_allclose(np.einsum("iijk", M), 0, atol=1e-12)  # traceless

    def test_interior_are_derivatives_at_center(self):
        q, xyz = random_charges(scale=10)
        mpe = MultipoleExpansion.from_charges(q, xyz, 6, interior=True, center=(0, 0, 0))
        npt.assert_allclose(mpe.cartesian_moment(1), -mpe.field(0, 0, 0), atol=1e-14)
        npt.assert_allclose(mpe.cartesian_moment(2), mpe.hessian(0, 0, 0), atol=1e-14)

    def test_order_above_l_max_raises(self):
        with pytest.raises(ValueError):
            self.mpe.cartesian_moment(5)


class TestMomentConventions:
    q, xyz = random_charges()
    mpe = MultipoleExpansion.from_charges(q, xyz, 5)

    def test_orthonormal_matches_scipy(self):
        special = pytest.importorskip("scipy.special")
        if hasattr(special, "sph_harm_y"):
            Y = special.sph_harm_y
        else:  # SciPy < 1.15
            def Y(l, m, theta, phi):
                return special.sph_harm(m, l, phi, theta)

        from multipoles.expansion import cartesian_to_spherical

        r, phi, theta = cartesian_to_spherical(*(self.xyz - self.mpe.center).T)
        moments = self.mpe.moments("orthonormal")
        for l in range(6):
            for m in range(-l, l + 1):
                expected = np.sum(self.q * r**l * np.conj(Y(l, m, theta, phi)))
                assert moments[l, m] == pytest.approx(expected, abs=1e-10)

    def test_real_moments(self):
        real = self.mpe.moments(real=True)
        assert real.dtype == float
        npt.assert_allclose([real[1, 1], real[1, -1], real[1, 0]], self.mpe.dipole_moment, atol=1e-13)

        # the real harmonics reproduce the potential
        x = np.array([6.0, -4, 3]) - self.mpe.center
        complex_moments = self.mpe.multipole_moments_array
        rebuilt = np.zeros_like(complex_moments)
        for l in range(6):
            rebuilt[l, 0] = real[l, 0]
            for m in range(1, l + 1):
                rebuilt[l, m] = (-1) ** m * (real[l, m] - 1j * real[l, -m]) / np.sqrt(2)
                rebuilt[l, -m] = (-1) ** m * np.conj(rebuilt[l, m])
        npt.assert_allclose(rebuilt, complex_moments, atol=1e-12)
        assert np.linalg.norm(x) > 0

    def test_invalid_normalization_raises(self):
        with pytest.raises(ValueError):
            self.mpe.moments("schmidt")

    def test_power_spectrum_is_rotation_invariant(self):
        theta = 0.7
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        rotation = rotation @ np.array([[1, 0, 0], [0, np.cos(1.1), -np.sin(1.1)], [0, np.sin(1.1), np.cos(1.1)]])
        center = self.mpe.center
        rotated = MultipoleExpansion.from_charges(self.q, (self.xyz - center) @ rotation.T + center, 5, center=center)
        npt.assert_allclose(rotated.power_spectrum(), self.mpe.power_spectrum(), rtol=1e-10)
        assert self.mpe.power_spectrum()[0] == pytest.approx(abs(np.sum(self.q)))


# ----------------------------------------------------------------------------------------
# Transformations
# ----------------------------------------------------------------------------------------


class TestShifted:
    q, xyz = random_charges()

    def test_with_charges_recomputes(self):
        mpe = MultipoleExpansion.from_charges(self.q, self.xyz, 8)
        shifted = mpe.shifted((1, 2, 3))
        npt.assert_array_equal(shifted.center, (1, 2, 3))
        npt.assert_allclose(
            shifted.multipole_moments_array,
            MultipoleExpansion.from_charges(self.q, self.xyz, 8, center=(1, 2, 3)).multipole_moments_array,
        )

    def test_exterior_translation_of_moments_is_exact(self):
        mpe = MultipoleExpansion.from_charges(self.q, self.xyz, 10)
        new_center = mpe.center + (0.5, -0.3, 0.2)
        translated = MultipoleExpansion.from_moments(
            mpe.multipole_moments, mpe.center, validity_radius=mpe.validity_radius, abs_charge=1
        ).shifted(new_center)
        direct = MultipoleExpansion.from_charges(self.q, self.xyz, 10, center=new_center)
        scale = np.abs(direct.multipole_moments_array).max(axis=1, keepdims=True)
        npt.assert_allclose(translated.multipole_moments_array / scale, direct.multipole_moments_array / scale,
                            atol=1e-12)
        assert translated.validity_radius == pytest.approx(mpe.validity_radius + np.sqrt(0.38))
        assert translated._profile[0][0] == pytest.approx(mpe.validity_radius + np.sqrt(0.38))

    def test_interior_translation_keeps_polynomial(self):
        q, xyz = random_charges(scale=10)
        xyz[np.linalg.norm(xyz, axis=1) < 5] *= 3
        mpe = MultipoleExpansion.from_charges(q, xyz, 12, interior=True, center=(0, 0, 0))
        moments_only = MultipoleExpansion.from_moments(
            mpe.multipole_moments, interior=True, validity_radius=mpe.validity_radius
        )
        shifted = moments_only.shifted((0.2, 0.1, -0.1))
        p = np.array([0.3, 0.1, 0.05])
        assert shifted(*p) == pytest.approx(mpe(*p), rel=1e-12)
        npt.assert_allclose(shifted.field(*p), mpe.field(*p), rtol=1e-10)
        assert shifted.validity_radius == pytest.approx(mpe.validity_radius - np.sqrt(0.06))

        with pytest.raises(InvalidExpansionException):
            moments_only.shifted((mpe.validity_radius, 0, 0))


class TestArithmetic:
    q, xyz = random_charges()
    a = MultipoleExpansion.from_charges(q[:4], xyz[:4], 14)
    b = MultipoleExpansion.from_charges(q[4:], xyz[4:] + (3, 0, 0), 12)
    point = np.array([20.0, 10, -15])

    def exact(self, point):
        return coulomb_potential(self.q, np.concatenate([self.xyz[:4], self.xyz[4:] + (3, 0, 0)]), point)[0]

    def test_add_with_charges(self):
        total = self.a + self.b
        assert total.l_max == 12
        assert total(*self.point) == pytest.approx(self.exact(self.point), rel=1e-10)
        assert total.total_charge == pytest.approx(np.sum(self.q))
        assert "combined distribution of 8 points" in repr(total)

    def test_add_with_moments(self):
        a = MultipoleExpansion.from_moments(self.a.multipole_moments, self.a.center)
        b = MultipoleExpansion.from_moments(self.b.multipole_moments, self.b.center)
        total = a + b
        npt.assert_array_equal(total.center, self.a.center)
        assert total(*self.point) == pytest.approx(self.exact(self.point), rel=1e-10)
        assert total.total_charge == pytest.approx(np.sum(self.q))

    def test_add_keeps_explicit_center(self):
        a = MultipoleExpansion.from_charges(self.q[:4], self.xyz[:4], 6, center=(1, 1, 1))
        npt.assert_array_equal((a + self.b).center, (1, 1, 1))

    def test_scale_and_subtract(self):
        P = self.point
        assert (2 * self.a)(*P) == pytest.approx(2 * self.a(*P))
        assert (self.a * 2)(*P) == pytest.approx(2 * self.a(*P))
        assert (self.a / 4)(*P) == pytest.approx(self.a(*P) / 4)
        assert (-self.a)(*P) == pytest.approx(-self.a(*P))
        assert (self.a - self.a)(*P) == pytest.approx(0, abs=1e-15)
        assert (-self.a).total_charge == pytest.approx(-self.a.total_charge)
        npt.assert_allclose((-self.a).dipole_moment, -self.a.dipole_moment)

    def test_scaled_density(self):
        axes = (np.linspace(-6, 6, 25),) * 3
        mpe = MultipoleExpansion.from_density(gaussian_density(axes, (0, 0, 0), 1), axes, 2)
        doubled = 2 * mpe
        npt.assert_allclose(doubled[0], 2 * mpe[0])
        npt.assert_allclose(doubled.rho, 2 * mpe.rho)

    def test_incompatible_raises(self):
        interior = MultipoleExpansion.from_charges(1, (5, 0, 0), 2, interior=True, center=(0, 0, 0))
        with pytest.raises(InvalidExpansionException):
            self.a + interior
        with pytest.raises(InvalidExpansionException):
            self.a + MultipoleExpansion.from_charges(1, (0, 0, 0), 2, prefactor="SI")
        with pytest.raises(TypeError):
            self.a + 1


# ----------------------------------------------------------------------------------------
# Moments in, moments out
# ----------------------------------------------------------------------------------------


class TestFromMoments:
    def test_dict_and_array(self):
        moments = {(0, 0): 1, (1, 0): 2}
        mpe = MultipoleExpansion.from_moments(moments)
        assert mpe.l_max == 1
        assert mpe(0, 0, 10) == pytest.approx(0.1 + 0.02)
        assert mpe.total_charge == 1
        npt.assert_allclose(mpe.dipole_moment, (0, 0, 2))
        assert mpe.center_of_charge is None
        assert repr(mpe) == "MultipoleExpansion(from moments, l_max=1, exterior, center=[0.0, 0.0, 0.0])"

        q, xyz = random_charges()
        ref = MultipoleExpansion.from_charges(q, xyz, 6)
        mpe = MultipoleExpansion.from_moments(ref.multipole_moments_array, ref.center)
        P = (10.0, -3, 7)
        assert mpe(*P) == pytest.approx(ref(*P), rel=1e-13)
        npt.assert_allclose(mpe.field(*P), ref.field(*P), rtol=1e-12)
        npt.assert_allclose(mpe.quadrupole_moment, ref.quadrupole_moment, atol=1e-12)

    def test_interior(self):
        mpe = MultipoleExpansion.from_moments({(0, 0): 2, (1, 1): -1, (1, -1): 1}, interior=True)
        assert mpe.total_charge is None
        assert mpe.validity_radius == np.inf
        assert mpe(0, 0, 0) == pytest.approx(2)

    @pytest.mark.parametrize(
        "moments",
        [{}, {(1, 2): 1}, {(0, 0): 1, (1, 1): 1, (1, -1): 1}, np.ones((2, 2))],
    )
    def test_invalid_raises(self, moments):
        with pytest.raises(ValueError):
            MultipoleExpansion.from_moments(moments)

    def test_without_charges_some_operations_fail(self):
        mpe = MultipoleExpansion.from_moments({(0, 0): 1})
        with pytest.raises(TypeError):
            mpe[0]


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        q, xyz = random_charges()
        mpe = MultipoleExpansion.from_charges(q, xyz, 8, prefactor="SI")
        path = tmp_path / "expansion.npz"
        mpe.save(path)
        loaded = MultipoleExpansion.load(path)

        P = (10.0, -3, 7)
        assert loaded(*P) == mpe(*P)
        npt.assert_array_equal(loaded.field(*P), mpe.field(*P))
        assert loaded.error_bound(*P) == mpe.error_bound(*P)
        assert loaded.validity_radius == mpe.validity_radius
        assert loaded.prefactor == mpe.prefactor
        assert loaded.total_charge == mpe.total_charge
        npt.assert_array_equal(loaded.center_of_charge, mpe.center_of_charge)
        with pytest.warns(ExpansionValidityWarning):
            loaded(*mpe.center)

    def test_interior_from_moments(self, tmp_path):
        mpe = MultipoleExpansion.from_moments({(0, 0): 2, (1, 0): 1}, interior=True)
        mpe.save(tmp_path / "e.npz")
        loaded = MultipoleExpansion.load(tmp_path / "e.npz")
        assert loaded.interior and loaded.total_charge is None and loaded.center_of_charge is None
        assert loaded(0, 0, 1) == pytest.approx(3)

    def test_pickle_density(self):
        axes = (np.linspace(-6, 6, 21),) * 3
        mpe = MultipoleExpansion.from_density(gaussian_density(axes, (0, 0, 0), 1), axes, 2)
        assert len(mpe.internal_coords_spherical) == 3  # cached, but not pickled
        restored = pickle.loads(pickle.dumps(mpe))
        npt.assert_array_equal(restored[0], mpe[0])


class TestEnergy:
    def test_matches_sum(self):
        q, xyz = random_charges()
        mpe = MultipoleExpansion.from_charges(q, xyz, 8)
        assert mpe.energy([1, 2], [(20, 0, 0), (0, 20, 0)]) == pytest.approx(mpe(20, 0, 0) + 2 * mpe(0, 20, 0))
        assert mpe.energy(3, (20, 0, 0)) == pytest.approx(3 * mpe(20, 0, 0))

    def test_dipole_in_field_of_charge(self):
        mpe = MultipoleExpansion.from_charges(1, (0, 0, 0), 0)
        # dipole p = 2e-3 e_x at x = 10: W = -p . E = -2e-3 / 100
        W = mpe.energy([1, -1], [(10.001, 0, 0), (9.999, 0, 0)])
        assert W == pytest.approx(-2e-5, rel=1e-6)


class TestPlot:
    def test_plot_slice(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        q, xyz = random_charges()
        mpe = MultipoleExpansion.from_charges(q, xyz, 4)
        fig, axes = plt.subplots(1, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ax = mpe.plot_slice(ax=axes[0], field=True)
            mpe.plot_slice("x", offset=1.0, extent=10, resolution=50, ax=axes[1], cmap="RdBu_r")
        assert ax is axes[0]
        assert ax.get_xlabel() == "x" and axes[1].get_xlabel() == "y"
        plt.close(fig)

        with pytest.raises(ValueError):
            mpe.plot_slice("w")


class TestWarningsPointToCaller:
    def test_new_methods(self):
        mpe = MultipoleExpansion.from_charges([1, -1], [(0, 0, 1), (0, 0, -1)], 3)
        for method in (mpe.hessian, mpe.contributions, mpe.evaluate, mpe.__call__, mpe.field):
            with pytest.warns(ExpansionValidityWarning) as record:
                method(0, 0, 0.1)
            assert record[0].filename == __file__
        with pytest.warns(ExpansionValidityWarning) as record:
            mpe.energy(1, (0, 0, 0.1))
        assert record[0].filename == __file__


class TestPrecision:
    def test_close_to_the_z_axis(self):
        # arccos(z / r) loses all digits of theta close to the z-axis
        mpe = MultipoleExpansion.from_charges(1, (2, 2, 2), 20, interior=True, center=(0, 0, 0))
        p = np.array([5.8e-11, 5.8e-11, 0.866])
        exact = 1 / np.linalg.norm(p - (2, 2, 2))
        assert mpe(*p) == pytest.approx(exact, rel=1e-13)
