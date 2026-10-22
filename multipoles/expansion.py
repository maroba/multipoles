from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import simpson

# Distances below this threshold are considered to be zero
_EPS = 1.0e-6

# Number of points evaluated at once by MultipoleExpansion.field, to limit memory usage
_CHUNK_SIZE = 2**16


class MultipoleExpansion(object):
    """Perform a multipole expansion for a given charge or mass distribution.

    Determines the spherical multipole moments of the given distribution and
    can calculate the solution of the electrostatic or gravitational potential
    based on the multipole expansion.

    Attributes:
        l_max (int): The maximum multipole moment of the expansion.
        exterior (bool): Whether this is an exterior expansion.
        interior (bool): Whether this is an interior expansion.
        center (numpy.ndarray): The expansion center.
        center_of_charge (numpy.ndarray): The center of charge, weighted by the absolute
            values of the charges.
        total_charge (float): The total charge.
        validity_radius (float): The exterior expansion converges outside, the interior
            expansion inside the sphere of this radius about the expansion center.
        multipole_moments (dict): The multipole moments q_lm (exterior) or I_lm
            (interior), with keys (l, m).
    """

    def __init__(
        self,
        charge_dist: dict,
        l_max: int,
        exterior: bool | None = None,
        interior: bool | None = None,
        center: ArrayLike | None = None,
        charge_tol: float = 1e-6,
    ):
        """Create a MultipoleExpansion object for a given charge or mass distribution.

        Args:
            charge_dist (dict): Description of the charge distribution.

                For discrete charge distributions (point charges) the dict must contain
                exactly the following items:

                - `'discrete'`: `True`
                - `'charges'`: a list of dicts, one for each point charge, with the items
                    - `'q'`: the charge (positive or negative number)
                    - `'xyz'`: the location of the charge in cartesian coordinates
                      (tuple, list or array of length 3)

                For continuous charge distributions (charge density) the dict must contain
                exactly the following items:

                - `'discrete'`: `False`
                - `'rho'`: the charge density on a grid (3D numpy array)
                - `'xyz'`: the grid, a 3-tuple of 3D coordinate arrays as returned by
                  `np.meshgrid(x, y, z, indexing='ij')`

            l_max (int): The maximum multipole moment to consider
                (0 = monopole, 1 = dipole, etc.).
            exterior (bool): Whether to perform an exterior expansion (default).
                If `False`, the interior expansion is used.
            interior (bool): Whether to perform an interior expansion. Alternative to
                `exterior`; only one of both needs to be given.
            center (array_like, optional): The point (x, y, z) about which to expand.
                Defaults to the center of charge, weighted by the absolute values of
                the charges.
            charge_tol (float): Fraction of the total absolute charge of a continuous
                distribution that may be ignored when determining the region where
                the expansion is valid (see `validity_radius`). Smooth densities like
                gaussians never vanish exactly, so without such a tolerance no point
                of the grid would count as outside the distribution.

        The exterior expansion converges only outside the smallest sphere about the
        center that contains all charge, the interior expansion only inside the largest
        sphere about the center that contains no charge. The radius of this sphere is
        stored in `validity_radius`. Evaluating the expansion at points where it does
        not converge emits an `ExpansionValidityWarning`. To turn these warnings into
        errors, use `warnings.simplefilter("error", ExpansionValidityWarning)`.

        Raises:
            InvalidChargeDistributionException: If `charge_dist` is malformed, the total
                absolute charge is zero, or the grid is invalid.
            InvalidExpansionException: If `interior` and `exterior` contradict each other,
                or an interior expansion is requested with charge at the expansion center.
            ValueError: If `center` is not a finite point in 3D.

        Examples:
            A discrete charge distribution: two point charges with positive and negative
            unit charge located on the z-axis:

            >>> from multipoles import MultipoleExpansion
            >>> charge_dist = {
            ...     'discrete': True,
            ...     'charges': [
            ...         {'q': 1, 'xyz': (0, 0, 1)},
            ...         {'q': -1, 'xyz': (0, 0, -1)},
            ...     ],
            ... }
            >>> Phi = MultipoleExpansion(charge_dist, l_max=2)

            A continuous charge distribution: the same charges, smeared out as gaussians
            on a cubic grid of edge length 16 centered at the origin:

            >>> import numpy as np
            >>> x, y, z = [np.linspace(-8, 8, 81)] * 3
            >>> XYZ = np.meshgrid(x, y, z, indexing='ij')
            >>> def gaussian(XYZ, xyz0, sigma):
            ...     g = np.ones_like(XYZ[0])
            ...     for k in range(3):
            ...         g *= np.exp(-(XYZ[k] - xyz0[k])**2 / sigma**2)
            ...     return g * (sigma**2 * np.pi)**-1.5
            >>> rho = gaussian(XYZ, (0, 0, 1), 1.5) - gaussian(XYZ, (0, 0, -1), 1.5)
            >>> Phi = MultipoleExpansion({'discrete': False, 'rho': rho, 'xyz': XYZ}, l_max=2)

            Evaluate the potential at arbitrary points with `Phi(x, y, z)` or on the
            grid points of a continuous distribution with `Phi[...]`, and the field
            with `Phi.field(x, y, z)`. The examples of these methods use `Phi` and
            `gaussian` from here.
        """

        self.charge_dist = charge_dist
        if exterior is None and interior is None:
            exterior = True
        elif exterior is None:
            exterior = not interior
        elif interior is not None and bool(exterior) == bool(interior):
            if exterior:
                raise InvalidExpansionException(
                    "Interior and exterior expansion cannot both be set."
                )
            raise InvalidExpansionException("Either interior or exterior must be set.")

        self.exterior = bool(exterior)
        self.interior = not self.exterior

        self._assert_charge_dist()

        if l_max < 0 or l_max != int(l_max):
            raise ValueError("'lmax' must be integer >= 0.")
        self.l_max = l_max

        # Represent both kinds of distributions as charges (weights) at points xyz
        if charge_dist["discrete"]:
            self.charges = list(charge_dist["charges"])
            xyz = np.array([chg["xyz"] for chg in self.charges], dtype=float).reshape(-1, 3)
            weights = np.array([chg["q"] for chg in self.charges], dtype=float)
            center_weights = np.abs(weights)
            # point charges have exact positions, there is nothing to neglect
            charge_tol = 0

        else:
            rho = np.asarray(charge_dist["rho"])
            X, Y, Z = (np.asarray(c) for c in charge_dist["xyz"])

            if rho.ndim != 3 or any(c.shape != rho.shape for c in (X, Y, Z)):
                raise InvalidChargeDistributionException(
                    "'rho' and the coordinate arrays in 'xyz' must be 3D arrays of the same shape."
                )
            if min(rho.shape) < 2:
                raise InvalidChargeDistributionException(
                    "The grid must have at least 2 points along each axis."
                )

            # Absolute values, so that grids with descending coordinates work, too
            self.dxyz = (
                abs(X[1, 0, 0] - X[0, 0, 0]),
                abs(Y[0, 1, 0] - Y[0, 0, 0]),
                abs(Z[0, 0, 1] - Z[0, 0, 0]),
            )
            if 0 in self.dxyz:
                raise InvalidChargeDistributionException(
                    "Grid spacing is zero along at least one axis. Create the "
                    "coordinate arrays with np.meshgrid(..., indexing='ij')."
                )
            self.dvol = np.prod(self.dxyz)
            self.external_coords = X, Y, Z
            self.rho = rho

            # Integrals over the grid become weighted sums with the Simpson weights
            wx, wy, wz = (_simpson_weights(n, d) for n, d in zip(rho.shape, self.dxyz))
            weights = (wx[:, None, None] * wy[None, :, None] * wz[None, None, :] * rho).ravel()
            xyz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            # Plain average over the grid for the center of charge: the Simpson weights
            # are not symmetric for grids with an even number of points, and would shift
            # the center of symmetric distributions
            center_weights = np.abs(rho).ravel()

        q_abs = np.sum(np.abs(weights))
        if q_abs == 0:
            raise InvalidChargeDistributionException(
                "Total absolute charge must not be zero."
            )
        self.total_charge = np.sum(weights)
        self.center_of_charge = center_weights @ xyz / np.sum(center_weights)
        self.center = self._validated_center(center)

        # The charges relative to the expansion center; points without charge are dropped
        nonzero = weights != 0
        self._charge_xyz = xyz[nonzero] - self.center
        self._charge_weights = weights[nonzero]

        r = np.linalg.norm(self._charge_xyz, axis=1)
        if self.interior and np.any(r < _EPS):
            raise InvalidExpansionException(
                "Interior expansion requires zero charge at the expansion center."
            )
        self.validity_radius = _validity_radius(
            r, self._charge_weights, charge_tol, self.exterior
        )

        if not charge_dist["discrete"]:
            # The grid in the internal coordinate system centered at the expansion center
            self.internal_coords = tuple(
                c - self.center[k] for c, k in zip([X, Y, Z], range(3))
            )
            self.internal_coords_spherical = cartesian_to_spherical(
                *self.internal_coords
            )

        # The multipole moments are a dict with (l,m) as keys
        self.multipole_moments = self._calc_multipole_moments()

    def __call__(self, *xyz: ArrayLike, l_max: int | None = None) -> float | np.ndarray:
        """Evaluate the multipole expansion at points with given coordinates.

        Args:
            *xyz (float | array_like): The x, y, z coordinates of the points where to
                evaluate the expansion. If three numbers are given, the expansion is
                evaluated at the single point (x, y, z). If arrays are given, it is
                evaluated at each point (x[i], y[i], z[i]). Arrays may have any shape;
                numbers and arrays are broadcast against each other.
            l_max (int, optional): Only use contributions up to this angular momentum.
                Defaults to the `l_max` the expansion was created with, and must not
                exceed it.

        Returns:
            float | numpy.ndarray: The value of the potential: a float for a single point,
                otherwise an array of the broadcast shape of the coordinates.

        Warns:
            ExpansionValidityWarning: If some of the points lie where the expansion does
                not converge.

        Examples:
            >>> value = Phi(30.5, 30.6, 30.7)
            >>> value = Phi(30.5, 30.6, 30.7, l_max=1)  # monopole and dipole only
            >>> values = Phi(np.linspace(10, 20, 5), 0, 0)  # 5 points on the x-axis
        """

        return self._eval(xyz, l_max)

    def __getitem__(self, key) -> np.ndarray:
        """Evaluate the multipole expansion on grid points of the charge distribution.

        Only available for continuous charge distributions.

        Args:
            key (index expression | array_like): Slices, indices or a boolean mask
                selecting the points of the grid `charge_dist['xyz']` at which to
                evaluate the expansion.

        Returns:
            numpy.ndarray: The values of the multipole expansion, in the shape defined
                by `key`.

        Raises:
            TypeError: If the charge distribution is discrete.

        Warns:
            ExpansionValidityWarning: If some of the points lie where the expansion does
                not converge.

        Examples:
            >>> x, y, z = [np.linspace(-8, 8, 81)] * 3
            >>> X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            >>> rho = gaussian((X, Y, Z), (0, 0, 0), 1.5)
            >>> mpe = MultipoleExpansion({'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=3)

            Evaluate on the plane `z = -8` (an 81x81 array):

            >>> plane = mpe[:, :, 0]

            Evaluate only on the boundary of the grid, using a boolean mask:

            >>> mask = np.ones_like(rho, dtype=bool)
            >>> mask[1:-1, 1:-1, 1:-1] = False
            >>> boundary_values = np.zeros_like(rho)
            >>> boundary_values[mask] = mpe[mask]
        """
        if self.charge_dist["discrete"]:
            raise TypeError(
                "Evaluation on grid points via [] is only available for continuous "
                "charge distributions. Use Phi(x, y, z) instead."
            )
        r, phi, theta = self.internal_coords_spherical
        self._check_validity(r[key])
        return sum(self._contribs_spherical(r[key], phi[key], theta[key], self.l_max))

    def field(self, *xyz: ArrayLike, l_max: int | None = None) -> np.ndarray:
        """Evaluate the field E = -∇Φ of the multipole expansion.

        The derivatives are computed analytically from the multipole moments.

        Args:
            *xyz (float | array_like): The x, y, z coordinates of the points, as for
                `__call__`.
            l_max (int, optional): Only use contributions up to this angular momentum,
                as for `__call__`.

        Returns:
            numpy.ndarray: The field components (E_x, E_y, E_z), stacked along the first
                axis: shape (3,) for a single point, otherwise (3, *shape) where shape
                is the broadcast shape of the coordinates.

        Warns:
            ExpansionValidityWarning: If some of the points lie where the expansion does
                not converge.

        Examples:
            >>> Ex, Ey, Ez = Phi.field(30.5, 30.6, 30.7)
            >>> E = Phi.field(np.linspace(10, 20, 5), 0, 0)  # shape (3, 5)
        """
        l_max = self._validated_l_max(l_max)
        r, phi, theta = cartesian_to_spherical(*self._internal_coords(xyz))
        self._check_validity(r)

        shape = np.shape(r)
        r, phi, theta = (np.ravel(a) for a in (r, phi, theta))
        E = np.empty((3, r.size))
        for start in range(0, r.size, _CHUNK_SIZE):
            chunk = slice(start, start + _CHUNK_SIZE)
            E[:, chunk] = self._field_spherical(r[chunk], phi[chunk], theta[chunk], l_max)
        return E.reshape((3,) + shape)

    def is_valid(self, *xyz: ArrayLike) -> bool | np.ndarray:
        """Check where the multipole expansion converges.

        Args:
            *xyz (float | array_like): The x, y, z coordinates of the points, as for
                `__call__`.

        Returns:
            bool | numpy.ndarray: Whether the expansion converges at each point, i.e.
                whether the point lies outside (exterior expansion) or inside (interior
                expansion) the sphere of radius `validity_radius` about the expansion
                center.

        Examples:
            >>> X, Y, Z = Phi.external_coords
            >>> valid = Phi.is_valid(X, Y, Z)  # boolean mask of the grid
            >>> values = np.full(valid.shape, np.nan)
            >>> values[valid] = Phi[valid]
        """
        x, y, z = self._internal_coords(xyz)
        return self._is_valid_radius(np.sqrt(x**2 + y**2 + z**2))[()]

    @property
    def dipole_moment(self) -> np.ndarray:
        """The dipole moment vector p = Σ q (r - center), about the expansion center."""
        return self._charge_weights @ self._charge_xyz

    @property
    def quadrupole_moment(self) -> np.ndarray:
        """The traceless quadrupole moment tensor, about the expansion center.

        Q_ij = Σ q (3 x_i x_j - r^2 δ_ij), with x = r - center, so that the potential
        is Φ = q / r + p·x / r^3 + Σ_ij Q_ij x_i x_j / (2 r^5) + ...
        """
        xyz, w = self._charge_xyz, self._charge_weights
        r2 = np.sum(xyz**2, axis=1)
        return 3 * (w[:, None] * xyz).T @ xyz - np.sum(w * r2) * np.eye(3)

    @property
    def multipole_moments_array(self) -> np.ndarray:
        """The multipole moments q_lm as array of shape (l_max + 1, 2 l_max + 1).

        Index it as `[l, m]`, with negative `m` wrapping around as usual in Python.
        Entries with |m| > l are zero.
        """
        moments = np.zeros((self.l_max + 1, 2 * self.l_max + 1), dtype=complex)
        for (l, m), q_lm in self.multipole_moments.items():
            moments[l, m] = q_lm
        return moments

    def __repr__(self) -> str:
        if self.charge_dist["discrete"]:
            distribution = "{} point charges".format(len(self.charges))
        else:
            distribution = "charge density on {}x{}x{} grid".format(*self.rho.shape)
        return "MultipoleExpansion({}, l_max={}, {}, center={})".format(
            distribution,
            self.l_max,
            "exterior" if self.exterior else "interior",
            (np.round(self.center, 6) + 0.0).tolist(),  # + 0.0 turns -0.0 into 0.0
        )

    def _eval(self, xyz, l_max=None):
        """
        Evaluate multipole expansion at a point with given coordinates.

        Parameters
        ----------

        xyz: 3-tuple of (floats or arrays)
            The x,y,z coordinates of the points where to evaluate the expansion.
            If three floats are given, then only one point with coordinates (x, y, z)
            is evaluated. If three arrays xyz=(x, y, z) are given, the evaluation is done at each
            point (x[i], y[i], z[i]).

        l_max: int, optional
            The maximum angular momentum to use for the expansion. If no value
            is given, use l_max from the original computation of the expansion.
            If l_max is given, only use contributions up to this angular momentum
            in the evaluation.
        """

        l_max = self._validated_l_max(l_max)
        r, phi, theta = cartesian_to_spherical(*self._internal_coords(xyz))
        self._check_validity(r, stacklevel=4)  # called via __call__
        return sum(self._contribs_spherical(r, phi, theta, l_max))[()]

    def _validated_l_max(self, l_max):
        if l_max is None:
            return self.l_max
        if l_max > self.l_max:
            raise ValueError(
                "Multipole expansion only contains multipoles up to l_max={}.".format(
                    self.l_max
                )
            )
        return l_max

    def _validated_center(self, center):
        if center is None:
            return self.center_of_charge
        center = np.asarray(center, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("'center' must be a finite point (x, y, z).")
        return center

    def _internal_coords(self, xyz):
        """Broadcast the coordinates and shift them to the expansion center."""
        assert isinstance(xyz, tuple) and len(xyz) == 3

        coords = [np.asarray(c, dtype=float) for c in xyz]
        try:
            coords = np.broadcast_arrays(*coords)
        except ValueError:
            raise ValueError(
                "All arrays must have the same shape. Received: {}".format(
                    [c.shape for c in coords]
                )
            ) from None
        return tuple(c - c0 for c, c0 in zip(coords, self.center))

    def _check_validity(self, r, stacklevel=3):
        """Warn if some of the points with internal radii r lie where the expansion diverges."""
        n_invalid = np.size(r) - np.count_nonzero(self._is_valid_radius(r))
        if n_invalid:
            warnings.warn(
                "{} of {} points lie {} the sphere of radius {:.4g} about the expansion "
                "center {}, where the {} expansion does not converge.".format(
                    n_invalid,
                    np.size(r),
                    "inside" if self.exterior else "outside",
                    self.validity_radius,
                    (np.round(self.center, 6) + 0.0).tolist(),  # + 0.0 turns -0.0 into 0.0
                    "exterior" if self.exterior else "interior",
                ),
                ExpansionValidityWarning,
                stacklevel=stacklevel,
            )

    def _is_valid_radius(self, r):
        if self.exterior:
            return r > self.validity_radius
        return r < self.validity_radius

    def _multipole_contribs(self, xyz, l_max=None):
        if l_max is None:
            l_max = self.l_max

        r, phi, theta = cartesian_to_spherical(*self._internal_coords(tuple(xyz)))
        return self._contribs_spherical(r, phi, theta, l_max)

    def _contribs_spherical(self, r, phi, theta, l_max):
        """Contributions of each l <= l_max at points given in internal spherical coordinates."""
        if self.exterior:
            r = np.maximum(r, _EPS)

        angular = [0] * (l_max + 1)
        for m, column in _racah_columns(l_max, theta, phi):
            for l, C_lm in zip(range(m, l_max + 1), column):
                angular[l] = angular[l] + self.multipole_moments[(l, m)] * C_lm
                if m > 0:
                    # C_l,-m = (-1)^m conj(C_lm)
                    angular[l] = angular[l] + (
                        self.multipole_moments[(l, -m)] * (-1) ** m * np.conj(C_lm)
                    )

        mp_contribs = []
        for l in range(l_max + 1):
            radial = r ** -(l + 1) if self.exterior else r**l
            mp_contribs.append(np.real(angular[l] * radial)[()])
        return mp_contribs

    def _field_spherical(self, r, phi, theta, l_max):
        """The field E = -grad(Phi) at points given as 1D arrays of internal spherical coordinates."""
        if self.exterior:
            r = np.maximum(r, _EPS)

        # Solid harmonics r^l C_lm (interior) or C_lm / r^(l+1) (exterior), of the orders
        # l -/+ 1 needed for the gradient of the terms up to l_max
        l_solid = l_max + 1 if self.exterior else l_max - 1
        solid_harmonics = {}
        for m, column in _racah_columns(l_solid, theta, phi):
            for l, C_lm in zip(range(m, l_solid + 1), column):
                radial = r ** -(l + 1) if self.exterior else r**l
                solid_harmonics[(l, m)] = C_lm * radial
                if m > 0:
                    solid_harmonics[(l, -m)] = (-1) ** m * np.conj(C_lm) * radial

        def solid(l, m):
            return solid_harmonics.get((l, m), 0)

        # Gradient in the components d/dz and D± = d/dx ± i d/dy, using the ladder
        # relations of the solid harmonics
        grad_z, grad_p, grad_m = 0, 0, 0
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                q_lm = self.multipole_moments[(l, m)]
                if self.exterior:
                    grad_z -= q_lm * np.sqrt((l + m + 1) * (l - m + 1)) * solid(l + 1, m)
                    grad_p += q_lm * np.sqrt((l + m + 1) * (l + m + 2)) * solid(l + 1, m + 1)
                    grad_m -= q_lm * np.sqrt((l - m + 1) * (l - m + 2)) * solid(l + 1, m - 1)
                else:
                    grad_z += q_lm * np.sqrt((l + m) * (l - m)) * solid(l - 1, m)
                    grad_p += q_lm * np.sqrt((l - m) * (l - m - 1)) * solid(l - 1, m + 1)
                    grad_m -= q_lm * np.sqrt((l + m) * (l + m - 1)) * solid(l - 1, m - 1)

        grad = [(grad_p + grad_m) / 2, (grad_p - grad_m) / 2j, grad_z]
        # Components without any contribution are scalar zeros
        return [-np.broadcast_to(np.real(g), r.shape) for g in grad]

    def _calc_multipole_moments(self):
        r, phi, theta = cartesian_to_spherical(*self._charge_xyz.T)

        # The charges times r^l (exterior) or 1 / r^(l+1) (interior), for each l
        weighted = []
        factor = self._charge_weights if self.exterior else self._charge_weights / r
        for l in range(self.l_max + 1):
            weighted.append(factor)
            factor = factor * r if self.exterior else factor / r

        moments = {}
        for m, column in _racah_columns(self.l_max, theta, phi):
            for l, C_lm in zip(range(m, self.l_max + 1), column):
                moments[(l, m)] = np.dot(weighted[l], np.conj(C_lm))
                if m > 0:
                    # C_l,-m = (-1)^m conj(C_lm)
                    moments[(l, -m)] = (-1) ** m * np.dot(weighted[l], C_lm)

        return {
            (l, m): moments[(l, m)]
            for l in range(self.l_max + 1)
            for m in range(-l, l + 1)
        }

    def _assert_charge_dist(self):

        if "discrete" not in self.charge_dist:
            raise InvalidChargeDistributionException("Parameter 'discrete' missing.")

        if self.charge_dist["discrete"]:
            _check_dict_for_keys(self.charge_dist, ["discrete", "charges"])

            if not hasattr(self.charge_dist["charges"], "__len__"):
                raise InvalidChargeDistributionException(
                    "Parameter 'charges' must be an array-like of dicts."
                )

            for charge in self.charge_dist["charges"]:
                _check_dict_for_keys(charge, ["q", "xyz"])

        else:
            _check_dict_for_keys(self.charge_dist, ["discrete", "rho", "xyz"])


class InvalidChargeDistributionException(Exception):
    """Raised when the charge distribution is malformed or cannot be expanded, e.g. because
    keys are missing, its total absolute charge is zero, or the grid is invalid."""


class InvalidExpansionException(Exception):
    """Raised when the requested expansion is contradictory or impossible, e.g. an interior
    expansion with charge at the expansion center."""


class ExpansionValidityWarning(UserWarning):
    """Emitted when the expansion is evaluated at points where it does not converge."""


def _racah_columns(l_max, theta, phi):
    """Spherical harmonics in Racah normalization, C_lm = sqrt(4 pi / (2l+1)) Y_lm.

    Yields (m, [C_mm, C_m+1,m, ..., C_l_max,m]) for m = 0, ..., l_max, computed with
    the standard three-term recurrence of the associated Legendre functions. The
    harmonics for negative m follow from C_l,-m = (-1)^m conj(C_lm).
    """
    x, s = np.cos(theta), np.sin(theta)
    e_iphi = np.exp(1j * np.asarray(phi))
    # sqrt((l-m)! / (l+m)!) P_lm(cos theta) for l = m, with Condon-Shortley phase
    c_mm = np.ones_like(x)
    e_imphi = np.ones_like(e_iphi)
    for m in range(l_max + 1):
        if m > 0:
            c_mm = -np.sqrt((2 * m - 1) / (2 * m)) * s * c_mm
            e_imphi = e_imphi * e_iphi
        column = [c_mm]
        if m < l_max:
            column.append(np.sqrt(2 * m + 1) * x * c_mm)
        for l in range(m + 2, l_max + 1):
            column.append(
                (
                    (2 * l - 1) * x * column[-1]
                    - np.sqrt((l - 1) ** 2 - m**2) * column[-2]
                )
                / np.sqrt(l**2 - m**2)
            )
        yield m, [c * e_imphi for c in column]


def _simpson_weights(n, dx):
    """Weights w such that sum(w * f) equals simpson(f, dx=dx) for 1D arrays f of length n."""
    return simpson(np.eye(n), dx=dx, axis=-1)


def _validity_radius(r, charges, charge_tol, exterior):
    """Radius of the sphere about the expansion center that separates the charges
    at radii r from the region where the expansion converges.

    For the exterior expansion, this is the radius of the smallest sphere containing
    all but a fraction `charge_tol` of the absolute charge. For the interior expansion,
    it is the radius of the largest sphere containing at most this fraction.
    """
    weights = np.abs(charges)
    order = np.argsort(r)
    radii = r[order]
    cumulative = np.cumsum(weights[order]) / np.sum(weights)
    if exterior:
        idx = np.searchsorted(cumulative, 1 - charge_tol, side="left")
    else:
        idx = np.searchsorted(cumulative, charge_tol, side="right")
    return radii[min(idx, len(radii) - 1)]


def cartesian_to_spherical(*coords):
    X, Y, Z = coords
    R = np.sqrt(X**2 + Y**2 + Z**2)
    R_xy = np.sqrt(X**2 + Y**2)

    # Z / R is undefined at the origin; those points are replaced afterwards
    with np.errstate(invalid="ignore", divide="ignore"):
        Phi = np.where(R_xy == 0, 0.0, np.arctan2(Y, X))
        Theta = np.where(R == 0, np.pi / 2, np.arccos(Z / R))

    # np.where returns 0-d arrays for scalar input; turn them into scalars
    return R, Phi[()], Theta[()]


def spherical_to_cartesian(r, phi, theta):
    return (
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    )


def _check_dict_for_keys(d, keys):
    msgs = ""
    for key in keys:
        if key not in d:
            msgs += "Parameter '%s' missing.\n" % key
    for key in d.keys():
        if key not in keys:
            msgs += "Unknown parameter '%s'.\n" % key

    if msgs:
        raise InvalidChargeDistributionException(msgs)
