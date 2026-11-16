from __future__ import annotations

import copy
import operator
import os
import sys
import warnings
from functools import cached_property
from itertools import combinations_with_replacement, permutations
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from ._grid import GridError, parse_grid
from ._grid import simpson_weights as _simpson_weights_axis
from ._harmonics import (
    cartesian_to_spherical,
    partial_derivative,
    solid_sum,
    spherical_to_cartesian,
    translate_exterior,
    translate_interior,
)
from ._harmonics import (
    racah_columns as _racah_columns,
)

__all__ = [
    "MultipoleExpansion",
    "Evaluation",
    "ExpansionValidityWarning",
    "InvalidChargeDistributionException",
    "InvalidExpansionException",
    "EPSILON_0",
    "GRAVITATIONAL_CONSTANT",
    "cartesian_to_spherical",
    "spherical_to_cartesian",
]

#: Vacuum permittivity in F/m (CODATA 2022)
EPSILON_0 = 8.8541878188e-12

#: Newtonian constant of gravitation in m^3 / (kg s^2) (CODATA 2022)
GRAVITATIONAL_CONSTANT = 6.67430e-11

_PREFACTORS = {
    "gaussian": 1.0,
    "si": 1 / (4 * np.pi * EPSILON_0),
    "gravity": -GRAVITATIONAL_CONSTANT,
}

# Distances below this threshold are considered to be zero
_EPS = 1.0e-6

# Number of points evaluated at once, to limit memory usage
_CHUNK_SIZE = 2**16

# Number of charges processed at once when computing the moments
_SOURCE_CHUNK_SIZE = 2**15

# Number of radial bins of the absolute charge used for the error bound
_PROFILE_BINS = 512

# The largest l_max that l_max='auto' may choose
_AUTO_L_MAX_LIMIT = 100

# Default relative tolerance for l_max='auto'
_DEFAULT_TOL = 1e-6

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


class Evaluation(NamedTuple):
    """The result of `MultipoleExpansion.evaluate`. Quantities that were not requested are `None`."""

    potential: float | np.ndarray
    field: np.ndarray | None = None
    hessian: np.ndarray | None = None


class _Sources:
    """Charges (weights) at points, the common representation of all charge distributions.

    Attributes:
        xyz: (N, 3) array of positions.
        weights: (N,) array of charges; for densities, the density times the Simpson weight.
        abs_weights: (N,) array of the absolute charge used for the center of charge.
        charge_tol: fraction of the absolute charge that may be neglected for the validity radius.
        kind: 'charges', 'density' or 'combined'.
        grid: the Grid for densities, else None.
        rho: the density for densities, else None.
    """

    def __init__(self, xyz, weights, abs_weights, charge_tol, kind, grid=None, rho=None):
        self.xyz = xyz
        self.weights = weights
        self.abs_weights = abs_weights
        self.charge_tol = charge_tol
        self.kind = kind
        self.grid = grid
        self.rho = rho

    def scaled(self, factor):
        rho = None if self.rho is None else factor * self.rho
        return _Sources(
            self.xyz, factor * self.weights, abs(factor) * self.abs_weights,
            self.charge_tol, self.kind, self.grid, rho,
        )

    @staticmethod
    def combined(a, b):
        return _Sources(
            np.concatenate([a.xyz, b.xyz]),
            np.concatenate([a.weights, b.weights]),
            np.concatenate([a.abs_weights, b.abs_weights]),
            max(a.charge_tol, b.charge_tol),
            "combined",
        )

    def __len__(self):
        return len(self.weights)


class MultipoleExpansion:
    """Perform a multipole expansion for a given charge or mass distribution.

    Determines the spherical multipole moments of the given distribution and
    can calculate the solution of the electrostatic or gravitational potential
    based on the multipole expansion.

    Create it from point charges with `MultipoleExpansion.from_charges`, from a charge
    density on a grid with `MultipoleExpansion.from_density`, from known moments with
    `MultipoleExpansion.from_moments`, or with a dict describing the distribution.

    Attributes:
        l_max (int): The maximum multipole moment of the expansion.
        exterior (bool): Whether this is an exterior expansion.
        interior (bool): Whether this is an interior expansion.
        center (numpy.ndarray): The expansion center.
        center_of_charge (numpy.ndarray | None): The center of charge, weighted by the absolute
            values of the charges. `None` for expansions created from moments.
        total_charge (float | None): The total charge. `None` for interior expansions created
            from moments.
        prefactor (float): The constant the potential and its derivatives are multiplied with.
        validity_radius (float): The exterior expansion converges outside, the interior
            expansion inside the sphere of this radius about the expansion center.
        multipole_moments (dict): The multipole moments q_lm (exterior) or I_lm
            (interior), with keys (l, m), in Racah normalization (see the theory section
            of the documentation).
    """

    def __init__(
        self,
        charge_dist: dict,
        l_max: int | str,
        exterior: bool | None = None,
        interior: bool | None = None,
        center: ArrayLike | None = None,
        charge_tol: float = 1e-6,
        *,
        prefactor: float | str = 1.0,
        tol: float | None = None,
        radius: float | None = None,
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
                - `'xyz'`: the grid, either the three 1D axes `(x, y, z)`, or a 3-tuple of
                  3D coordinate arrays as returned by `np.meshgrid(x, y, z, indexing='ij')`.
                  The axes may be unevenly spaced.

                `MultipoleExpansion.from_charges` and `MultipoleExpansion.from_density` do
                the same without the dict.

            l_max (int | str): The maximum multipole moment to consider
                (0 = monopole, 1 = dipole, etc.), or `'auto'` to choose the smallest
                `l_max` for which the error bound (see `error_bound`) at distance `radius`
                from the center is below `tol`.
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
            prefactor (float | str): The constant the potential, field and Hessian are
                multiplied with. `'gaussian'` (default) is 1, `'SI'` is 1 / (4π ε₀) for the
                electrostatic potential in SI units, and `'gravity'` is -G for the
                gravitational potential of masses in SI units. Any number is fine, too.
            tol (float, optional): Only for `l_max='auto'`: the error bound of the potential at
                distance `radius`, relative to Σ|q| / r, the potential the absolute charge would
                have there. Defaults to 1e-6.
            radius (float, optional): Only for `l_max='auto'`: the smallest (exterior) or
                largest (interior) distance from the center where you will evaluate the
                expansion. For exterior expansions of charge densities it defaults to the
                distance from the center to the boundary of the grid.

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
            ValueError: If `center` is not a finite point in 3D, or `l_max` is not a
                non-negative integer or `'auto'`.

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
        interior = _resolve_interior(exterior, interior)
        sources = _sources_from_dict(charge_dist, charge_tol)
        self._build(sources, l_max, interior, center, prefactor, tol, radius)
        self.charge_dist = charge_dist
        if charge_dist["discrete"]:
            self.charges = list(charge_dist["charges"])

    # ------------------------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------------------------

    @classmethod
    def from_charges(
        cls,
        q: ArrayLike,
        xyz: ArrayLike,
        l_max: int | str,
        *,
        exterior: bool | None = None,
        interior: bool | None = None,
        center: ArrayLike | None = None,
        prefactor: float | str = 1.0,
        tol: float | None = None,
        radius: float | None = None,
    ) -> MultipoleExpansion:
        """Create the multipole expansion of point charges or point masses.

        Args:
            q (float | array_like): The charges, an array of shape (N,), or a number for
                N equal charges.
            xyz (array_like): The positions of the charges, an array of shape (N, 3).
            l_max (int | str): The maximum multipole moment, or `'auto'`, as for the
                constructor.
            exterior, interior, center, prefactor, tol, radius: As for the constructor.

        Returns:
            MultipoleExpansion: The expansion.

        Examples:
            >>> q = [1, -1]
            >>> xyz = [(0, 0, 1), (0, 0, -1)]
            >>> Phi = MultipoleExpansion.from_charges(q, xyz, l_max=4)
            >>> value = Phi(0, 0, 10)
        """
        interior = _resolve_interior(exterior, interior)
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim == 1 and xyz.size == 3:
            xyz = xyz.reshape(1, 3)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise InvalidChargeDistributionException(
                f"The positions 'xyz' must be an array of shape (N, 3), got shape {xyz.shape}."
            )
        try:
            q = np.broadcast_to(np.asarray(q, dtype=float), (len(xyz),))
        except ValueError:
            raise InvalidChargeDistributionException(
                f"Got {np.size(q)} charges for {len(xyz)} positions."
            ) from None
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(xyz))):
            raise InvalidChargeDistributionException("Charges and positions must be finite.")

        self = cls.__new__(cls)
        sources = _Sources(xyz, q.copy(), np.abs(q), 0.0, "charges")
        self._build(sources, l_max, interior, center, prefactor, tol, radius)
        self.charge_dist = None
        return self

    @classmethod
    def from_density(
        cls,
        rho: ArrayLike,
        grid: tuple,
        l_max: int | str,
        *,
        exterior: bool | None = None,
        interior: bool | None = None,
        center: ArrayLike | None = None,
        charge_tol: float = 1e-6,
        prefactor: float | str = 1.0,
        tol: float | None = None,
        radius: float | None = None,
    ) -> MultipoleExpansion:
        """Create the multipole expansion of a charge or mass density on a grid.

        Args:
            rho (array_like): The density, a 3D array.
            grid (tuple): The grid: either the three 1D axes `(x, y, z)`, which may be
                unevenly spaced, or three 3D arrays as returned by
                `np.meshgrid(x, y, z, indexing='ij')`.
            l_max (int | str): The maximum multipole moment, or `'auto'`, as for the
                constructor.
            exterior, interior, center, charge_tol, prefactor, tol, radius: As for the
                constructor.

        Returns:
            MultipoleExpansion: The expansion.

        Examples:
            >>> x = y = z = np.linspace(-8, 8, 81)
            >>> X, Y, Z = np.meshgrid(x, y, z, indexing='ij', sparse=True)
            >>> rho = np.exp(-(X**2 + Y**2 + (Z - 1)**2)) - np.exp(-(X**2 + Y**2 + (Z + 1)**2))
            >>> Phi = MultipoleExpansion.from_density(rho, (x, y, z), l_max=4)
        """
        interior = _resolve_interior(exterior, interior)
        try:
            rho, grid = parse_grid(rho, grid)
        except GridError as e:
            raise InvalidChargeDistributionException(str(e)) from None
        self = cls.__new__(cls)
        self._build(
            _sources_from_density(rho, grid, charge_tol), l_max, interior, center, prefactor,
            tol, radius,
        )
        self.charge_dist = None
        return self

    @classmethod
    def from_moments(
        cls,
        moments: dict | ArrayLike,
        center: ArrayLike = (0, 0, 0),
        *,
        exterior: bool | None = None,
        interior: bool | None = None,
        prefactor: float | str = 1.0,
        validity_radius: float | None = None,
        abs_charge: float | None = None,
    ) -> MultipoleExpansion:
        """Create a multipole expansion from known multipole moments.

        Use it for moments from other programs, or to restore an expansion from
        `multipole_moments` or `multipole_moments_array`. To save and load expansions,
        see also `save` and `load`.

        Args:
            moments (dict | array_like): The moments q_lm (exterior) or I_lm (interior) in
                Racah normalization, as dict with keys (l, m), or as complex array of shape
                (l_max + 1, 2 l_max + 1) indexed by `[l, m]` as `multipole_moments_array`.
                Missing moments are zero. They must describe a real potential, i.e.
                q_l,-m = (-1)^m conj(q_lm).
            center (array_like): The expansion center.
            exterior, interior: The kind of expansion, as for the constructor.
            prefactor (float | str): As for the constructor.
            validity_radius (float, optional): The radius of the sphere about the center
                outside (exterior) or inside (interior) of which the expansion converges.
                Defaults to 0 (exterior) or infinity (interior), i.e. no validity checks.
            abs_charge (float, optional): The total absolute charge. Together with
                `validity_radius`, it enables `error_bound`.

        Returns:
            MultipoleExpansion: The expansion. It has no charges, so the operations that need
            them (e.g. evaluating on a grid) are not available.

        Examples:
            >>> moments = {(0, 0): 1, (1, 0): 2}  # a charge and a dipole along z
            >>> Phi = MultipoleExpansion.from_moments(moments)
            >>> value = Phi(0, 0, 10)  # 1/10 + 2/100
        """
        interior = _resolve_interior(exterior, interior)
        moments = _moments_to_dict(moments)

        self = cls.__new__(cls)
        self._set_kind(interior)
        self.prefactor = _resolve_prefactor(prefactor)
        self.center = _validated_center(center, None)
        self._explicit_center = True
        self._sources = None
        self._charge_xyz = self._charge_weights = None
        self.center_of_charge = None
        self.charge_dist = None
        self.charges = None
        self.l_max = max(l for l, _ in moments)
        self.multipole_moments = moments
        self.total_charge = float(np.real(moments[(0, 0)])) if self.exterior else None

        if validity_radius is None:
            validity_radius = 0.0 if self.exterior else np.inf
        validity_radius = float(validity_radius)
        if not validity_radius >= 0:
            raise ValueError("'validity_radius' must be >= 0.")
        self.validity_radius = validity_radius
        if abs_charge is not None and np.isfinite(validity_radius):
            self._profile = (np.array([validity_radius]), np.array([float(abs_charge)]))
        else:
            self._profile = None
        return self

    @classmethod
    def from_dict(cls, data: dict) -> MultipoleExpansion:
        """Restore an expansion from the dict returned by `to_dict`.

        Args:
            data (dict): The dict, e.g. as stored with `save` and read with `np.load`.

        Returns:
            MultipoleExpansion: The expansion, without charges, as from `from_moments`.
        """
        self = cls.from_moments(
            np.asarray(data["moments"]),
            np.asarray(data["center"]),
            interior=bool(data["interior"]),
            prefactor=float(data["prefactor"]),
            validity_radius=float(data["validity_radius"]),
        )
        radii, charges = np.asarray(data["profile_radii"]), np.asarray(data["profile_charges"])
        self._profile = (radii, charges) if radii.size else None
        total_charge = float(data["total_charge"])
        self.total_charge = None if np.isnan(total_charge) else total_charge
        if "center_of_charge" in data:
            coc = np.asarray(data["center_of_charge"], dtype=float)
            self.center_of_charge = None if np.any(np.isnan(coc)) else coc
        return self

    @classmethod
    def load(cls, file) -> MultipoleExpansion:
        """Load an expansion saved with `save`.

        Args:
            file (str | path | file): The `.npz` file.

        Returns:
            MultipoleExpansion: The expansion, without charges, as from `from_moments`.
        """
        with np.load(file) as data:
            return cls.from_dict({key: data[key] for key in data.files})

    def to_dict(self) -> dict:
        """The expansion as dict of numbers and arrays, e.g. for storing it in a file.

        It contains the moments, the center, the kind of expansion, the prefactor and what is
        needed for the validity checks and the error bound, but not the charges. Restore the
        expansion with `MultipoleExpansion.from_dict`.

        Returns:
            dict: The data of the expansion.
        """
        radii, charges = self._profile if self._profile is not None else ([], [])
        return {
            "format_version": 1,
            "l_max": self.l_max,
            "interior": self.interior,
            "center": np.array(self.center),
            "prefactor": self.prefactor,
            "moments": self.multipole_moments_array,
            "validity_radius": self.validity_radius,
            "total_charge": np.nan if self.total_charge is None else self.total_charge,
            "center_of_charge": (
                np.full(3, np.nan) if self.center_of_charge is None else np.array(self.center_of_charge)
            ),
            "profile_radii": np.asarray(radii, dtype=float),
            "profile_charges": np.asarray(charges, dtype=float),
        }

    def save(self, file) -> None:
        """Save the expansion to a NumPy `.npz` file. Load it with `MultipoleExpansion.load`.

        The charges are not saved, only what is needed to evaluate the expansion.

        Args:
            file (str | path | file): Where to save the expansion.
        """
        np.savez(file, **self.to_dict())

    def _set_kind(self, interior):
        self.interior = bool(interior)
        self.exterior = not self.interior

    def _build(self, sources, l_max, interior, center, prefactor, tol, radius):
        """Set up the expansion from charges at points."""
        self._set_kind(interior)
        self.prefactor = _resolve_prefactor(prefactor)
        self._sources = sources
        self._explicit_center = center is not None
        self.charges = None

        if len(sources) == 0:
            raise InvalidChargeDistributionException("The charge distribution contains no charges.")
        q_abs = np.sum(sources.abs_weights)
        if q_abs == 0:
            raise InvalidChargeDistributionException("Total absolute charge must not be zero.")
        self.total_charge = float(np.sum(sources.weights))
        self.center_of_charge = sources.abs_weights @ sources.xyz / q_abs
        self.center = _validated_center(center, self.center_of_charge)

        # The charges relative to the expansion center; points without charge are dropped
        nonzero = sources.weights != 0
        self._charge_xyz = sources.xyz[nonzero] - self.center
        self._charge_weights = sources.weights[nonzero]

        r = np.linalg.norm(self._charge_xyz, axis=1)
        if self.interior and np.any(r < _EPS):
            raise InvalidExpansionException(
                "Interior expansion requires zero charge at the expansion center."
            )
        self.validity_radius = float(
            _validity_radius(r, self._charge_weights, sources.charge_tol, self.exterior)
        )
        self._profile = _radial_profile(
            r, np.abs(self._charge_weights), self.validity_radius, self.exterior
        )

        if isinstance(l_max, str) and l_max == "auto":
            self.l_max = self._auto_l_max(tol, radius)
        else:
            if tol is not None or radius is not None:
                raise ValueError("'tol' and 'radius' are only used with l_max='auto'.")
            self.l_max = _validated_l_max(l_max)

        self.multipole_moments = _calc_multipole_moments(
            self._charge_xyz, self._charge_weights, self.l_max, self.exterior
        )

    def _auto_l_max(self, tol, radius):
        tol = _DEFAULT_TOL if tol is None else float(tol)
        if not tol > 0:
            raise ValueError("'tol' must be positive.")
        if radius is None:
            grid = self._sources.grid
            if self.exterior and grid is not None:
                radius = grid.distance_to_boundary(self.center)
            if radius is None:
                raise ValueError(
                    "l_max='auto' needs the 'radius' from the center {} the expansion "
                    "will be evaluated.".format("beyond which" if self.exterior else "within which")
                )
        radius = float(radius)
        if not self._is_valid_radius(radius):
            raise ValueError(
                "The expansion does not converge at radius={:.4g}, which lies {} the sphere of "
                "radius {:.4g} about the expansion center. Choose a {} radius{}.".format(
                    radius,
                    "inside" if self.exterior else "outside",
                    self.validity_radius,
                    "larger" if self.exterior else "smaller",
                    ", or a larger grid" if self._sources.grid is not None else "",
                )
            )

        s, Q = self._profile
        if self.exterior:
            ratio, scale = s / radius, np.sum(Q) / radius
            term = Q / (radius - s) * ratio
        else:
            ratio, scale = radius / s, np.sum(Q / s)
            term = Q / (s - radius) * ratio
        for l_max in range(_AUTO_L_MAX_LIMIT + 1):
            if np.sum(term) <= tol * scale:
                return l_max
            term = term * ratio
        raise ValueError(
            f"The error bound at radius={radius:.4g} needs l_max > {_AUTO_L_MAX_LIMIT} for tol={tol:g}. "
            f"Choose a larger tol, or a radius farther from the validity radius {self.validity_radius:.4g}."
        )

    # ------------------------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------------------------

    def __call__(self, *xyz: ArrayLike, l_max: int | None = None) -> float | np.ndarray:
        """Evaluate the multipole expansion at points with given coordinates.

        Args:
            *xyz (float | array_like): The points where to evaluate the expansion: either
                their coordinates x, y, z as three numbers or arrays, or one array of shape
                (..., 3) of points. If three numbers are given, the expansion is evaluated
                at the single point (x, y, z). If arrays are given, it is evaluated at each
                point (x[i], y[i], z[i]). Arrays may have any shape; numbers and arrays are
                broadcast against each other.
            l_max (int, optional): Only use contributions up to this angular momentum.
                Defaults to the `l_max` the expansion was created with, and must not
                exceed it.

        Returns:
            float | numpy.ndarray: The value of the potential: a float for a single point,
                otherwise an array of the broadcast shape of the coordinates (or the shape of
                the array of points without its last axis).

        Warns:
            ExpansionValidityWarning: If some of the points lie where the expansion does
                not converge.

        Examples:
            >>> value = Phi(30.5, 30.6, 30.7)
            >>> value = Phi(30.5, 30.6, 30.7, l_max=1)  # monopole and dipole only
            >>> values = Phi(np.linspace(10, 20, 5), 0, 0)  # 5 points on the x-axis
            >>> values = Phi(np.array([[10, 0, 0], [0, 0, 10]]))  # 2 points
        """
        return self.evaluate(*xyz, field=False, l_max=l_max).potential

    def evaluate(
        self,
        *xyz: ArrayLike,
        field: bool = True,
        hessian: bool = False,
        l_max: int | None = None,
    ) -> Evaluation:
        """Evaluate the potential and, at little extra cost, its derivatives at once.

        Args:
            *xyz (float | array_like): The points, as for `__call__`.
            field (bool): Whether to compute the field, as with `field`.
            hessian (bool): Whether to compute the Hessian, as with `hessian`.
            l_max (int, optional): Only use contributions up to this angular momentum.

        Returns:
            Evaluation: A named tuple `(potential, field, hessian)`, with `None` for the
                quantities that were not requested.

        Warns:
            ExpansionValidityWarning: If some of the points lie where the expansion does
                not converge.

        Examples:
            >>> potential, E, _ = Phi.evaluate(30.5, 30.6, 30.7)
            >>> result = Phi.evaluate(30.5, 30.6, 30.7, hessian=True)
            >>> H = result.hessian
        """
        coords, points_form = self._coords(xyz)
        return self._evaluate(coords, points_form, l_max, field, hessian)

    def __getitem__(self, key) -> np.ndarray:
        """Evaluate the multipole expansion on grid points of the charge distribution.

        Only available for continuous charge distributions.

        Args:
            key (index expression | array_like): Slices, indices or a boolean mask
                selecting the points of the grid at which to evaluate the expansion.

        Returns:
            numpy.ndarray: The values of the multipole expansion, in the shape defined
                by `key`.

        Raises:
            TypeError: If the expansion was not created from a charge density.

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
        return self._evaluate(self._grid_coords(key), False, None, False, False).potential

    def field(self, *xyz: ArrayLike, l_max: int | None = None) -> np.ndarray:
        """Evaluate the field E = -∇Φ of the multipole expansion.

        The derivatives are computed analytically from the multipole moments.

        Args:
            *xyz (float | array_like): The points, as for `__call__`.
            l_max (int, optional): Only use contributions up to this angular momentum,
                as for `__call__`.

        Returns:
            numpy.ndarray: The field components (E_x, E_y, E_z). If the points are given as
                coordinates x, y, z, the components are stacked along the first axis: shape
                (3,) for a single point, otherwise (3, *shape) where shape is the broadcast
                shape of the coordinates. If the points are given as one array of shape
                (..., 3), the result has the same shape.

        Warns:
            ExpansionValidityWarning: If some of the points lie where the expansion does
                not converge.

        Examples:
            >>> Ex, Ey, Ez = Phi.field(30.5, 30.6, 30.7)
            >>> E = Phi.field(np.linspace(10, 20, 5), 0, 0)  # shape (3, 5)
            >>> E = Phi.field(np.array([[10, 0, 0], [0, 0, 10]]))  # shape (2, 3)
        """
        coords, points_form = self._coords(xyz)
        return self._evaluate(coords, points_form, l_max, True, False, potential=False).field

    def hessian(self, *xyz: ArrayLike, l_max: int | None = None) -> np.ndarray:
        """Evaluate the Hessian ∂²Φ/∂x_i∂x_j of the multipole expansion.

        The derivatives are computed analytically from the multipole moments. The negative
        Hessian is the gradient of the field (e.g. the electric field gradient, or the tidal
        tensor in gravity).

        Args:
            *xyz (float | array_like): The points, as for `__call__`.
            l_max (int, optional): Only use contributions up to this angular momentum,
                as for `__call__`.

        Returns:
            numpy.ndarray: The symmetric 3x3 matrix of the second derivatives at each point.
                If the points are given as coordinates x, y, z, the shape is (3, 3, *shape),
                otherwise (..., 3, 3).

        Warns:
            ExpansionValidityWarning: If some of the points lie where the expansion does
                not converge.

        Examples:
            >>> H = Phi.hessian(30.5, 30.6, 30.7)  # shape (3, 3)
        """
        coords, points_form = self._coords(xyz)
        return self._evaluate(coords, points_form, l_max, False, True, potential=False).hessian

    def contributions(self, *xyz: ArrayLike, l_max: int | None = None) -> np.ndarray:
        """The contributions of each order l to the potential.

        Useful to see how fast the expansion converges. Their sum is the potential.

        Args:
            *xyz (float | array_like): The points, as for `__call__`.
            l_max (int, optional): Only return the contributions up to this angular momentum.

        Returns:
            numpy.ndarray: The contributions Φ_l for l = 0, ..., l_max along the first axis,
                i.e. of shape (l_max + 1, *shape).

        Warns:
            ExpansionValidityWarning: If some of the points lie where the expansion does
                not converge.

        Examples:
            >>> phi_l = Phi.contributions(0, 0, 30)  # monopole, dipole, quadrupole
            >>> bool(np.isclose(phi_l.sum(), Phi(0, 0, 30)))
            True
        """
        l_max = self._validated_eval_l_max(l_max)
        coords, _ = self._coords(xyz)
        r = np.sqrt(sum(c**2 for c in coords))
        self._check_validity(r)
        coeffs = self._moments_up_to(l_max)

        shape = r.shape
        flat = [c.ravel() for c in coords]
        result = np.empty((l_max + 1, r.size))
        for chunk in _chunks(r.size):
            rr, phi, theta = self._spherical(*(c[chunk] for c in flat))
            result[:, chunk] = solid_sum([coeffs], rr, phi, theta, self.interior, per_l=True)[0]
        return self.prefactor * result.reshape((l_max + 1,) + shape)

    def field_on_grid(self, key=Ellipsis, l_max: int | None = None) -> np.ndarray:
        """Evaluate the field on grid points of the charge density, like `Phi[key]` for the potential.

        Args:
            key (index expression | array_like): Slices, indices or a boolean mask
                selecting the points of the grid. Defaults to the whole grid.
            l_max (int, optional): Only use contributions up to this angular momentum.

        Returns:
            numpy.ndarray: The field components stacked along the first axis, of shape
                (3, *shape) where shape is the shape defined by `key`.

        Raises:
            TypeError: If the expansion was not created from a charge density.

        Examples:
            >>> E = Phi.field_on_grid(np.s_[0, :, :])  # on the plane x = -8, shape (3, 81, 81)
        """
        return self._evaluate(self._grid_coords(key), False, l_max, True, False, potential=False).field

    def boundary_values(self, width: int = 1, l_max: int | None = None) -> np.ndarray:
        """The potential on the boundary of the grid, e.g. as boundary condition for a Poisson solver.

        Args:
            width (int): The number of layers of grid points at the boundary to fill.
            l_max (int, optional): Only use contributions up to this angular momentum.

        Returns:
            numpy.ndarray: An array of the shape of the grid, with the potential on the
                `width` outermost layers of grid points, and zeros inside.

        Raises:
            TypeError: If the expansion was not created from a charge density.

        Warns:
            ExpansionValidityWarning: If the boundary of the grid is too close to the charge.

        Examples:
            >>> bc = Phi.boundary_values()  # shape (81, 81, 81)
        """
        if operator.index(width) < 1:
            raise ValueError("'width' must be at least 1.")
        grid = self._require_grid()
        mask = grid.boundary_mask(width)
        values = np.zeros(grid.shape)
        values[mask] = self._evaluate(self._grid_coords(mask), False, l_max, False, False).potential
        return values

    def is_valid(self, *xyz: ArrayLike) -> bool | np.ndarray:
        """Check where the multipole expansion converges.

        Args:
            *xyz (float | array_like): The points, as for `__call__`.

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
        coords, _ = self._coords(xyz)
        return self._is_valid_radius(np.sqrt(sum(c**2 for c in coords)))[()]

    def error_bound(self, *xyz: ArrayLike, l_max: int | None = None) -> float | np.ndarray:
        """An upper bound of the truncation error of the potential.

        For each charge q at distance s from the center, the error of the exterior
        expansion at distance r from the center is at most |q| (s/r)^(l_max+1) / (r - s);
        for the interior expansion, swap r and s. The bound is the sum over all charges,
        which are binned radially for speed. For continuous distributions, it ignores the
        fraction `charge_tol` of the charge that lies beyond `validity_radius`, as well
        as the integration error.

        Args:
            *xyz (float | array_like): The points, as for `__call__`.
            l_max (int, optional): The order of the truncated expansion. Defaults to `l_max`.

        Returns:
            float | numpy.ndarray: The bound at each point, infinite where the expansion does
                not converge.

        Raises:
            ValueError: If the expansion was created from moments without `abs_charge`.

        Examples:
            >>> bound = Phi.error_bound(30, 0, 0)
        """
        if self._profile is None:
            raise ValueError(
                "The error bound needs the charges, or 'validity_radius' and 'abs_charge' "
                "for expansions created from moments."
            )
        l_max = self._validated_eval_l_max(l_max)
        coords, _ = self._coords(xyz)
        r = np.sqrt(sum(c**2 for c in coords))
        shape = r.shape
        r = r.ravel()

        s, Q = self._profile
        bound = np.full(r.size, np.inf)
        valid = np.flatnonzero(self._is_valid_radius(r))
        step = max(1, _CHUNK_SIZE // max(1, s.size))
        for start in range(0, valid.size, step):
            idx = valid[start:start + step]
            rr = r[idx, None]
            if self.exterior:
                terms = Q * (s / rr) ** (l_max + 1) / (rr - s)
            else:
                terms = Q * (rr / s) ** (l_max + 1) / (s - rr)
            bound[idx] = np.sum(terms, axis=1)
        return (abs(self.prefactor) * bound).reshape(shape)[()]

    def energy(self, q: ArrayLike, xyz: ArrayLike) -> float:
        """The potential energy Σ q_i Φ(x_i) of point charges in the potential of the expansion.

        Args:
            q (float | array_like): The charges, an array of shape (N,) or a number.
            xyz (array_like): Their positions, an array of shape (N, 3).

        Returns:
            float: The energy, including the `prefactor`.

        Warns:
            ExpansionValidityWarning: If some of the charges lie where the expansion does
                not converge.

        Examples:
            >>> W = Phi.energy([1, -1], [(30, 0, 1), (30, 0, -1)])
        """
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        q = np.broadcast_to(np.asarray(q, dtype=float), (len(xyz),))
        return float(q @ self(xyz))

    def plot_slice(
        self,
        normal: str = "z",
        offset: float | None = None,
        extent: float | None = None,
        resolution: int = 200,
        field: bool = False,
        ax=None,
        l_max: int | None = None,
        **kwargs,
    ):
        """Plot the potential on a plane through the expansion. Needs matplotlib.

        Points where the expansion does not converge are left blank.

        Args:
            normal (str): The axis normal to the plane, `'x'`, `'y'` or `'z'`.
            offset (float, optional): The position of the plane along `normal`. Defaults to
                the expansion center.
            extent (float, optional): Half the edge length of the plotted square about the
                center. Defaults to 4 times (exterior) or 1 times (interior) the validity radius.
            resolution (int): The number of points along each side.
            field (bool): Whether to draw field lines on top.
            ax (matplotlib.axes.Axes, optional): Where to plot. Defaults to the current axes.
            l_max (int, optional): Only use contributions up to this angular momentum.
            **kwargs: Passed on to `ax.pcolormesh`.

        Returns:
            matplotlib.axes.Axes: The axes.
        """
        from ._plot import plot_slice

        return plot_slice(self, normal, offset, extent, resolution, field, ax, l_max, **kwargs)

    # ------------------------------------------------------------------------------------
    # Moments
    # ------------------------------------------------------------------------------------

    @property
    def dipole_moment(self) -> np.ndarray:
        """The dipole moment vector p = Σ q (r - center), about the expansion center."""
        if self._charge_xyz is None:
            return self.cartesian_moment(1)
        return self._charge_weights @ self._charge_xyz

    @property
    def quadrupole_moment(self) -> np.ndarray:
        """The traceless quadrupole moment tensor, about the expansion center.

        Q_ij = Σ q (3 x_i x_j - r^2 δ_ij), with x = r - center, so that the potential
        is Φ = q / r + p·x / r^3 + Σ_ij Q_ij x_i x_j / (2 r^5) + ...
        """
        if self._charge_xyz is None:
            return self.cartesian_moment(2)
        xyz, w = self._charge_xyz, self._charge_weights
        r2 = np.sum(xyz**2, axis=1)
        return 3 * (w[:, None] * xyz).T @ xyz - np.sum(w * r2) * np.eye(3)

    def cartesian_moment(self, l: int) -> np.ndarray:
        """The cartesian multipole moment of order l, a symmetric traceless tensor of rank l.

        For the exterior expansion, it is the tensor M with
        Φ_l = M_{i1...il} x_i1 ... x_il / (l! r^(2l+1)), i.e.
        M = (2l-1)!! Σ q (x^l with traces removed): the total charge for l=0, the dipole
        moment for l=1, the quadrupole moment for l=2, the octupole moment for l=3 etc.,
        all about the expansion center. For the interior expansion, it is the tensor of the
        l-th derivatives of the potential at the center.

        Args:
            l (int): The order, at most `l_max`.

        Returns:
            numpy.ndarray: The tensor, of shape (3,) * l.

        Examples:
            >>> quadrupole = Phi.cartesian_moment(2)  # shape (3, 3), same as Phi.quadrupole_moment
        """
        l = _validated_l_max(l)
        if l > self.l_max:
            raise ValueError(
                f"Multipole expansion only contains multipoles up to l_max={self.l_max}."
            )
        harmonic = {(l, m): self.multipole_moments[(l, m)] for m in range(-l, l + 1)}
        tensor = np.empty((3,) * l)
        for indices in combinations_with_replacement(range(3), l):
            coeffs = harmonic
            for axis in indices:
                coeffs = partial_derivative(coeffs, axis, regular=True)
            value = float(np.real(coeffs.get((0, 0), 0)))
            for perm in set(permutations(indices)):
                tensor[perm] = value
        return tensor

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

    def moments(self, normalization: str = "racah", real: bool = False) -> np.ndarray:
        """The multipole moments in other conventions, as array indexed by `[l, m]`.

        Args:
            normalization (str): `'racah'` (default) for the moments q_lm of this package,
                with the spherical harmonics C_lm = sqrt(4π / (2l+1)) Y_lm, or `'orthonormal'`
                for the moments with the orthonormal Y_lm, e.g. as in Jackson's Classical
                Electrodynamics: sqrt((2l+1) / 4π) q_lm.
            real (bool): Return the real moments for the real spherical harmonics instead:
                `[l, 0]` is q_l0, `[l, m]` with m > 0 is √2 (-1)^m Re q_lm (the cosine part),
                and `[l, -m]` is √2 (-1)^(m+1) Im q_lm (the sine part). With Racah
                normalization, `[1, 1]`, `[1, -1]` and `[1, 0]` are the dipole components
                p_x, p_y and p_z.

        Returns:
            numpy.ndarray: The moments, of shape (l_max + 1, 2 l_max + 1), complex unless
                `real`. Negative `m` wrap around; entries with |m| > l are zero.
        """
        moments = self.multipole_moments_array
        if normalization == "orthonormal":
            l = np.arange(self.l_max + 1)
            moments = moments * np.sqrt((2 * l + 1) / (4 * np.pi))[:, None]
        elif normalization != "racah":
            raise ValueError("'normalization' must be 'racah' or 'orthonormal'.")
        if not real:
            return moments

        result = np.zeros(moments.shape)
        for l in range(self.l_max + 1):
            result[l, 0] = moments[l, 0].real
            for m in range(1, l + 1):
                result[l, m] = np.sqrt(2) * (-1) ** m * moments[l, m].real
                result[l, -m] = np.sqrt(2) * (-1) ** (m + 1) * moments[l, m].imag
        return result

    def power_spectrum(self) -> np.ndarray:
        """The strength of each order l, sqrt(Σ_m |q_lm|²), for l = 0, ..., l_max.

        Unlike the moments themselves, it does not change when the distribution is
        rotated about the expansion center.

        Returns:
            numpy.ndarray: Array of shape (l_max + 1,).
        """
        return np.sqrt(np.sum(np.abs(self.multipole_moments_array) ** 2, axis=1))

    # ------------------------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------------------------

    def shifted(self, center: ArrayLike) -> MultipoleExpansion:
        """The same expansion about another center.

        If the expansion knows its charges, the moments are recomputed about the new center.
        Otherwise, they are translated with the addition theorem of the solid harmonics: for
        exterior expansions, this gives the exact moments up to `l_max` about the new center,
        and the validity radius grows by the distance between the centers. For interior
        expansions, the result is the same truncated polynomial, so it has the same
        truncation error, and the validity radius shrinks by the distance.

        Args:
            center (array_like): The new expansion center.

        Returns:
            MultipoleExpansion: The new expansion.

        Examples:
            >>> shifted = Phi.shifted((0, 0, 1))
        """
        center = _validated_center(center, None)
        if self._sources is not None:
            return self._rebuild(self._sources, center, self.l_max)

        result = copy.copy(self)
        result.center = center
        result._explicit_center = True
        distance = np.linalg.norm(center - self.center)
        if self.exterior:
            result.multipole_moments = translate_exterior(
                self.multipole_moments, self.l_max, self.center - center
            )
            result.validity_radius = self.validity_radius + distance
            if self._profile is not None:
                result._profile = (self._profile[0] + distance, self._profile[1])
        else:
            result.multipole_moments = translate_interior(
                self.multipole_moments, self.l_max, center - self.center
            )
            result.validity_radius = self.validity_radius - distance
            if result.validity_radius <= 0:
                raise InvalidExpansionException(
                    "The new center lies outside the region where the interior expansion converges."
                )
            if self._profile is not None:
                result._profile = (self._profile[0] - distance, self._profile[1])
        return result

    def _rebuild(self, sources, center, l_max):
        result = type(self).__new__(type(self))
        result._build(sources, l_max, self.interior, center, self.prefactor, None, None)
        result.charge_dist = None
        return result

    def __add__(self, other: MultipoleExpansion) -> MultipoleExpansion:
        """The expansion of the combined charge distribution of two expansions.

        Both must be of the same kind (exterior or interior) and have the same prefactor. The
        result has the smaller of both `l_max`. If both know their charges, it is computed
        from all charges, about the center of charge of both, unless an expansion center
        was given explicitly for the left operand. Otherwise, the moments of the right
        operand are translated to the center of the left one and added.
        """
        if not isinstance(other, MultipoleExpansion):
            return NotImplemented
        if self.interior != other.interior:
            raise InvalidExpansionException("Cannot add an interior and an exterior expansion.")
        if not np.isclose(self.prefactor, other.prefactor, rtol=1e-12, atol=0):
            raise InvalidExpansionException("Cannot add expansions with different prefactors.")
        l_max = min(self.l_max, other.l_max)

        if self._sources is not None and other._sources is not None:
            center = self.center if self._explicit_center else None
            return self._rebuild(_Sources.combined(self._sources, other._sources), center, l_max)

        other = other.shifted(self.center)
        result = copy.copy(self)
        result._sources = None
        result._charge_xyz = result._charge_weights = None
        result.charge_dist = None
        result.charges = None
        result.l_max = l_max
        result.multipole_moments = {
            (l, m): self.multipole_moments[(l, m)] + other.multipole_moments[(l, m)]
            for l in range(l_max + 1)
            for m in range(-l, l + 1)
        }
        if self.total_charge is None or other.total_charge is None:
            result.total_charge = None
        else:
            result.total_charge = self.total_charge + other.total_charge
        result.center_of_charge = None
        combine = max if self.exterior else min
        result.validity_radius = combine(self.validity_radius, other.validity_radius)
        if self._profile is None or other._profile is None:
            result._profile = None
        else:
            result._profile = tuple(np.concatenate([a, b]) for a, b in zip(self._profile, other._profile))
        return result

    def __mul__(self, factor: float) -> MultipoleExpansion:
        """The expansion of the charge distribution multiplied by a number."""
        if isinstance(factor, MultipoleExpansion) or not np.isscalar(factor):
            return NotImplemented
        factor = float(factor)
        result = copy.copy(self)
        result.multipole_moments = {k: factor * v for k, v in self.multipole_moments.items()}
        if self.total_charge is not None:
            result.total_charge = factor * self.total_charge
        if self._charge_weights is not None:
            result._charge_weights = factor * self._charge_weights
        if self._sources is not None:
            result._sources = self._sources.scaled(factor)
        if self._profile is not None:
            result._profile = (self._profile[0], abs(factor) * self._profile[1])
        result.charge_dist = None
        result.charges = None
        return result

    __rmul__ = __mul__

    def __truediv__(self, divisor: float) -> MultipoleExpansion:
        if isinstance(divisor, MultipoleExpansion) or not np.isscalar(divisor):
            return NotImplemented
        return self * (1 / divisor)

    def __neg__(self) -> MultipoleExpansion:
        return self * -1

    def __sub__(self, other: MultipoleExpansion) -> MultipoleExpansion:
        if not isinstance(other, MultipoleExpansion):
            return NotImplemented
        return self + (-other)

    # ------------------------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------------------------

    @property
    def rho(self) -> np.ndarray:
        """The charge density, for expansions created from a density."""
        self._require_grid()
        return self._sources.rho

    @property
    def grid_axes(self) -> tuple:
        """The 1D axes (x, y, z) of the grid, for expansions created from a density."""
        return self._require_grid().axes

    @property
    def external_coords(self) -> tuple:
        """The 3D coordinate arrays (X, Y, Z) of the grid, for expansions created from a density."""
        return self._require_grid().coords

    @cached_property
    def internal_coords(self) -> tuple:
        """The grid coordinates relative to the expansion center."""
        return tuple(c - c0 for c, c0 in zip(self.external_coords, self.center))

    @cached_property
    def internal_coords_spherical(self) -> tuple:
        """The grid coordinates relative to the expansion center, in spherical coordinates (r, phi, theta)."""
        return cartesian_to_spherical(*self.internal_coords)

    @property
    def dxyz(self) -> tuple | None:
        """The grid spacing along each axis, for evenly spaced grids, else `None`."""
        grid = self._require_grid()
        if not grid.uniform:
            return None
        return tuple(abs(a[1] - a[0]) for a in grid.axes)

    @property
    def dvol(self) -> float | None:
        """The volume of a grid cell, for evenly spaced grids, else `None`."""
        dxyz = self.dxyz
        return None if dxyz is None else float(np.prod(dxyz))

    def _require_grid(self):
        if self._sources is None or self._sources.grid is None:
            raise TypeError(
                "Evaluation on grid points is only available for expansions of charge "
                "densities on a grid. Use Phi(x, y, z) instead."
            )
        return self._sources.grid

    def _grid_coords(self, key):
        grid = self._require_grid()
        return tuple(np.asarray(c[key], dtype=float) - c0 for c, c0 in zip(grid.coords, self.center))

    # ------------------------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------------------------

    def __repr__(self) -> str:
        sources = self._sources
        if sources is None:
            distribution = "from moments"
        elif sources.kind == "charges":
            distribution = f"{len(sources)} point charges"
        elif sources.kind == "density":
            distribution = "charge density on {}x{}x{} grid".format(*sources.grid.shape)
        else:
            distribution = f"combined distribution of {len(sources)} points"
        return "MultipoleExpansion({}, l_max={}, {}, center={})".format(
            distribution,
            self.l_max,
            "exterior" if self.exterior else "interior",
            (np.round(self.center, 6) + 0.0).tolist(),  # + 0.0 turns -0.0 into 0.0
        )

    def __getstate__(self):
        # cached grid coordinates can be recomputed
        state = self.__dict__.copy()
        state.pop("internal_coords", None)
        state.pop("internal_coords_spherical", None)
        return state

    def _coords(self, xyz):
        """Broadcast the coordinates and shift them to the expansion center.

        Returns the tuple of internal coordinates, and whether the points were given as
        one array of shape (..., 3).
        """
        if len(xyz) == 1:
            points = np.asarray(xyz[0], dtype=float)
            if points.ndim == 0 or points.shape[-1] != 3:
                raise ValueError(
                    "Points must be given as coordinates x, y, z or as one array of shape "
                    f"(..., 3), got an array of shape {points.shape}."
                )
            coords = [points[..., k] for k in range(3)]
            points_form = True
        elif len(xyz) == 3:
            coords = [np.asarray(c, dtype=float) for c in xyz]
            try:
                coords = np.broadcast_arrays(*coords)
            except ValueError:
                raise ValueError(
                    f"All arrays must have the same shape. Received: {[c.shape for c in coords]}"
                ) from None
            points_form = False
        else:
            raise TypeError(
                "Expected the coordinates x, y, z or one array of points of shape (..., 3), "
                f"got {len(xyz)} arguments."
            )
        return tuple(c - c0 for c, c0 in zip(coords, self.center)), points_form

    def _spherical(self, x, y, z):
        r, phi, theta = cartesian_to_spherical(x, y, z)
        if self.exterior:
            r = np.maximum(r, _EPS)
        return r, phi, theta

    def _moments_up_to(self, l_max):
        return {(l, m): q for (l, m), q in self.multipole_moments.items() if l <= l_max}

    def _evaluate(self, coords, points_form, l_max, field, hessian, potential=True):
        """Evaluate the potential and its derivatives at points given in internal coordinates."""
        l_max = self._validated_eval_l_max(l_max)
        r = np.sqrt(sum(c**2 for c in coords))
        self._check_validity(r)

        regular = self.interior
        moments = self._moments_up_to(l_max)
        sets = []
        if potential:
            sets.append(moments)
        if field or hessian:
            gradient = [partial_derivative(moments, i, regular) for i in range(3)]
        if field:
            sets += [{k: -v for k, v in g.items()} for g in gradient]
        if hessian:
            pairs = [(i, j) for i in range(3) for j in range(i, 3)]
            sets += [partial_derivative(gradient[i], j, regular) for i, j in pairs]

        shape = r.shape
        flat = [c.ravel() for c in coords]
        values = np.empty((len(sets), r.size))
        for chunk in _chunks(r.size):
            rr, phi, theta = self._spherical(*(c[chunk] for c in flat))
            values[:, chunk] = solid_sum(sets, rr, phi, theta, regular)
        values = self.prefactor * values.reshape((len(sets),) + shape)

        k = 0
        result_potential = result_field = result_hessian = None
        if potential:
            result_potential = values[0][()]
            k = 1
        if field:
            result_field = values[k:k + 3]
            if points_form:
                result_field = np.moveaxis(result_field, 0, -1)
            k += 3
        if hessian:
            H = np.empty((3, 3) + shape)
            for (i, j), v in zip(pairs, values[k:]):
                H[i, j] = H[j, i] = v
            result_hessian = np.moveaxis(H, (0, 1), (-2, -1)) if points_form else H
        return Evaluation(result_potential, result_field, result_hessian)

    def _validated_eval_l_max(self, l_max):
        if l_max is None:
            return self.l_max
        l_max = _validated_l_max(l_max)
        if l_max > self.l_max:
            raise ValueError(
                f"Multipole expansion only contains multipoles up to l_max={self.l_max}."
            )
        return l_max

    def _check_validity(self, r):
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
                stacklevel=_caller_stacklevel(),
            )

    def _is_valid_radius(self, r):
        if self.exterior:
            return r > self.validity_radius
        return r < self.validity_radius


class InvalidChargeDistributionException(Exception):
    """Raised when the charge distribution is malformed or cannot be expanded, e.g. because
    keys are missing, its total absolute charge is zero, or the grid is invalid."""


class InvalidExpansionException(Exception):
    """Raised when the requested expansion is contradictory or impossible, e.g. an interior
    expansion with charge at the expansion center."""


class ExpansionValidityWarning(UserWarning):
    """Emitted when the expansion is evaluated at points where it does not converge."""


def _caller_stacklevel():
    """The stacklevel for warnings.warn that points to the first caller outside this package,
    counted from the function that calls this one."""
    frame = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        frame = frame.f_back
        level += 1
    return level


def _chunks(n):
    for start in range(0, n, _CHUNK_SIZE):
        yield slice(start, start + _CHUNK_SIZE)


def _resolve_interior(exterior, interior):
    """Whether the expansion is interior, from the exterior and interior arguments."""
    if exterior is None and interior is None:
        return False
    if exterior is None:
        return bool(interior)
    if interior is not None and bool(exterior) == bool(interior):
        if exterior:
            raise InvalidExpansionException("Interior and exterior expansion cannot both be set.")
        raise InvalidExpansionException("Either interior or exterior must be set.")
    return not exterior


def _resolve_prefactor(prefactor):
    if isinstance(prefactor, str):
        try:
            return _PREFACTORS[prefactor.lower()]
        except KeyError:
            raise ValueError(
                "Unknown prefactor '{}'. Use a number or one of {}.".format(
                    prefactor, ", ".join(repr(k) for k in ("gaussian", "SI", "gravity"))
                )
            ) from None
    prefactor = float(prefactor)
    if not np.isfinite(prefactor) or prefactor == 0:
        raise ValueError("'prefactor' must be a finite, nonzero number.")
    return prefactor


def _validated_l_max(l_max):
    if isinstance(l_max, (bool, np.bool_)):
        raise ValueError("'l_max' must be an integer >= 0.")
    try:
        value = operator.index(l_max)
    except TypeError:
        if isinstance(l_max, (float, np.floating)) and float(l_max).is_integer():
            value = int(l_max)
        else:
            raise ValueError(
                f"'l_max' must be an integer >= 0, got {l_max!r}."
            ) from None
    if value < 0:
        raise ValueError(f"'l_max' must be an integer >= 0, got {value}.")
    return value


def _validated_center(center, default):
    if center is None:
        return default
    try:
        center = np.asarray(center, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("'center' must be a finite point (x, y, z).") from None
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("'center' must be a finite point (x, y, z).")
    return center


def _moments_to_dict(moments):
    """Moments given as dict or array [l, m] as complete dict {(l, m): q_lm}."""
    if isinstance(moments, dict):
        if not moments:
            raise ValueError("No moments given.")
        for l, m in moments:
            if not (operator.index(l) >= 0 and abs(m) <= l):
                raise ValueError(f"Invalid moment index (l, m) = {(l, m)}.")
        l_max = max(l for l, _ in moments)
        result = {
            (l, m): complex(moments.get((l, m), 0))
            for l in range(l_max + 1)
            for m in range(-l, l + 1)
        }
    else:
        array = np.asarray(moments)
        if array.ndim != 2 or array.shape[1] != 2 * array.shape[0] - 1:
            raise ValueError(
                "The moments must be a dict or an array of shape (l_max + 1, 2 l_max + 1)."
            )
        l_max = array.shape[0] - 1
        result = {
            (l, m): complex(array[l, m]) for l in range(l_max + 1) for m in range(-l, l + 1)
        }

    scale = max(abs(v) for v in result.values())
    for (l, m), q in result.items():
        if m > 0 and abs(result[(l, -m)] - (-1) ** m * q.conjugate()) > 1e-9 * max(scale, 1e-300):
            raise ValueError(
                "The moments do not describe a real potential: q_l,-m must equal "
                f"(-1)^m conj(q_lm), but not for (l, m) = {(l, m)}."
            )
    return result


def _sources_from_dict(charge_dist, charge_tol):
    if not isinstance(charge_dist, dict):
        raise InvalidChargeDistributionException("The charge distribution must be a dict.")
    if "discrete" not in charge_dist:
        raise InvalidChargeDistributionException("Parameter 'discrete' missing.")

    if charge_dist["discrete"]:
        _check_dict_for_keys(charge_dist, ["discrete", "charges"])
        charges = charge_dist["charges"]
        if not hasattr(charges, "__len__") or isinstance(charges, dict):
            raise InvalidChargeDistributionException(
                "Parameter 'charges' must be an array-like of dicts."
            )
        for charge in charges:
            if not isinstance(charge, dict):
                raise InvalidChargeDistributionException(
                    "Each charge must be a dict with the items 'q' and 'xyz'."
                )
            _check_dict_for_keys(charge, ["q", "xyz"])
        try:
            xyz = np.array([chg["xyz"] for chg in charges], dtype=float).reshape(-1, 3)
            weights = np.array([chg["q"] for chg in charges], dtype=float)
        except ValueError:
            raise InvalidChargeDistributionException(
                "Each charge needs a number 'q' and a position 'xyz' of length 3."
            ) from None
        if len(xyz) != len(weights):
            raise InvalidChargeDistributionException("Each position 'xyz' must have length 3.")
        # point charges have exact positions, there is nothing to neglect
        return _Sources(xyz, weights, np.abs(weights), 0.0, "charges")

    _check_dict_for_keys(charge_dist, ["discrete", "rho", "xyz"])
    try:
        rho, grid = parse_grid(charge_dist["rho"], charge_dist["xyz"])
    except GridError as e:
        raise InvalidChargeDistributionException(str(e)) from None
    return _sources_from_density(rho, grid, charge_tol)


def _sources_from_density(rho, grid, charge_tol):
    if not 0 <= charge_tol < 1:
        raise ValueError("'charge_tol' must be in [0, 1).")
    xyz = np.stack([c.ravel() for c in grid.coords], axis=1)
    # Integrals over the grid become weighted sums with the Simpson weights. The center of
    # charge uses the symmetric trapezoid weights: the Simpson weights are not symmetric for
    # grids with an even number of points, and would shift the center of symmetric distributions
    weights = (grid.simpson_weights() * rho).ravel()
    abs_weights = (grid.volume_weights() * np.abs(rho)).ravel()
    return _Sources(xyz, weights, abs_weights, charge_tol, "density", grid, rho)


def _calc_multipole_moments(xyz, weights, l_max, exterior):
    """The multipole moments of charges `weights` at positions `xyz` relative to the center."""
    moments = np.zeros((l_max + 1, 2 * l_max + 1), dtype=complex)
    for start in range(0, len(weights), _SOURCE_CHUNK_SIZE):
        chunk = slice(start, start + _SOURCE_CHUNK_SIZE)
        r, phi, theta = cartesian_to_spherical(*xyz[chunk].T)
        w = weights[chunk]

        # The charges times r^l (exterior) or 1 / r^(l+1) (interior), for each l
        weighted = []
        factor = w if exterior else w / r
        for _ in range(l_max + 1):
            weighted.append(factor)
            factor = factor * r if exterior else factor / r

        for m, column in _racah_columns(l_max, theta, phi):
            for l, C_lm in zip(range(m, l_max + 1), column):
                moments[l, m] += np.dot(weighted[l], np.conj(C_lm))
                if m > 0:
                    # C_l,-m = (-1)^m conj(C_lm)
                    moments[l, -m] += (-1) ** m * np.dot(weighted[l], C_lm)

    return {
        (l, m): moments[l, m] for l in range(l_max + 1) for m in range(-l, l + 1)
    }


def _simpson_weights(n, dx):
    """Weights w such that sum(w * f) equals the Simpson integral of f for n points with spacing dx."""
    return _simpson_weights_axis(np.arange(n) * dx)


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


def _radial_profile(r, abs_charges, validity_radius, exterior):
    """The absolute charge binned by the distance from the center, for the error bound.

    Only the charge on the side of the validity sphere where it belongs is included. Each
    bin is represented by its radius farthest from the region of convergence, so that the
    error bound stays an upper bound.

    Returns:
        (radii, charges) of the nonempty bins.
    """
    if exterior:
        inside = r <= validity_radius
        r, abs_charges = r[inside], abs_charges[inside]
        edges = validity_radius * np.arange(1, _PROFILE_BINS + 1) / _PROFILE_BINS
        idx = np.minimum(np.searchsorted(edges, r, side="left"), _PROFILE_BINS - 1)
    else:
        outside = r >= validity_radius
        r, abs_charges = r[outside], abs_charges[outside]
        r_max = np.max(r) if r.size else validity_radius
        edges = validity_radius + (r_max - validity_radius) * np.arange(_PROFILE_BINS) / _PROFILE_BINS
        idx = np.maximum(np.searchsorted(edges, r, side="right") - 1, 0)
    charges = np.bincount(idx, weights=abs_charges, minlength=_PROFILE_BINS)
    nonempty = charges > 0
    return edges[nonempty], charges[nonempty]


def _check_dict_for_keys(d, keys):
    msgs = ""
    for key in keys:
        if key not in d:
            msgs += f"Parameter '{key}' missing.\n"
    for key in d.keys():
        if key not in keys:
            msgs += f"Unknown parameter '{key}'.\n"

    if msgs:
        raise InvalidChargeDistributionException(msgs)
