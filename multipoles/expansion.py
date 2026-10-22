import numpy as np
from scipy.integrate import simpson

try:
    from scipy.special import sph_harm_y
except ImportError:  # SciPy < 1.15
    from scipy.special import sph_harm as _sph_harm

    def sph_harm_y(n, m, theta, phi):
        return _sph_harm(m, n, phi, theta)


# Distances below this threshold are considered to be zero
_EPS = 1.0e-6


class MultipoleExpansion(object):
    """Perform a multipole expansion for a given charge or mass distribution.

    Determines the spherical multipole moments of the given distribution and
    can calculate the solution of the electrostatic or gravitational potential
    based on the multipole expansion.
    """

    def __init__(self, charge_dist, l_max, exterior=None, interior=None):
        """Create a MultipoleExpansion object for a given charge or mass distribution.

        Args:
            charge_dist (dict): description of the charge distribution

               For discrete charge distributions (point charges) the dict MUST contain the
               following (key:value)-pairs:

                   `'discrete'`:    True

                   `'charges'`:     a list of dicts, one for each point charge, with the items

                                    `'q'`: the charge (positive or negative floating point number)

                                    `'xyz'`: the location of the charge in cartesian coordinates
                                    (a tuple, list or array of length 3)

                For continuous charge distributions (charge density) the dict MUST contain the
                following items:

                   'discrete':    False
                   'rho':         the 3D charge distribution (3D numpy array)
                   'xyz':         the domain of the charge distribution
                                  (3-tuple of 3D coordinate arrays as returned by
                                  np.meshgrid(..., indexing='ij'), see example below)

            l_max (int): the maximum multipole moment to consider (0=monopole, 1=dipole, etc.)

            exterior (bool): whether to perform an exterior expansion (default).
                    If false, interior expansion will be used.

            interior (bool): syntactic override for exterior expansion parameter

        ## Examples

        As example for a discrete charge distribution we model two point charges with
        positive and negative unit charge located on the z-axis:

            >>> from multipoles import MultipoleExpansion

        Prepare the charge distribution dict for the MultipoleExpansion object:

            >>> charge_dist = {'discrete': True, 'charges': [{'q': 1, 'xyz': (0, 0, 1)}, {'q': -1, 'xyz': (0, 0, -1)}]}
            >>> l_max = 2
            >>> Phi = MultipoleExpansion(charge_dist, l_max)

        Then evaluate on any point desired using Phi(...) or Phi[]. See
        the docstrings of __call__ and __getitem__, respectively.

        As an example for a continuous charge distribution, we smear out the point charges from the previous example:

            >>> from multipoles import MultipoleExpansion
            >>> import numpy as np

        First we set up our grid, a cube of length 10 centered at the origin:

            >>> npoints = 101
            >>> edge = 10
            >>> x, y, z = [np.linspace(-edge/2., edge/2., npoints)]*3
            >>> XYZ = np.meshgrid(x, y, z, indexing='ij')

        We model our smeared out charges as gaussian functions:

            >>> def gaussian(XYZ, xyz0, sigma):
            >>>    g = np.ones_like(XYZ[0])
            >>>    for k in range(3):
            >>>        g *= np.exp(-(XYZ[k] - xyz0[k])**2 / sigma**2)
            >>>    g *= (sigma**2*np.pi)**-1.5
            >>>    return g

        The width of our gaussians:

            >>> sigma = 1.5

        Initialize the charge density rho, which is a 3D numpy array:

            >>> rho = gaussian(XYZ, (0, 0, 1), sigma) - gaussian(XYZ, (0, 0, -1), sigma)

        Prepare the charge distribution dict for the MultipoleExpansion object:

            >>> charge_dist = {'discrete': False, 'rho': rho, 'xyz': XYZ}

        The rest is the same as for the discrete case:

            >>> l_max = 2
            >>> Phi = MultipoleExpansion(charge_dist, l_max)

        Then evaluate on any point desired using Phi(...) or Phi[]. See
        the docstrings of __call__ and __getitem__, respectively.

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

        if charge_dist["discrete"]:
            self.charges = list(charge_dist["charges"])

            # calculate center of internal coordinate system = center of absolute charge
            center = np.zeros(3)
            q_total = 0
            for chg in self.charges:
                q = abs(chg["q"])
                q_total += q
                xyz = np.array(chg["xyz"])
                center += q * xyz
            if q_total == 0:
                raise InvalidChargeDistributionException(
                    "Total absolute charge must not be zero."
                )
            center /= q_total

            self.center_of_charge = center

            if self.interior and any(
                chg["q"] != 0 and np.linalg.norm(np.asarray(chg["xyz"]) - center) < _EPS
                for chg in self.charges
            ):
                raise InvalidExpansionException(
                    "Interior expansion requires all charges to be away from the "
                    "expansion center (the center of charge)."
                )

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
            self.total_charge = np.sum(rho) * self.dvol

            # The center of charge expressed in the external coordinate system
            q_abs = np.sum(np.abs(rho)) * self.dvol
            if q_abs == 0:
                raise InvalidChargeDistributionException(
                    "Total absolute charge must not be zero."
                )
            self.center_of_charge = (
                np.array([np.sum(np.abs(rho) * c) for c in [X, Y, Z]])
                * self.dvol
                / q_abs
            )

            # The internal coordinate system is centered at the center of charge
            self.internal_coords = tuple(
                c - self.center_of_charge[k] for c, k in zip([X, Y, Z], range(3))
            )
            self.internal_coords_spherical = cartesian_to_spherical(
                *self.internal_coords
            )

            if self.interior and np.any(
                (self.internal_coords_spherical[0] < _EPS) & (rho != 0)
            ):
                raise InvalidExpansionException(
                    "Interior expansion requires zero charge density at the "
                    "expansion center (the center of charge)."
                )

        if l_max < 0 or l_max != int(l_max):
            raise ValueError("'lmax' must be integer >= 0.")
        self.l_max = l_max

        # The multipole moments are a dict with (l,m) as keys
        self.multipole_moments = self._calc_multipole_moments()

    def __call__(self, *xyz, l_max=None):
        """Evaluate multipole expansion at a point with given coordinates.

        Args:
            xyz (3-tuple of floats|array): The x,y,z coordinates of the points where to evaluate the expansion.
                If three floats are given, then only one point with coordinates (x, y, z)
                is evaluated. If three arrays xyz=(x, y, z) are given, the evaluation is done at each
                point (x[i], y[i], z[i]). Arrays may have any shape (numbers and arrays are
                broadcast against each other); the result has the broadcast shape.

            l_max (int, optional): The maximum angular momentum to use for the expansion.
                If no value is given, use l_max from the original computation of the expansion.
                If l_max is given, only use contributions up to this angular momentum
                in the evaluation.
        """

        return self._eval(xyz, l_max)

    def __getitem__(self, key):
        """
        Evaluate multipole expansion on grid points specified by mask
        or slices.

        Parameters
        ----------
        key: index expression (e.g. three slices) or boolean mask array
            Defines on which points of the grid of the (continuous) charge
            distribution to evaluate the multipole expansion.

        Returns
        -------
        Array of shape defined by `key` with the values evaluated from
        the multipole expansion.

        Examples
        --------

        >>> # Define a grid 51x51x51 and a gaussian charge distribution
        >>> x, y, z = [np.linspace(-5, 5, 51)] * 3
        >>> X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        >>> sigma = 1.5
        >>> rho = gaussian((X, Y, Z), (0, 0, 0), sigma)
        >>> mpe = MultipoleExpansion(charge_dist={'discrete': False, 'rho': rho, 'xyz': (X, Y, Z)}, l_max=3)

        With the multipole expansion thus defined, we can evaluate using slices
        or masks.

        1. Evaluating with slices

        >>> actual = mpe[:, :, :]

        `actual` is now a 51x51x51 array with values filled by evaluating
        the multipole expansion on X, Y, Z.

        >>> actual = mpe[:, :, 0]

        `actual is now a 51x51 array with values filled by evaluating at
        X[:, :, 0], Y[:, :, 0], Z[:, :, 0]

        2. Evaluating with masks

        Define a boolean mask for the boundary:

        >>> mask = np.ones_like(rho, dtype=bool)
        >>> mask[1:-1, 1:-1, 1:-1] = False

        Define an empty array and evaluate only multipole expansion only
        at the boundary:

        >>> actual = np.zeros_like(rho)
        >>> actual[mask] = mpe[mask]
        """
        if self.charge_dist["discrete"]:
            raise TypeError(
                "Evaluation on grid points via [] is only available for continuous "
                "charge distributions. Use Phi(x, y, z) instead."
            )
        r, phi, theta = self.internal_coords_spherical
        return sum(self._contribs_spherical(r[key], phi[key], theta[key], self.l_max))

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

        if l_max is None:
            l_max = self.l_max
        if l_max > self.l_max:
            raise ValueError(
                "Multipole expansion only contains multipoles up to l_max={}.".format(
                    self.l_max
                )
            )
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

        return sum(self._multipole_contribs(coords, l_max))[()]

    def _multipole_contribs(self, xyz, l_max=None):
        if l_max is None:
            l_max = self.l_max

        x, y, z = (np.asarray(c, dtype=float) - c0 for c, c0 in zip(xyz, self.center_of_charge))
        r, phi, theta = cartesian_to_spherical(x, y, z)
        return self._contribs_spherical(r, phi, theta, l_max)

    def _contribs_spherical(self, r, phi, theta, l_max):
        """Contributions of each l <= l_max at points given in internal spherical coordinates."""
        if self.exterior:
            r = np.maximum(r, _EPS)

        mp_contribs = []

        for l in range(l_max + 1):
            phi_l = 0
            for m in range(-l, l + 1):
                Y_lm = sph_harm_y(l, m, theta, phi)
                q_lm = self.multipole_moments[(l, m)]
                if self.exterior:
                    phi_l += (
                        np.sqrt(4 * np.pi / (2 * l + 1)) * q_lm * Y_lm / r ** (l + 1)
                    )
                else:
                    phi_l += np.sqrt(4 * np.pi / (2 * l + 1)) * q_lm * Y_lm * r**l
            mp_contribs.append(np.real(phi_l)[()])
        return mp_contribs

    def _calc_multipole_moments(self):
        moments = {}
        for l in range(0, self.l_max + 1):
            for m in range(-l, l + 1):
                moments[(l, m)] = self._calc_multipole_coef(l, m)
        return moments

    def _calc_multipole_coef(self, l, m):

        prefac = np.sqrt(4 * np.pi / (2 * l + 1))

        if self.charge_dist["discrete"]:
            q_lm = 0
            for chg in self.charges:
                xyz = np.asarray(chg["xyz"]) - self.center_of_charge
                q = chg["q"]
                if q == 0:
                    continue
                r, phi, theta = cartesian_to_spherical(*xyz)
                Y_lm = sph_harm_y(l, m, theta, phi)
                if self.exterior:
                    q_lm += q * r**l * np.conj(Y_lm)
                else:
                    q_lm += q / r ** (l + 1) * np.conj(Y_lm)
            q_lm *= prefac
            return q_lm
        else:
            R, Phi, Theta = self.internal_coords_spherical
            Y_lm = sph_harm_y(l, m, Theta, Phi)
            if self.exterior:
                integrand = R**l * self.rho * np.conj(Y_lm)
            else:
                # rho vanishes where R < _EPS (checked in __init__), so clipping R
                # avoids 0 * inf = nan without changing the integral
                integrand = self.rho * np.conj(Y_lm) / np.maximum(R, _EPS) ** (l + 1)
            return self._integrate(integrand) * prefac

    def _integrate(self, integrand):
        return simpson(
            simpson(
                simpson(integrand, dx=self.dxyz[2], axis=-1), dx=self.dxyz[1], axis=-1
            ),
            dx=self.dxyz[0],
            axis=-1,
        )

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
    pass


class InvalidExpansionException(Exception):
    pass


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
