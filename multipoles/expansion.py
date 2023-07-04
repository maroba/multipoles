import numpy as np
from scipy.special import sph_harm


class MultipoleExpansion(object):
    """
    Perform a multipole expansion for a given charge or mass distribution.

    Determines the spherical multipole moments of the given distribution and
    can calculate the solution of the electrostatic or gravitational potential
    based on the multipole expansion.
    """

    def __init__(self, charge_dist, l_max, exterior=None, interior=None):

        """
        Create a MultipoleExpansion object for a given charge or mass distribution.

        Parameters
        ----------

        charge_dist: dict
            description of the charge distribution (see below)

        l_max: positive int
            the maximum multipole moment to consider (0=monopole, 1=dipole, etc.)

        exterior: bool
            whether to perform an exterior expansion (default).
            If false, interior expansion will be used

        interior: bool
            sytaxic override for exterior expansion parameter

        The charge_dist dict:

           For discrete charge distributions (point charges) the dict MUST contain the
           following items:

               Key           Value
               ------------------------------------------------------------------------
               'discrete'    True
               'q'           the charge (positive or negative floating point number)
               'xyz'         the location of the charge in cartesian coordinates
                             (a tuple, list or array of length 3)

            For continuous charge distributions (charge density) the dict MUST contain the
            following items:

               Key           Value
               ------------------------------------------------------------------------
               'discrete'    False
               'rho'         the 3D charge distribution (3D numpy array)
               'xyz'         the domain of the charge distribution
                             (3-tuple of 3D coordinate arrays, see example below)


        =======================
        **Example (Discrete)**:

        As example for a discrete charge distribution we model two point charges with
        positive and negative unit charge located on the z-axis:

        >>> from multipoles import MultipoleExpansion

        Prepare the charge distribution dict for the MultipoleExpansion object:

        >>> charge_dist = {'discrete': True, 'charges': [{'q': 1, 'xyz': (0, 0, 1)}, {'q': -1, 'xyz': (0, 0, -1)}]}
        >>> l_max = 2
        >>> Phi = MultipoleExpansion(charge_dist, l_max)

        Then evaluate on any point desired using Phi(...) or Phi[]. See
        the docstrings of __call__ and __getitem__, respectively.

        =========================
        **Example (Continuous)**:

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
        elif interior is not None and not interior and exterior is not None and not exterior:
            raise InvalidExpansionException("Either interior or exeterior must be set.")
        else:
            exterior = bool(exterior)
            interior = bool(interior)
            exterior = exterior or not interior
            interior = interior or not exterior
            if interior and exterior:
                raise InvalidExpansionException("Interior and exeterior expansion cannot both be set.")

        self.exterior = exterior
        self.interior = interior

        self._assert_charge_dist()

        if charge_dist['discrete']:
            self.charges = list(charge_dist['charges'])

            # calculate center of internal coordinate system = center of absolute charge
            center = np.zeros(3)
            q_total = 0
            for chg in self.charges:
                q = abs(chg['q'])
                q_total += q
                xyz = np.array(chg['xyz'])
                center += q * xyz
            center /= q_total

            self.center_of_charge = center

        else:
            rho = charge_dist['rho']
            X, Y, Z = charge_dist['xyz']

            self.dvol = (X[1, 0, 0] - X[0, 0, 0]) * (Y[0, 1, 0] - Y[0, 0, 0]) * (Z[0, 0, 1] - Z[0, 0, 0])
            self.external_coords = X, Y, Z
            self.rho = rho
            self.total_charge = np.sum(rho) * self.dvol

            # The center of charge expressed in the external coordinate system
            q_abs = np.sum(np.abs(rho)) * self.dvol
            self.center_of_charge = np.array([np.sum(np.abs(rho) * c) for c in [X, Y, Z]]) * self.dvol / q_abs

            # The internal coordinate system is centered at the center of charge
            self.internal_coords = tuple(c - self.center_of_charge[k] for c, k in zip([X, Y, Z], range(3)))
            self.internal_coords_spherical = cartesian_to_spherical(*self.internal_coords)

        if l_max < 0 or l_max != int(l_max):
            raise ValueError("'lmax' must be integer >= 0.")
        self.l_max = l_max

        # The multipole moments are a dict with (l,m) as keys
        self.multipole_moments = self._calc_multipole_moments()

    def __call__(self, *args, **kwargs):
        return self.eval(args, **kwargs)

    def __getitem__(self, *mask):
        """
        Evaluate multipole expansion on grid points specified by mask
        or slices.

        Parameters
        ----------
        mask: three slices or one mask array
            Slices or boolean mask defining on which grid points to
            evaluate the multipole expansion.

        Returns
        -------
        Array of shape defined by `mask` with the values evaluated from
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
        if not isinstance(mask[0], np.ndarray):
            mask = tuple(*mask)
        else:
            mask = mask[0]
        mp_contribs = []
        r, phi, theta = self.internal_coords_spherical
        for l in range(self.l_max + 1):
            phi_l = 0
            for m in range(-l, l + 1):
                Y_lm = sph_harm(m, l, phi[mask], theta[mask])
                q_lm = self.multipole_moments[(l, m)]
                phi_l += np.sqrt(4 * np.pi / (2 * l + 1)) * q_lm * Y_lm / r[mask] ** (l + 1)
            mp_contribs.append(phi_l.real)

        return sum(mp_contribs)

    def eval(self, xyz, l_max=None):
        """
        Evaluate multipole expansion at a point with given coordinates.

        Parameters
        ----------

        xyz: 3-tuple of floats
            The x,y,z coordinates of the points where to evaluate the expansion.

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
                "Multipole expansion only contains multipoles up to l_max={}.".format(self.l_max)
            )
        contribs = self._multipole_contribs(xyz)
        return sum(contribs[:l_max + 1])

    def _multipole_contribs(self, xyz):
        if not isinstance(xyz, np.ndarray):
            xyz = np.array(xyz)

        xyz_internal = xyz - self.center_of_charge
        r, phi, theta = cartesian_to_spherical(*xyz_internal)

        mp_contribs = []

        for l in range(self.l_max + 1):
            phi_l = 0
            for m in range(-l, l + 1):
                Y_lm = sph_harm(m, l, phi, theta)
                q_lm = self.multipole_moments[(l, m)]
                if self.exterior:
                    phi_l += np.sqrt(4 * np.pi / (2 * l + 1)) * q_lm * Y_lm / r ** (l + 1)
                else:
                    phi_l += np.sqrt(4 * np.pi / (2 * l + 1)) * q_lm * Y_lm * r ** l
            mp_contribs.append(phi_l.real)

        return mp_contribs

    def _calc_multipole_moments(self):
        moments = {}
        for l in range(0, self.l_max + 1):
            for m in range(0, l + 1):
                moments[(l, m)] = self._calc_multipole_coef(l, m)
                if m != 0:
                    moments[(l, -m)] = (-1) ** m * np.conj(moments[(l, m)])
        return moments

    def _calc_multipole_coef(self, l, m):

        prefac = np.sqrt(4 * np.pi / (2 * l + 1))

        if self.charge_dist['discrete']:
            q_lm = 0
            for chg in self.charges:
                xyz = chg['xyz'] - self.center_of_charge
                q = chg['q']
                r, phi, theta = cartesian_to_spherical(*xyz)
                Y_lm = sph_harm(m, l, phi, theta)
                if self.exterior:
                    q_lm += q * r ** l * np.conj(Y_lm)
                else:
                    q_lm += q / r ** (l + 1) * np.conj(Y_lm)
            q_lm *= prefac
            return q_lm.real
        else:
            R, Phi, Theta = self.internal_coords_spherical
            Y_lm = sph_harm(m, l, Phi, Theta)
            if self.exterior:
                integrand = R ** l * self.rho * np.conj(Y_lm)
            else:
                integrand = 1 / R ** (l + 1) * self.rho * np.conj(Y_lm)
            return integrand.sum() * self.dvol * prefac

    def _assert_charge_dist(self):

        if 'discrete' not in self.charge_dist:
            raise InvalidChargeDistributionException("Parameter 'discrete' missing.")

        if self.charge_dist['discrete']:
            _check_dict_for_keys(self.charge_dist, ['discrete', 'charges'])

            if not hasattr(self.charge_dist['charges'], '__len__'):
                raise InvalidChargeDistributionException("Parameter 'charges' must be an array-like of dicts.")

            for charge in self.charge_dist['charges']:
                _check_dict_for_keys(charge, ['q', 'xyz'])

        else:
            _check_dict_for_keys(self.charge_dist, ['discrete', 'rho', 'xyz'])


class InvalidChargeDistributionException(Exception):
    pass


class InvalidExpansionException(Exception):
    pass


def cartesian_to_spherical(*coords):
    X, Y, Z = coords
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    R_xy = np.sqrt(X ** 2 + Y ** 2)

    # the 'where' argument in np.arccos is not working as expected,
    # so handle invalid points manually and turn off floating point
    # errors temporarily
    old_settings = np.geterr()
    np.seterr(all='ignore')

    if hasattr(R_xy, '__len__'):
        Phi = np.arccos(X / R_xy)
        Phi[R_xy == 0] = 0
        Theta = np.arccos(Z / R)
        Theta[R == 0] = np.pi / 2
    else:
        if R_xy == 0:
            Phi = 0
        else:
            Phi = np.arccos(X / R_xy)
        if R == 0:
            Theta = np.pi / 2
        else:
            Theta = np.arccos(Z / R)

    np.seterr(**old_settings)
    return R, Phi, Theta


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
