import numpy as np
from scipy.special import sph_harm


class MultipoleExpansion(object):
    """
    Perform a multipole expansion for a given charge or mass distribution.
    
    Determines the spherical multipole moments of the given distribution and
    can calculate the solution of the electrostatic or gravitational potential
    based on the multipole expansion.    
    """

    def __init__(self, charge_dist, l_max):

        """
        Create a MultipoleExpansion object for a given charge or mass distribution.
        
        :param charge_dist:    a dict describing the charge distribution (see below)
        :param l_max:          the maximum multipole moment to consider (0=monopole, 1=dipole, etc.)
        
        
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
                
        """

        self.charge_dist = charge_dist

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
        if l_max is None:
            l_max = self.l_max
        contribs = self.multipole_contribs(xyz)
        return sum(contribs[:l_max+1])

    def multipole_contribs(self, xyz):
        if not isinstance(xyz, np.ndarray):
            xyz = np.array(xyz)

        xyz_internal = xyz - self.center_of_charge
        r, phi, theta = cartesian_to_spherical(*xyz_internal)
     
        mp_contribs = []

        for l in range(self.l_max + 1):
            phi_l = 0
            for m in range(-l, l+1):
                Y_lm = sph_harm(m, l, phi, theta)
                q_lm = self.multipole_moments[(l, m)]
                phi_l += np.sqrt(4*np.pi/(2*l+1)) * q_lm * Y_lm / r**(l+1)
            mp_contribs.append(phi_l.real)

        return mp_contribs

    def _calc_multipole_moments(self):
        moments = {}
        for l in range(0, self.l_max + 1):
            for m in range(0, l+1):
                moments[(l, m)] = self._calc_multipole_coef(l, m)
                if m != 0:
                    moments[(l, -m)] = (-1)**m * np.conj(moments[(l, m)])
        return moments

    def _calc_multipole_coef(self, l, m):

        prefac = np.sqrt(4*np.pi/(2*l+1))

        if self.charge_dist['discrete']:
            q_lm = 0
            for chg in self.charges:
                xyz = chg['xyz']
                q = chg['q']
                r, phi, theta = cartesian_to_spherical(*xyz)
                Y_lm = sph_harm(m, l, phi, theta)
                q_lm += q * r**l * np.conj(Y_lm)
            q_lm *= prefac
            return q_lm.real
        else:
            R, Phi, Theta = self.internal_coords_spherical
            Y_lm = sph_harm(m, l, Phi, Theta)
            integrand = R**l * self.rho * np.conj(Y_lm)
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


def cartesian_to_spherical(*coords):

    X, Y, Z = coords
    R = np.sqrt(X**2 + Y**2 + Z**2)
    R_xy = np.sqrt(X**2 + Y**2)
    
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
