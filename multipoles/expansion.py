import numpy as np
from scipy.special import sph_harm


class MultipoleExpansion(object):

    def __init__(self, charge_dist, l_max):

        self.charge_dist = charge_dist

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

        self.l_max = l_max

        # The multipole moments are a dict with (l,m) as keys
        self.multipole_moments = self._calc_multipole_moments()

    def __call__(self, *args, **kwargs):
        return self.eval(args, **kwargs)

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