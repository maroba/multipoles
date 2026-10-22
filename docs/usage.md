# Usage

## Installation

Simply use pip:

```
pip install --upgrade multipoles
```

## Discrete Charge Distribution

As example for a discrete charge distribution we model two point charges with positive and negative unit charge located on the z-axis:

```python
from multipoles import MultipoleExpansion

# Prepare the charge distribution dict for the MultipoleExpansion object:

charge_dist = {
    'discrete': True,     # point charges are discrete charge distributions
    'charges': [
        {'q': 1, 'xyz': (0, 0, 1)},
        {'q': -1, 'xyz': (0, 0, -1)},
    ]
}

l_max = 2   # where to stop the infinite multipole sum; here we expand up to the quadrupole (l=2)

Phi = MultipoleExpansion(charge_dist, l_max)

# We can evaluate the multipole expanded potential at a given point like this:

x, y, z = 30.5, 30.6, 30.7
value = Phi(x, y, z)

# The multipole moments are stored in a dict, where the keys are (l, m) and the values q_lm:
Phi.multipole_moments
```

## Continuous Charge Distribution

As an example for a continuous charge distribution, we smear out the point charges from the previous
example:

```python
from multipoles import MultipoleExpansion
import numpy as np

# First we set up our grid, a cube of length 16 centered at the origin:

npoints = 81
edge = 16
x, y, z = [np.linspace(-edge/2., edge/2., npoints)]*3
XYZ = np.meshgrid(x, y, z, indexing='ij')


# We model our smeared out charges as gaussian functions:

def gaussian(XYZ, xyz0, sigma):
    g = np.ones_like(XYZ[0])
    for k in range(3):
        g *= np.exp(-(XYZ[k] - xyz0[k])**2 / sigma**2)
    g *= (sigma**2*np.pi)**-1.5
    return g

sigma = 1.5   # the width of our gaussians

# Initialize the charge density rho, which is a 3D numpy array:
rho = gaussian(XYZ, (0, 0, 1), sigma) - gaussian(XYZ, (0, 0, -1), sigma)


# Prepare the charge distribution dict for the MultipoleExpansion object:

charge_dist = {
    'discrete': False,     # we have a continuous charge distribution here
    'rho': rho,
    'xyz': XYZ
}

# The rest is the same as for the discrete case:

l_max = 2   # where to stop the infinite multipole sum; here we expand up to the quadrupole (l=2)

Phi = MultipoleExpansion(charge_dist, l_max)

x, y, z = 30.5, 30.6, 30.7
value = Phi(x, y, z)
```
## Evaluating the Expansion

`Phi(x, y, z)` accepts numbers as well as arrays of any shape. Numbers and arrays are broadcast
against each other, and the result has the broadcast shape:

```python
import numpy as np

Phi(30.5, 30.6, 30.7)                  # a single point

x = np.linspace(10, 20, 5)
Phi(x, 0, 0)                           # 5 points on the x-axis

X, Y = np.meshgrid(x, x, indexing='ij')
Phi(X, Y, 30)                          # a 5x5 array on the plane z=30
```

With `l_max` you can restrict the evaluation to the lower multipoles, e.g. to see how much each
order contributes. It must not exceed the `l_max` the expansion was created with:

```python
Phi(30.5, 30.6, 30.7, l_max=0)         # monopole only
Phi(30.5, 30.6, 30.7, l_max=1)         # monopole + dipole
```

For continuous charge distributions you can also evaluate directly on the grid of the charge
density, using slices, indices or boolean masks. A typical use is computing boundary values for a
Poisson solver:

```python
Phi[:, :, 0]                           # the plane z = -8, an 81x81 array

mask = np.ones_like(rho, dtype=bool)   # only the boundary of the grid
mask[1:-1, 1:-1, 1:-1] = False
boundary = np.zeros_like(rho)
boundary[mask] = Phi[mask]
```

The multipole expansion is only valid outside of the charge distribution, so the grid must be
large enough for the charge density to have decayed at its boundary (see
[below](#where-the-expansion-is-valid)). Inside the charge distribution, evaluating `Phi[...]`
gives meaningless values and an `ExpansionValidityWarning`.

## The Field

`Phi.field(x, y, z)` returns the field $\mathbf{E} = -\nabla\Phi$, computed analytically from the
multipole moments. It takes the same arguments as `Phi(x, y, z)`, including `l_max`, and returns
the components $(E_x, E_y, E_z)$ stacked along the first axis:

```python
Ex, Ey, Ez = Phi.field(30.5, 30.6, 30.7)

E = Phi.field(x, 0, 0)                 # shape (3, 5) for the 5 points from above
```

## Where the Expansion is Valid

The exterior expansion converges only *outside* the smallest sphere about the expansion center
that contains all charge, the interior expansion only *inside* the largest sphere that contains
no charge. The radius of this sphere is available as

```python
Phi.validity_radius

X, Y, Z = XYZ
valid = Phi.is_valid(X, Y, Z)          # boolean mask of the grid points where the
values = np.full(rho.shape, np.nan)    # expansion converges
values[valid] = Phi[valid]
```

If you evaluate the potential or the field at points where the expansion does not converge,
*multipoles* emits an `ExpansionValidityWarning`, telling you how many points are affected.
To turn these warnings into errors:

```python
import warnings
from multipoles import ExpansionValidityWarning

warnings.simplefilter("error", ExpansionValidityWarning)
```

A continuous charge density like a gaussian never vanishes exactly. For continuous
distributions, the sphere therefore contains all but a fraction `charge_tol` (default `1e-6`)
of the total absolute charge. If you evaluate on the boundary of a grid that is too small for the
charge density, you get a warning, and the result is less accurate. Enlarge the grid in this case,
or, if you are fine with the accuracy, choose a larger `charge_tol`:

```python
Phi = MultipoleExpansion(charge_dist, l_max, charge_tol=1e-4)
```

## Interior Expansion

By default, *multipoles* computes the exterior expansion, which is valid *outside* of the charge
distribution. For the potential *inside* a region free of charge, surrounded by the charges, use
the interior expansion:

```python
charge_dist = {
    'discrete': True,
    'charges': [
        {'q': 1, 'xyz': (4, 0, 0)},
        {'q': 1, 'xyz': (-4, 0, 0)},
        {'q': 1, 'xyz': (0, 4, 0)},
        {'q': 1, 'xyz': (0, -4, 0)},
    ]
}

Phi = MultipoleExpansion(charge_dist, l_max=3, interior=True)   # or exterior=False
Phi(0, 0, 0)   # = 4 * 1/4 = 1
```

## Multipole Moments

The total charge, the dipole moment vector and the traceless quadrupole moment tensor, all about
the expansion center, are available as

```python
Phi.total_charge         # q
Phi.dipole_moment        # p_i = sum of q x_i
Phi.quadrupole_moment    # Q_ij = sum of q (3 x_i x_j - r^2 delta_ij)
```

so that $\Phi = \frac{q}{r} + \frac{\mathbf{p}\cdot\mathbf{x}}{r^3}
+ \frac{1}{2}\sum_{ij} \frac{Q_{ij} x_i x_j}{r^5} + \dots$ They don't depend on `l_max`.
Except for the lowest non-vanishing moment, they depend on the expansion center.

The spherical multipole moments $q_{lm}$ (or $I_{lm}$ for the interior expansion) are available as
a dict with keys `(l, m)`, or as an array that you index with `[l, m]`, where negative `m` wrap
around as usual in Python:

```python
Phi.multipole_moments[(2, -1)]
Phi.multipole_moments_array[2, -1]   # the same
```

## The Expansion Center

By default, the expansion is done about the center of charge (weighted by the absolute values of
the charges). You can choose any other point with the `center` argument. This is useful in
particular for the interior expansion, which is valid in a sphere about the center that contains
no charge:

```python
charge_dist = {
    'discrete': True,
    'charges': [{'q': 1, 'xyz': (0, 0, 0)}]
}

# The center of charge is (0, 0, 0), right at the charge, so the interior expansion
# about it is impossible. Expanding about (5, 0, 0) is fine and valid for r < 5:
Phi = MultipoleExpansion(charge_dist, l_max=10, interior=True, center=(5, 0, 0))
Phi(5, 1, 0)   # = 1 / sqrt(26)
```

## Units and Gravity

*multipoles* computes $\Phi = \sum_i q_i / |\mathbf{r} - \mathbf{r}_i|$ (and the corresponding integral for
charge densities), i.e. the potential in Gaussian units, and its field $\mathbf{E} = -\nabla\Phi$. For other
unit systems or for gravity, multiply by the appropriate constant:

| Quantity                                   | Formula                               |
|--------------------------------------------|---------------------------------------|
| Electrostatic potential, SI units          | `Phi(x, y, z) / (4 * pi * epsilon_0)` |
| Electric field, SI units                   | `Phi.field(x, y, z) / (4 * pi * epsilon_0)` |
| Gravitational potential (masses as `q`)    | `-G * Phi(x, y, z)`                   |
| Gravitational acceleration (masses as `q`) | `-G * Phi.field(x, y, z)`             |

For example, the gravitational potential and acceleration of the Earth-Moon system far away from both:

```python
import numpy as np
from multipoles import MultipoleExpansion

G = 6.674e-11  # m^3 / (kg s^2)

earth_moon = {
    'discrete': True,
    'charges': [
        {'q': 5.972e24, 'xyz': (0, 0, 0)},        # Earth: mass in kg, position in m
        {'q': 7.342e22, 'xyz': (3.844e8, 0, 0)},  # Moon
    ],
}
M = MultipoleExpansion(earth_moon, l_max=4)   # expanded about the center of mass

point = (1e9, 2e9, 0.5e9)
potential = -G * M(*point)            # in J / kg
acceleration = -G * M.field(*point)   # in m / s^2, pointing towards the center of mass
```

Since all masses are positive, the center of charge is the center of mass here.
