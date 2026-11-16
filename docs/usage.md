# Usage

## Installation

Simply use pip:

```
pip install --upgrade multipoles
```

*multipoles* only needs NumPy. To plot expansions, install matplotlib, too, e.g. with
`pip install multipoles[plot]`.

## Point Charges

As example for a discrete charge distribution we model two point charges with positive and negative unit
charge located on the z-axis:

```python
from multipoles import MultipoleExpansion

q = [1, -1]                      # the charges
xyz = [(0, 0, 1), (0, 0, -1)]    # their positions, an array of shape (N, 3)

l_max = 2   # where to stop the infinite multipole sum; here we expand up to the quadrupole (l=2)

Phi = MultipoleExpansion.from_charges(q, xyz, l_max)

# We can evaluate the multipole expanded potential at a given point like this:

x, y, z = 30.5, 30.6, 30.7
value = Phi(x, y, z)

# The multipole moments are stored in a dict, where the keys are (l, m) and the values q_lm:
Phi.multipole_moments
```

For N equal charges, e.g. equal masses, `q` may also be a single number.

## Charge Densities

As an example for a continuous charge distribution, we smear out the point charges from the previous
example. The charge density is given on a grid, described by its axes `x`, `y`, `z`:

```python
import numpy as np

# First we set up our grid, a cube of length 16 centered at the origin:

npoints = 81
edge = 16
x = y = z = np.linspace(-edge/2., edge/2., npoints)
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

# The rest is the same as for the discrete case:

Phi = MultipoleExpansion.from_density(rho, (x, y, z), l_max=2)

value = Phi(30.5, 30.6, 30.7)
```

Instead of the axes, you can also pass the coordinate arrays from `np.meshgrid(x, y, z, indexing='ij')`,
also with `sparse=True`:

```python
Phi = MultipoleExpansion.from_density(rho, XYZ, l_max=2)
```

The axes don't have to be evenly spaced. A grid that is finer where the charge is concentrated saves a lot of
points:

```python
u = np.concatenate([np.linspace(-8, -3, 6)[:-1], np.linspace(-3, 3, 61), np.linspace(3, 8, 6)[1:]])
UVW = np.meshgrid(u, u, u, indexing='ij')
rho_uneven = gaussian(UVW, (0, 0, 1), sigma) - gaussian(UVW, (0, 0, -1), sigma)

Phi_uneven = MultipoleExpansion.from_density(rho_uneven, (u, u, u), l_max=2)
```

The integrals over the grid are computed with Simpson's rule, so the grid must resolve the charge density
well.

### The Dict Interface

Before version 0.6, charge distributions were passed as dict, which still works:

```python
point_charges = {
    'discrete': True,     # point charges are discrete charge distributions
    'charges': [
        {'q': 1, 'xyz': (0, 0, 1)},
        {'q': -1, 'xyz': (0, 0, -1)},
    ]
}
Phi_discrete = MultipoleExpansion(point_charges, l_max=2)

density = {
    'discrete': False,    # a continuous charge distribution
    'rho': rho,
    'xyz': XYZ,           # or the axes (x, y, z)
}
Phi = MultipoleExpansion(density, l_max=2)
```

## Evaluating the Expansion

`Phi(x, y, z)` accepts numbers as well as arrays of any shape. Numbers and arrays are broadcast
against each other, and the result has the broadcast shape:

```python
Phi(30.5, 30.6, 30.7)                  # a single point

xs = np.linspace(10, 20, 5)
Phi(xs, 0, 0)                          # 5 points on the x-axis

XS, YS = np.meshgrid(xs, xs, indexing='ij')
Phi(XS, YS, 30)                        # a 5x5 array on the plane z=30
```

You can also pass the points as one array of shape `(..., 3)`, e.g. the positions of particles:

```python
points = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
Phi(points)                            # 3 values
```

With `l_max` you can restrict the evaluation to the lower multipoles. It must not exceed the `l_max`
the expansion was created with. `contributions` gives the contribution of each order separately, which
shows how fast the expansion converges:

```python
Phi(30.5, 30.6, 30.7, l_max=0)         # monopole only
Phi(30.5, 30.6, 30.7, l_max=1)         # monopole + dipole
Phi.contributions(30.5, 30.6, 30.7)    # [monopole, dipole, quadrupole]
```

## The Field and the Hessian

`Phi.field(x, y, z)` returns the field $\mathbf{E} = -\nabla\Phi$, and `Phi.hessian(x, y, z)` the second
derivatives $\partial^2\Phi/\partial x_i \partial x_j$, both computed analytically from the multipole
moments. They take the same arguments as `Phi(x, y, z)`, including `l_max`. For coordinates `x, y, z`,
the components are stacked along the first axes; for points given as one array of shape `(..., 3)`, along
the last axes:

```python
Ex, Ey, Ez = Phi.field(30.5, 30.6, 30.7)

E = Phi.field(xs, 0, 0)                # shape (3, 5) for the 5 points from above
E = Phi.field(points)                  # shape (3, 3): one row (Ex, Ey, Ez) per point

H = Phi.hessian(30.5, 30.6, 30.7)      # shape (3, 3)
```

The negative Hessian is the gradient of the field, e.g. the electric field gradient, or the tidal tensor in
gravity. If you need several of these quantities at the same points, `evaluate` computes them together,
which is faster:

```python
potential, E, H = Phi.evaluate(points, hessian=True)
```

## Evaluating on the Grid

For charge densities you can also evaluate directly on the grid of the charge
density, using slices, indices or boolean masks:

```python
Phi[:, :, 0]                           # the plane z = -8, an 81x81 array
Phi.field_on_grid(np.s_[:, :, 0])      # the field there, shape (3, 81, 81)
```

A typical use is computing boundary values for a Poisson solver, e.g. one built with
[findiff](https://github.com/maroba/findiff). `boundary_values` returns an array of the shape of the grid
with the potential on the boundary and zeros inside:

```python
boundary = Phi.boundary_values()       # shape (81, 81, 81)
boundary = Phi.boundary_values(width=2)   # the two outermost layers, for wider stencils
```

The multipole expansion is only valid outside of the charge distribution, so the grid must be
large enough for the charge density to have decayed at its boundary (see
[below](#where-the-expansion-is-valid)). Inside the charge distribution, evaluating `Phi[...]`
gives meaningless values and an `ExpansionValidityWarning`.

## Where the Expansion is Valid

The exterior expansion converges only *outside* the smallest sphere about the expansion center
that contains all charge, the interior expansion only *inside* the largest sphere that contains
no charge. The radius of this sphere is available as

```python
Phi.validity_radius

X, Y, Z = Phi.external_coords          # the grid points
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
Phi = MultipoleExpansion.from_density(rho, (x, y, z), l_max=2, charge_tol=1e-4)
```

## Accuracy and the Choice of l_max

`error_bound` gives an upper bound of the error of the truncated expansion at each point (see the
[theory](theory.md#accuracy)):

```python
Phi = MultipoleExpansion.from_charges(q, xyz, l_max=4)
Phi.error_bound(0, 0, 10)              # the error of Phi(0, 0, 10) is at most this
Phi.error_bound(0, 0, 10, l_max=2)     # ... and of Phi(0, 0, 10, l_max=2) at most this
```

Instead of choosing `l_max` yourself, you can let *multipoles* choose the smallest `l_max` for which
the error bound at a distance `radius` from the expansion center (and therefore farther away) is below a
relative tolerance `tol` (default `1e-6`). The tolerance is relative to the potential the absolute charge
would have there, i.e. $\sum_i |q_i| / r$:

```python
Phi = MultipoleExpansion.from_charges(q, xyz, l_max='auto', radius=5, tol=1e-8)
Phi.l_max                              # 11
```

For charge densities, `radius` defaults to the distance to the boundary of the grid, which is what you need
for boundary values:

```python
Phi = MultipoleExpansion.from_density(rho, (x, y, z), l_max='auto', tol=1e-6)
boundary = Phi.boundary_values()
```

## Interior Expansion

By default, *multipoles* computes the exterior expansion, which is valid *outside* of the charge
distribution. For the potential *inside* a region free of charge, surrounded by the charges, use
the interior expansion:

```python
square = [(4, 0, 0), (-4, 0, 0), (0, 4, 0), (0, -4, 0)]

Phi = MultipoleExpansion.from_charges(1, square, l_max=3, interior=True)   # or exterior=False
Phi(0, 0, 0)   # = 4 * 1/4 = 1
```

## The Expansion Center

By default, the expansion is done about the center of charge (weighted by the absolute values of
the charges). You can choose any other point with the `center` argument. This is useful in
particular for the interior expansion, which is valid in a sphere about the center that contains
no charge:

```python
# The center of charge of a single charge is right at the charge, so the interior expansion
# about it is impossible. Expanding about (5, 0, 0) is fine and valid for r < 5:
Phi = MultipoleExpansion.from_charges(1, (0, 0, 0), l_max=10, interior=True, center=(5, 0, 0))
Phi(5, 1, 0)   # = 1 / sqrt(26)
```

`shifted` gives the same expansion about another center:

```python
Phi = MultipoleExpansion.from_charges(q, xyz, l_max=4)
Phi_shifted = Phi.shifted((0, 0, 2))
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
`cartesian_moment(l)` gives the cartesian moment of any order up to `l_max`, a traceless tensor of rank `l`:

```python
Phi.cartesian_moment(3)  # the octupole moment, shape (3, 3, 3)
```

The spherical multipole moments $q_{lm}$ (or $I_{lm}$ for the interior expansion) are available as
a dict with keys `(l, m)`, or as an array that you index with `[l, m]`, where negative `m` wrap
around as usual in Python:

```python
Phi.multipole_moments[(2, -1)]
Phi.multipole_moments_array[2, -1]   # the same
```

They use the Racah normalization of the spherical harmonics (see the [theory](theory.md)). For the moments with
the orthonormal spherical harmonics, as in Jackson's *Classical Electrodynamics*, or for the moments of the real
spherical harmonics:

```python
Phi.moments(normalization='orthonormal')
Phi.moments(real=True)               # [1, 1], [1, -1], [1, 0] are p_x, p_y, p_z
```

`power_spectrum` gives the strength of each order, $\sqrt{\sum_m |q_{lm}|^2}$, which does not change
when you rotate the charge distribution:

```python
Phi.power_spectrum()                 # shape (l_max + 1,)
```

## Combining Expansions

Expansions of the same kind can be added, subtracted and multiplied by numbers. The result is the
expansion of the combined charge distribution:

```python
Phi_a = MultipoleExpansion.from_charges(1, (0, 0, 0), l_max=4)
Phi_b = MultipoleExpansion.from_charges(-1, (0, 0, 2), l_max=4)

Phi_ab = Phi_a + Phi_b                 # a dipole
Phi_ab = 2 * Phi_a - Phi_b / 2
```

## Moments from Elsewhere, Saving and Loading

`from_moments` creates an expansion from known moments, e.g. computed by another program:

```python
Phi = MultipoleExpansion.from_moments({(0, 0): 1, (1, 0): 2}, center=(0, 0, 0))
Phi(0, 0, 10)                          # 1/10 + 2/100
```

`save` stores an expansion in a NumPy `.npz` file, and `load` restores it. The charges are not saved, only
what is needed to evaluate the expansion and its error bound. You can also pickle expansions, which keeps
everything.

```python
import os
import tempfile

Phi = MultipoleExpansion.from_charges(q, xyz, l_max=4)
path = os.path.join(tempfile.mkdtemp(), 'expansion.npz')

Phi.save(path)
Phi_loaded = MultipoleExpansion.load(path)
```

## Energy

The potential energy $\sum_i q_i \Phi(\mathbf{x}_i)$ of other charges in the potential of the expansion:

```python
Phi.energy([1, -1], [(20, 0, 1), (20, 0, -1)])
```

## Plotting

With matplotlib installed, `plot_slice` shows the potential on a plane, optionally with field lines. Points
where the expansion does not converge are left blank:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
Phi.plot_slice(normal='y', field=True, ax=ax)   # the plane y = 0 through the center
plt.close(fig)                                  # or plt.show()
```

## Units and Gravity

*multipoles* computes $\Phi = \sum_i q_i / |\mathbf{r} - \mathbf{r}_i|$ (and the corresponding integral for
charge densities), i.e. the potential in Gaussian units, and its field $\mathbf{E} = -\nabla\Phi$. With
`prefactor`, the potential, field and Hessian are multiplied by a constant:

| `prefactor`          | Constant                    | For                                               |
|----------------------|-----------------------------|---------------------------------------------------|
| `'gaussian'` (default) | 1                         | Electrostatics in Gaussian units                  |
| `'SI'`               | $1 / (4\pi\varepsilon_0)$   | Electrostatics in SI units: V and V/m             |
| `'gravity'`          | $-G$                        | Gravity with masses as `q` in SI units: J/kg and m/s² |
| any number           | that number                 | Anything else                                     |

The constants are available as `multipoles.EPSILON_0` and `multipoles.GRAVITATIONAL_CONSTANT`. The
moments themselves are not scaled.

For example, the gravitational potential and acceleration of the Earth-Moon system far away from both:

```python
masses = [5.972e24, 7.342e22]                  # Earth and Moon, in kg
positions = [(0, 0, 0), (3.844e8, 0, 0)]       # in m

M = MultipoleExpansion.from_charges(masses, positions, l_max=4, prefactor='gravity')

point = (1e9, 2e9, 0.5e9)
potential = M(*point)            # in J / kg
acceleration = M.field(*point)   # in m / s^2, pointing towards the center of mass
```

Since all masses are positive, the center of charge is the center of mass here.
