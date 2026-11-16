# multipoles

<p>
<a href="https://pypi.org/project/multipoles/"><img src="https://badge.fury.io/py/multipoles.svg" alt="PyPI version"></a>
<a href="https://pypi.org/project/multipoles/"><img src="https://img.shields.io/pypi/pyversions/multipoles.svg" alt="Python versions"></a>
<a href="https://github.com/maroba/multipoles/actions/workflows/python-app.yml"><img src="https://github.com/maroba/multipoles/actions/workflows/python-app.yml/badge.svg" alt="Build status"></a>
<a href="https://codecov.io/gh/maroba/multipoles"><img src="https://codecov.io/gh/maroba/multipoles/branch/master/graph/badge.svg?token=JNH9SP7BRG" alt="Code coverage"></a>
<a href="https://github.com/maroba/multipoles/blob/master/LICENSE"><img src="https://img.shields.io/pypi/l/multipoles.svg" alt="License"></a>
</p>

*multipoles* computes multipole expansions of the solutions of the Poisson equation, such as electrostatic
or gravitational potentials. Instead of integrating over the whole charge or mass distribution for every point
where you need the potential, you compute a few multipole moments once and evaluate the expansion cheaply
anywhere outside (or, for the interior expansion, inside) the distribution.

**[Documentation](https://maroba.github.io/multipoles/)** ·
**[Changelog](https://github.com/maroba/multipoles/blob/master/CHANGELOG.md)**

<!-- --8<-- [start:features] -->
## Features

- Point charges, or charge densities on evenly or unevenly spaced grids
- Exterior and interior expansions, about the center of charge or any point you choose
- The potential, the field E = -∇Φ and the Hessian, computed analytically, at single points or for whole arrays
- Potential and field on the grid, e.g. boundary values for Poisson solvers
- Error bounds, and `l_max='auto'` to reach a given accuracy
- Spherical moments in several conventions, and cartesian moments of any order
- Moving, adding and scaling expansions; saving and loading them
- Electrostatics in Gaussian or SI units, and gravity
- Warnings when you evaluate the expansion where it does not converge

<!-- --8<-- [end:features] -->

## Installation

```
pip install --upgrade multipoles
```

*multipoles* only needs NumPy. For plotting, install it with `pip install multipoles[plot]`.

<!-- --8<-- [start:quickstart] -->
## Quickstart

Two point charges +1 and -1 on the z-axis:

```python
from multipoles import MultipoleExpansion

Phi = MultipoleExpansion.from_charges(q=[1, -1], xyz=[(0, 0, 1), (0, 0, -1)], l_max=8)

Phi(0, 0, 10)            # 0.0202020199..., exact: 1/9 - 1/11 = 0.0202020202...
Phi(0, 0, 10, l_max=1)   # 0.02, only monopole and dipole
Phi.field(0, 0, 10)      # [0, 0, 0.004081], the field E = -∇Φ
Phi.error_bound(0, 0, 10)  # 2.2e-10, an upper bound of the truncation error
Phi.dipole_moment        # [0, 0, 2]
```

A charge density on a grid, with `l_max` chosen for a relative accuracy of 1e-6 on the boundary of the grid:

```python
import numpy as np

x = y = z = np.linspace(-10, 10, 101)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij', sparse=True)
rho = np.exp(-(X**2 + Y**2 + (Z - 1)**2)) - np.exp(-(X**2 + Y**2 + (Z + 1)**2))

Phi = MultipoleExpansion.from_density(rho, (x, y, z), l_max='auto')
boundary = Phi.boundary_values()   # the potential on the boundary of the grid
```

For the interior expansion, units and gravity, moments, and more, see the
[usage guide](https://maroba.github.io/multipoles/usage/).

<!-- --8<-- [end:quickstart] -->

<!-- --8<-- [start:units] -->
## Units

*multipoles* computes Φ = Σ q / r, i.e. the potential in Gaussian units. Pass `prefactor='SI'` for the
electrostatic potential in SI units, or pass masses as charges and `prefactor='gravity'` for the gravitational
potential.

<!-- --8<-- [end:units] -->

## Contributing

Bug reports, ideas and pull requests are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md).
