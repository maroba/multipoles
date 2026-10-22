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

- Discrete (point charges) and continuous (charge density on a grid) distributions
- Exterior and interior expansions, about the center of charge or any point you choose
- The potential and its field E = -∇Φ, computed analytically, at single points or for whole arrays
- Spherical multipole moments, plus total charge, dipole vector and quadrupole tensor
- Warnings when you evaluate the expansion where it does not converge

<!-- --8<-- [end:features] -->

## Installation

```
pip install --upgrade multipoles
```

<!-- --8<-- [start:quickstart] -->
## Quickstart

Two point charges +1 and -1 on the z-axis:

```python
from multipoles import MultipoleExpansion

charge_dist = {
    'discrete': True,
    'charges': [
        {'q': 1, 'xyz': (0, 0, 1)},
        {'q': -1, 'xyz': (0, 0, -1)},
    ],
}

Phi = MultipoleExpansion(charge_dist, l_max=8)

Phi(0, 0, 10)            # 0.0202020199..., exact: 1/9 - 1/11 = 0.0202020202...
Phi(0, 0, 10, l_max=1)   # 0.02, only monopole and dipole
Phi.field(0, 0, 10)      # [0, 0, 0.004081], the field E = -∇Φ
Phi.dipole_moment        # [0, 0, 2]
```

For charge densities on a grid, gravity, the interior expansion and more, see the
[usage guide](https://maroba.github.io/multipoles/usage/).

<!-- --8<-- [end:quickstart] -->

<!-- --8<-- [start:units] -->
## Units

*multipoles* computes Φ = Σ q / r, i.e. the potential in Gaussian units. Multiply by 1 / (4πε₀) for the
electrostatic potential in SI units. For the gravitational potential, pass the masses as charges and multiply
by -G.

<!-- --8<-- [end:units] -->

## Contributing

Bug reports, ideas and pull requests are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md).
