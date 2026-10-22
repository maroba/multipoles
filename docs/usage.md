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

# First we set up our grid, a cube of length 10 centered at the origin:

npoints = 101
edge = 10
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
density, using slices, indices or boolean masks:

```python
Phi[:, :, :]                           # the whole grid, same shape as rho
Phi[:, :, 0]                           # the plane z = -5

mask = np.ones_like(rho, dtype=bool)   # only the boundary of the grid,
mask[1:-1, 1:-1, 1:-1] = False         # e.g. as boundary condition for
boundary = np.zeros_like(rho)          # a Poisson solver
boundary[mask] = Phi[mask]
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

Both expansions are done about the center of charge (weighted by the absolute values of the
charges). For the interior expansion, there must be no charge at this center.
