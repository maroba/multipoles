
# multipoles

**multipoles** is a Python package for performing multipole expansions for the solutions of the Poisson equation (e.g. electrostatic or gravitational potentials). It can handle discrete and continuous "charge" distributions.

## Background

For a given function <img src="docs/math/rho.png" alt="rho" height="20"/>, the solution <img src="docs/math/Phi.png" alt="Phi" height="20"/> of the Poisson equation <img src="docs/math/poisson.png" alt="Poisson" height="20"/> with vanishing Dirichlet boundary conditions at infinity is

<img src="docs/math/solution.png" alt="Solution" height="40"/>

If you need to evaluate <img src="docs/math/Phi.png" alt="Phi" height="20"/> at many points, calculating the integral for each point is computationally expensive. As a faster alternative, we can express <img src="docs/math/Phi.png" alt="Phi" height="20"/> in terms of the multipole moments <img src="docs/math/qlm.png" alt="qlm" height="15"/>:

<img src="docs/math/expansion.png" alt="Expansion" height="80"/>

where <img src="docs/math/coords.png" alt="Coordinates" height="20"/> are the usual spherical coordinates corresponding to the cartesian coordinates <img src="docs/math/cartesian.png" alt="Cartesian Coordinates" height="15"/> and <img src="docs/math/Ylm.png" alt="Spherical harmonics" height="20"/> are the spherical harmonics.

The multipole moments are:

<img src="docs/math/moments.png" alt="Multipole Moments" height="40"/>

This approach is usually much faster because the contributions <img src="docs/math/contrib.png" alt="Phi" height="20"/> are getting smaller with increasing <i>l</i>. So we just have to calculate a few integrals for obtaining some <img src="docs/math/qlm.png" alt="qlm" height="15"/>.
