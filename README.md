
# multipoles

**multipoles** is a Python package for performing multipole expansions for the solutions of the Poisson equation (e.g. electrostatic or gravitational potentials). Can handle discrete and continuous "charge" distributions.

## Background

For a given function <img src="docs/math/rho.png" alt="rho" height="30"/>, the solution $\Phi(x, y, z)$ of the Poisson equation $\nabla^2 \Phi = \rho$ with vanishing Dirichlet boundary conditions at infinity is

$$
\Phi({x, y, z}) = \int d^3r' \frac{\rho({\bf r'})}{|{\bf r - r'}|}
$$

If you need to evaluate $\Phi$ at many points, calculating the integral for each point is computationally expensive. As a faster alternative, we can express $\Phi$ in terms of the multipole moments $q_{lm}$:

$$
\Phi({x, y, z}) = 
\sum_{l=0}^{\infty}\underbrace{
\sqrt{\frac{4\pi}{2l+1}}
\sum_{m=-l}^l
Y_{lm}(\theta, \varphi)\frac{q_{lm}}{r^{l+1}}
}_{\Phi^{(l)}}
$$

where $r, \theta, \varphi$ are the usual spherical coordinates corresponding to the cartesian coordinates $x, y, z$ and $Y_{lm}(\theta, \varphi)$ are the spherical harmonics.

The multipole moments are:

$$
q_{lm} =
\sqrt{\frac{4\pi}{2l+1}}
\int d^3r' \rho({\bf r'}) r'^l Y_{lm}^*(\theta', \varphi')
$$

This approach is usually much faster because the contributions to $\Phi^{(l)}$ are getting smaller with increasing $l$. So we just have to calculate a few integrals for obtaining some $q_{lm}$.


```python
from multipoles import MultipoleExpansion
```


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    <ipython-input-2-b7d1ad316502> in <module>()
    ----> 1 from multipoles import MultipoleExpansion
    

    ImportError: cannot import name 'MultipoleExpansion'

