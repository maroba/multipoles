# Theory

For a given function $\rho(x,y,z)$, the solution $\Phi(x,y,z)$ of the Poisson equation $\nabla^2\Phi=-4\pi \rho$ with vanishing Dirichlet boundary conditions at infinity is

$$\Phi(x,y,z)=\int d^3r'\frac{\rho(r')}{|r-r'|}$$

Examples of this are the electrostatic and Newtonian gravitational potential.
If you need to evaluate $\Phi(x,y,z)$ at many points, calculating the integral for each point is computationally expensive. As a faster alternative, we can express $\Phi(x,y,z)$ in terms of the multipole moments $q_{lm}$ or $I_{lm}$ (note some literature uses the subscripts $(\cdot)_{nm}$):

$$\Phi(x,y,z)=\sum_{l=0}^\infty\underbrace{\sqrt{\frac{4\pi}{2l+1}}\sum_{m=-l}^lY_{lm}(\theta, \varphi)\frac{q_{lm}}{r^{l+1}}}_{\Phi^{(l)}}$$

for a exterior expansion, or

$$\Phi(x,y,z)=\sum_{l=0}^\infty\underbrace{\sqrt{\frac{4\pi}{2l+1}}\sum_{m=-l}^lY_{lm}(\theta, \varphi)I_{lm}r^{l}}_{\Phi^{(l)}}$$

for an interior expansion; where $r, \theta, \varphi$ are the usual spherical coordinates corresponding to the cartesian coordinates $x, y, z$ and $Y_{lm}(\theta, \varphi)$ are the spherical harmonics.

The multipole moments for the exterior expansion are:

$$q_{lm} = \sqrt{\frac{4\pi}{2l+1}}\int d^3 r' \rho(r')r'^l Y^*_{lm}(\theta', \varphi')$$

and the multipole moments for the interior expansion are:

$$I_{lm} = \sqrt{\frac{4\pi}{2l+1}}\int d^3 r' \frac{\rho(r')}{r'^{l+1}} Y^*_{lm}(\theta', \varphi')$$

This approach is usually much faster because the contributions $\Phi^{(l)}$ are getting smaller with increasing <i>l</i>. So we just have to calculate a few integrals for obtaining some  $q_{lm}$ or  $I_{lm}$.

Some literature considers the $\sqrt{\frac{4\pi}{2l+1}}$ as part of the definition of $Y_{lm}(\theta, \varphi)$.
