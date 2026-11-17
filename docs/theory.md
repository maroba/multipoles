# Theory

## The Multipole Expansion

For a given charge density $\rho(\mathbf{r})$, the solution $\Phi(\mathbf{r})$ of the Poisson equation
$\nabla^2\Phi=-4\pi \rho$ with vanishing Dirichlet boundary conditions at infinity is

$$\Phi(\mathbf{r})=\int d^3r'\frac{\rho(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|}$$

Examples of this are the electrostatic and the Newtonian gravitational potential (see
[Units and Gravity](usage.md#units-and-gravity) for the constants). If you need to evaluate
$\Phi(\mathbf{r})$ at many points, calculating the integral for each point is computationally expensive. As a
faster alternative, we can express $\Phi$ in terms of the multipole moments $q_{lm}$ or $I_{lm}$ (note some
literature uses the subscripts $(\cdot)_{nm}$):

$$\Phi(\mathbf{r})=\sum_{l=0}^\infty\underbrace{\sqrt{\frac{4\pi}{2l+1}}\sum_{m=-l}^lY_{lm}(\theta, \varphi)\frac{q_{lm}}{r^{l+1}}}_{\Phi^{(l)}}$$

for an exterior expansion, or

$$\Phi(\mathbf{r})=\sum_{l=0}^\infty\underbrace{\sqrt{\frac{4\pi}{2l+1}}\sum_{m=-l}^lY_{lm}(\theta, \varphi)I_{lm}r^{l}}_{\Phi^{(l)}}$$

for an interior expansion. Here $r, \theta, \varphi$ are the spherical coordinates of $\mathbf{r}$ relative to
the expansion center (by default the center of charge, weighted by $|\rho|$), and $Y_{lm}(\theta, \varphi)$ are
the spherical harmonics.

The multipole moments for the exterior expansion are

$$q_{lm} = \sqrt{\frac{4\pi}{2l+1}}\int d^3 r' \rho(\mathbf{r}')r'^l Y^*_{lm}(\theta', \varphi')$$

and the multipole moments for the interior expansion are

$$I_{lm} = \sqrt{\frac{4\pi}{2l+1}}\int d^3 r' \frac{\rho(\mathbf{r}')}{r'^{l+1}} Y^*_{lm}(\theta', \varphi')$$

For point charges, the integrals become sums over the charges.

This approach is usually much faster, because the contributions $\Phi^{(l)}$ get smaller with increasing $l$.
So we only have to calculate a few integrals to obtain some $q_{lm}$ or $I_{lm}$, and can then evaluate the
potential anywhere at the cost of a short sum.

Some literature considers the factor $\sqrt{\frac{4\pi}{2l+1}}$ as part of the definition of
$Y_{lm}(\theta, \varphi)$ (Racah normalization).

## Where the Expansion Converges

The expansion is based on

$$\frac{1}{|\mathbf{r}-\mathbf{r}'|} = \sum_{l=0}^\infty \frac{r_<^l}{r_>^{l+1}} P_l(\cos\gamma)$$

where $r_<$ and $r_>$ are the smaller and larger of $r$ and $r'$, and $\gamma$ is the angle between
$\mathbf{r}$ and $\mathbf{r}'$. The exterior expansion uses $r_> = r$ for all charges. It therefore converges
only *outside* the smallest sphere about the expansion center that contains all charge, with radius
$R = \max r'$. Likewise, the interior expansion converges only *inside* the largest sphere that contains no
charge, with radius $R = \min r'$. *multipoles* stores $R$ as `validity_radius` and warns when you evaluate the
expansion on the wrong side of this sphere.

## Accuracy

Since $|P_l| \le 1$, a charge $q$ at distance $r'$ from the center contributes at most
$|q|\, r_<^{l}/r_>^{l+1}$ to the term of order $l$. Summing the geometric series of the omitted terms, the
error of the exterior expansion truncated after $l_{max}$ is bounded by

$$\left|\Phi(\mathbf{r}) - \sum_{l=0}^{l_{max}} \Phi^{(l)}(\mathbf{r})\right| \le
\int d^3r' \frac{|\rho(\mathbf{r}')|}{r - r'} \left(\frac{r'}{r}\right)^{l_{max}+1}
\le \frac{Q}{r - R} \left(\frac{R}{r}\right)^{l_{max}+1}$$

where $Q = \int d^3r' |\rho(\mathbf{r}')|$ is the total absolute charge. For the interior expansion, swap
$r$ and $r'$, and $r$ and $R$. So the error decreases geometrically with $l_{max}$, and the faster, the
farther away from the sphere of radius $R$ you are. Each additional order gains a factor $R/r$ in accuracy.

`error_bound` computes the first, sharper bound. For speed, it bins the absolute charge into 512 shells by
the distance from the center, and counts the charge of each shell at the shell radius farthest from the region
of convergence, so the result stays an upper bound. For continuous distributions, it ignores the fraction
`charge_tol` of the charge beyond `validity_radius`.

With `l_max='auto'`, *multipoles* chooses the smallest $l_{max}$ for which this bound at the distance
`radius` is at most `tol` times $\int d^3r' |\rho(\mathbf{r}')| / r_>$, the potential the absolute charge
would have there. Since the bound decreases with the distance from the validity sphere, it then holds
everywhere beyond `radius` (exterior) or within `radius` (interior). The bound is rigorous, so it is often
pessimistic: the actual error is typically one to three orders of magnitude smaller (see the
[accuracy notebook](accuracy.md#the-error-bound)).

For continuous charge distributions on a grid, two more sources of error come in:

- **Resolution:** the moments are integrated numerically with Simpson's rule, so the grid must resolve the
  charge density well.
- **Box size:** charge outside of the grid is missing, so the density must have decayed at the boundary of
  the grid. Also, the expansion is only valid outside of the charge distribution, so if you need the
  potential on the boundary of the grid, the grid must be large enough for the charge to be well inside.

The [accuracy notebook](accuracy.md) shows how the error behaves for some gaussian charge distributions.

## The Field and the Hessian

The field $\mathbf{E} = -\nabla\Phi$ is computed analytically, using the ladder relations of the solid harmonics.
With the Racah-normalized spherical harmonics $C_{lm} = \sqrt{\frac{4\pi}{2l+1}} Y_{lm}$, the irregular solid
harmonics $I_{lm} = C_{lm} / r^{l+1}$ of the exterior expansion obey

$$
\begin{aligned}
\partial_z I_{lm} &= -\sqrt{(l+m+1)(l-m+1)}\, I_{l+1,m} \\
(\partial_x + i\partial_y) I_{lm} &= \sqrt{(l+m+1)(l+m+2)}\, I_{l+1,m+1} \\
(\partial_x - i\partial_y) I_{lm} &= -\sqrt{(l-m+1)(l-m+2)}\, I_{l+1,m-1}
\end{aligned}
$$

and the regular solid harmonics $R_{lm} = r^l C_{lm}$ of the interior expansion obey

$$
\begin{aligned}
\partial_z R_{lm} &= \sqrt{(l+m)(l-m)}\, R_{l-1,m} \\
(\partial_x + i\partial_y) R_{lm} &= \sqrt{(l-m)(l-m-1)}\, R_{l-1,m+1} \\
(\partial_x - i\partial_y) R_{lm} &= -\sqrt{(l+m)(l+m-1)}\, R_{l-1,m-1}
\end{aligned}
$$

Unlike derivatives in spherical coordinates, these relations have no singularities on the z-axis.
With $\partial_x = \frac{1}{2}(D_+ + D_-)$ and $\partial_y = \frac{1}{2i}(D_+ - D_-)$, where
$D_\pm = \partial_x \pm i\partial_y$, each derivative maps an expansion in solid harmonics to another one.
The Hessian $\partial_i\partial_j\Phi$ applies them twice.

## Cartesian Moments

The total charge $q$, the dipole moment $\mathbf{p}$ and the traceless quadrupole moment $Q_{ij}$,

$$q = \int d^3r' \rho(\mathbf{r}'), \quad
p_i = \int d^3r' \rho(\mathbf{r}')\, x_i', \quad
Q_{ij} = \int d^3r' \rho(\mathbf{r}') \left(3 x_i' x_j' - r'^2 \delta_{ij}\right)$$

give the first terms of the exterior expansion in cartesian form,

$$\Phi(\mathbf{r}) = \frac{q}{r} + \frac{\mathbf{p}\cdot\mathbf{r}}{r^3}
+ \frac{1}{2}\sum_{ij} \frac{Q_{ij}\, x_i x_j}{r^5} + \dots$$

They are related to the spherical moments by

$$
\begin{aligned}
q &= q_{00}, \\
p_x &= -\sqrt{2}\,\mathrm{Re}\, q_{11}, \quad p_y = \sqrt{2}\,\mathrm{Im}\, q_{11}, \quad p_z = q_{10}, \\
Q_{zz} &= 2 q_{20}, \quad Q_{xz} = -\sqrt{6}\,\mathrm{Re}\, q_{21}, \quad Q_{yz} = \sqrt{6}\,\mathrm{Im}\, q_{21}, \\
Q_{xx} - Q_{yy} &= 2\sqrt{6}\,\mathrm{Re}\, q_{22}, \quad Q_{xy} = -\sqrt{6}\,\mathrm{Im}\, q_{22}.
\end{aligned}
$$

In general, the term of order $l$ of the exterior expansion is

$$\Phi^{(l)}(\mathbf{r}) = \frac{1}{l!} \sum_{i_1 \dots i_l} M^{(l)}_{i_1 \dots i_l}
\frac{x_{i_1} \cdots x_{i_l}}{r^{2l+1}}, \qquad
M^{(l)} = (2l-1)!! \int d^3r' \rho(\mathbf{r}') \left[\mathbf{r}'^{\otimes l}\right]_\text{traceless}$$

with the symmetric traceless tensor $M^{(l)}$ of rank $l$, available as `cartesian_moment(l)`:
$M^{(0)} = q$, $M^{(1)} = \mathbf{p}$, $M^{(2)} = Q$, $M^{(3)}$ is the octupole moment, etc. Since
$r^{2l+1}\Phi^{(l)} = \sum_m q_{lm} R_{lm}(\mathbf{r})$ is a harmonic polynomial of degree $l$, $M^{(l)}$
is its $l$-th derivative, which *multipoles* computes with the ladder relations above. For the interior
expansion, `cartesian_moment(l)` is the same derivative of $\Phi^{(l)}$, i.e. the tensor of the $l$-th
derivatives of the potential at the center.

All moments are taken about the expansion center. Except for the lowest non-vanishing one, they change when you
choose a different center.

## Other Conventions

With the orthonormal spherical harmonics $Y_{lm}$ instead of $C_{lm}$, e.g. as in Jackson's *Classical
Electrodynamics*, the moments are $\sqrt{(2l+1)/4\pi}\, q_{lm}$. The real spherical harmonics give the real
moments $q_{l0}$, $\sqrt{2}(-1)^m\,\mathrm{Re}\, q_{lm}$ (cosine part) and
$\sqrt{2}(-1)^{m+1}\,\mathrm{Im}\, q_{lm}$ (sine part) for $m > 0$, which `moments(real=True)` stores at
`[l, m]` and `[l, -m]`. In Racah normalization, the real dipole moments are $(p_x, p_y, p_z)$.

The power spectrum $\sqrt{\sum_m |q_{lm}|^2}$ does not change when the charge distribution is rotated about the
expansion center, because rotations mix the $C_{lm}$ of each $l$ unitarily.

## Translation

The regular solid harmonics obey the addition theorem

$$R_{lm}(\mathbf{a} + \mathbf{b}) = \sum_{\lambda=0}^{l} \sum_{\mu=-\lambda}^{\lambda}
\sqrt{\binom{l+m}{\lambda+\mu}\binom{l-m}{\lambda-\mu}}\, R_{\lambda\mu}(\mathbf{a})\,
R_{l-\lambda,m-\mu}(\mathbf{b})$$

`shifted` uses it for expansions without charges, e.g. from `from_moments`. Moving the center by $\mathbf{d}$,
the exterior moments about the new center are

$$q'_{lm} = \sum_{\lambda\mu} \sqrt{\binom{l+m}{\lambda+\mu}\binom{l-m}{\lambda-\mu}}\, q_{\lambda\mu}\,
R^*_{l-\lambda,m-\mu}(-\mathbf{d})$$

which only needs the moments up to order $l$, so the translated moments are exact. The sphere containing all
charge grows by $|\mathbf{d}|$. For the interior expansion,

$$I'_{\lambda\mu} = \sum_{l \ge \lambda} \sum_m \sqrt{\binom{l+m}{\lambda+\mu}\binom{l-m}{\lambda-\mu}}\,
I_{lm}\, R_{l-\lambda,m-\mu}(\mathbf{d})$$

needs moments of all orders. With the moments up to $l_{max}$, it gives the same truncated expansion (a
polynomial of degree $l_{max}$) about the new center, with the same error, valid in a sphere smaller by
$|\mathbf{d}|$. Expansions that know their charges simply recompute the moments.

## Numerical Methods

- **Spherical harmonics** are computed with the standard three-term recurrence of the associated Legendre
  functions, which is stable to high orders and much faster than evaluating each $Y_{lm}$ separately.
- **Grid integrals** are weighted sums with the weights of the composite Simpson rule along each axis, for
  evenly or unevenly spaced points. For an even number of points, the last interval is integrated with a
  parabola through the last three points, as in `scipy.integrate.simpson`. Grid points without charge are
  skipped, and the moments are accumulated in chunks of points to limit the memory needed.
- **Spherical coordinates** use $\theta = \operatorname{atan2}(\sqrt{x^2+y^2}, z)$, which, unlike
  $\arccos(z/r)$, is accurate close to the z-axis.
- Only the harmonics with $m \ge 0$ are computed; those with negative $m$ follow from
  $C_{l,-m} = (-1)^m C_{lm}^*$. Since the potential is real, $q_{l,-m} = (-1)^m q_{lm}^*$, so the terms
  with $\pm m$ add up to $2\,\mathrm{Re}(q_{lm} C_{lm})$.
