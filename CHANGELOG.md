# Changelog

All notable changes to *multipoles* are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- `CITATION.cff`, so that GitHub shows how to cite *multipoles*.

### Changed

- A new README with a figure, the typical uses, and sections on related projects and citing.

- The documentation is built with Zensical instead of Material for MkDocs, which reaches its end of life in
  May 2027. The accuracy notebook is converted to a page with nbconvert.
- The GitHub Actions use the current major versions, which run on Node.js 24 instead of the deprecated Node.js 20.

## [0.6.0] - 2026-11-16

### Added

- `MultipoleExpansion.from_charges(q, xyz, l_max)` and `MultipoleExpansion.from_density(rho, (x, y, z), l_max)`:
  create expansions from arrays instead of dicts. For many point charges, this is also much faster.
- Charge densities on unevenly spaced grids, and grids given by their 1D axes or by sparse meshgrids.
- Points can be given as one array of shape `(..., 3)` everywhere instead of the coordinates `x, y, z`.
- `hessian(x, y, z)`: the second derivatives of the potential, computed analytically.
- `evaluate(x, y, z, field=True, hessian=False)`: potential and derivatives at once, faster than separately.
- `contributions(x, y, z)`: the contribution of each order l to the potential.
- `field_on_grid(key)` and `boundary_values(width=1)`: the field on the grid, and the potential on the boundary of
  the grid, e.g. as boundary condition for Poisson solvers.
- `prefactor` argument: `'SI'` for electrostatics in SI units, `'gravity'` for the gravitational potential, or any
  number. The constants `EPSILON_0` and `GRAVITATIONAL_CONSTANT`.
- `error_bound(x, y, z)`: an upper bound of the truncation error, and `l_max='auto'` with `tol` and `radius`
  to choose the smallest `l_max` for a given accuracy.
- `cartesian_moment(l)`: cartesian moments of any order, e.g. the octupole moment.
- `moments(normalization, real)`: the moments for orthonormal or real spherical harmonics, and
  `power_spectrum()`: the rotation-invariant strength of each order.
- `shifted(center)`: the same expansion about another center. Expansions can be added, subtracted and multiplied
  by numbers.
- `from_moments(moments, center)`: create an expansion from known moments. `save`/`load` and
  `to_dict`/`from_dict` to store expansions.
- `energy(q, xyz)`: the energy of point charges in the potential of the expansion.
- `plot_slice()`: plot the potential and the field lines on a plane, with the new extra `multipoles[plot]`.
- `py.typed`, so that type checkers use the type hints.

### Changed

- Requires Python >= 3.10 and NumPy >= 1.22, and no longer SciPy.
- Evaluating the expansion and computing the moments is about twice as fast. The moments of large grids are
  computed in chunks, which needs much less memory for large `l_max`.
- The center of charge of continuous distributions weights the grid points by their cell volume.
- `dxyz` and `dvol` are `None` for unevenly spaced grids.
- Expansions created with `from_charges`, `from_density` or `from_moments` have `charge_dist = None`.

### Fixed

- Grids with uneven spacing were integrated with the spacing of the first cell, which gave wrong moments without
  any warning. Coordinate arrays that don't form a rectilinear grid now raise an error.
- The potential lost accuracy close to the z-axis (relative error up to about 1e-11), because the polar angle
  was computed with `arccos`.
- `l_max=2.0` failed with a `TypeError`, and an invalid `l_max` in calls gave a cryptic error.
- Clear errors for an empty list of charges, and for points of the wrong shape.

## [0.5.0] - 2026-10-22

### Added

- `MultipoleExpansion.field(x, y, z)`: the field E = -∇Φ, computed analytically.
- `center` argument to expand about any point instead of the center of charge. This makes interior
  expansions possible where the center of charge coincides with a charge.
- `validity_radius` and `is_valid(x, y, z)`: where the expansion converges.
- `ExpansionValidityWarning` when evaluating the expansion where it does not converge, and the
  `charge_tol` argument to control how much of a continuous charge density may be neglected for this check.
- `dipole_moment` and `quadrupole_moment`: the cartesian dipole vector and traceless quadrupole tensor.
- `total_charge` for discrete charge distributions (previously only for continuous ones).
- `multipole_moments_array`: the multipole moments as array, indexed by `[l, m]`.
- A readable `repr`, type hints for the public API, and `ExpansionValidityWarning`,
  `InvalidChargeDistributionException` and `InvalidExpansionException` exported from `multipoles`.

### Changed

- Much faster: computing the moments of a 101³ grid takes 0.17 s instead of 2.4 s for `l_max=4`, and
  0.3 s instead of 24 s for `l_max=10`. Evaluating the expansion is about 20 times faster.
- Evaluating the potential at points inside the charge distribution (exterior expansion) or outside of it
  (interior expansion) now emits an `ExpansionValidityWarning`. This includes evaluating on the whole grid
  with `Phi[:, :, :]`. The results there were never meaningful.
- `total_charge` of continuous charge distributions is now integrated with Simpson's rule, consistent with the
  multipole moments, instead of a plain sum over the grid.
- *multipoles* no longer uses `scipy.special` for the spherical harmonics.

## [0.4.2] - 2026-10-22

### Fixed

- Import error with SciPy >= 1.17, where `scipy.special.sph_harm` was removed.
- `Phi(x, y, z, l_max=...)` ignored `l_max`.
- `exterior=False` alone selected the exterior instead of the interior expansion.
- `Phi[...]` used the exterior expansion for interior expansions and modified internal state.
- Evaluating `Phi[...]` for discrete charges failed with an `AttributeError`; it now raises a clear `TypeError`.
- Invalid input now raises errors instead of silently producing NaN: zero total charge, charge at the center of
  an interior expansion, grids from `np.meshgrid` without `indexing='ij'`, and mismatched grid shapes.
- Grids with descending coordinates gave wrong results.
- Evaluation accepts arrays of any shape, including 0-d arrays, and is vectorized.

### Changed

- Requires Python >= 3.9.

## Earlier versions

See the [commit history](https://github.com/maroba/multipoles/commits/master) and the
[releases](https://github.com/maroba/multipoles/releases).

[Unreleased]: https://github.com/maroba/multipoles/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/maroba/multipoles/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/maroba/multipoles/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/maroba/multipoles/compare/v0.4.1...v0.4.2
