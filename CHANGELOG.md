# Changelog

All notable changes to *multipoles* are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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

[Unreleased]: https://github.com/maroba/multipoles/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/maroba/multipoles/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/maroba/multipoles/compare/v0.4.1...v0.4.2
