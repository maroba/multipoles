"""Rectilinear grids for continuous charge distributions and integration weights on them."""

from __future__ import annotations

import numpy as np


class GridError(ValueError):
    """Raised for invalid grids. Converted to InvalidChargeDistributionException by the caller."""


class Grid:
    """A rectilinear grid with the axes x, y, z, which may be unevenly spaced.

    Attributes:
        axes: tuple of the 1D coordinate arrays along each axis.
        shape: the shape of the grid.
        coords: tuple of the 3D coordinate arrays (read-only broadcast views).
    """

    def __init__(self, axes):
        self.axes = tuple(np.asarray(a, dtype=float) for a in axes)
        self.shape = tuple(a.size for a in self.axes)
        self.coords = tuple(
            np.broadcast_to(a.reshape([-1 if k == i else 1 for k in range(3)]), self.shape)
            for i, a in enumerate(self.axes)
        )

    @property
    def uniform(self):
        return all(np.allclose(np.diff(a), a[1] - a[0], rtol=1e-8, atol=0) for a in self.axes)

    def simpson_weights(self):
        """Weights w on the grid such that sum(w * f) is the Simpson integral of f."""
        wx, wy, wz = (simpson_weights(a) for a in self.axes)
        return wx[:, None, None] * wy[None, :, None] * wz[None, None, :]

    def volume_weights(self):
        """Volume of the cell around each grid point (trapezoid weights)."""
        wx, wy, wz = (trapezoid_weights(a) for a in self.axes)
        return wx[:, None, None] * wy[None, :, None] * wz[None, None, :]

    def boundary_mask(self, width=1):
        """Boolean array that is True for the points within `width` points of the boundary."""
        mask = np.ones(self.shape, dtype=bool)
        if all(2 * width < n for n in self.shape):
            mask[width:-width, width:-width, width:-width] = False
        return mask

    def distance_to_boundary(self, point):
        """Distance from a point inside the grid box to the closest point of its surface,
        or None if the point lies outside the box."""
        distances = []
        for a, p in zip(self.axes, point):
            lo, hi = min(a[0], a[-1]), max(a[0], a[-1])
            if not lo < p < hi:
                return None
            distances += [p - lo, hi - p]
        return min(distances)


def parse_grid(rho, xyz):
    """Validate a charge density and its grid.

    Args:
        rho: the charge density, a 3D array.
        xyz: either the three 1D axes (x, y, z), or three 3D coordinate arrays as returned
            by np.meshgrid(x, y, z, indexing='ij'), possibly with sparse=True.

    Returns:
        (rho, Grid)
    """
    rho = np.asarray(rho)
    if rho.ndim != 3:
        raise GridError("The charge density 'rho' must be a 3D array.")
    if np.iscomplexobj(rho):
        raise GridError("The charge density 'rho' must be real.")
    if min(rho.shape) < 2:
        raise GridError("The grid must have at least 2 points along each axis.")

    try:
        coords = [np.asarray(c, dtype=float) for c in xyz]
    except (TypeError, ValueError):
        raise GridError(
            "The grid must be given as three 1D axes (x, y, z) or as three 3D coordinate "
            "arrays from np.meshgrid(x, y, z, indexing='ij')."
        ) from None
    if len(coords) != 3:
        raise GridError("The grid must have three coordinate arrays (x, y, z).")

    if all(c.ndim == 1 for c in coords):
        axes = coords
        if tuple(a.size for a in axes) != rho.shape:
            raise GridError(
                f"The lengths of the axes {tuple(a.size for a in axes)} do not match the shape {rho.shape} of 'rho'."
            )
    elif all(c.ndim == 3 for c in coords):
        try:
            coords = np.broadcast_arrays(*coords)
        except ValueError:
            coords = None
        if coords is None or any(c.shape != rho.shape for c in coords):
            raise GridError(
                "'rho' and the coordinate arrays in 'xyz' must be 3D arrays of the same shape."
            )
        axes = [coords[0][:, 0, 0], coords[1][0, :, 0], coords[2][0, 0, :]]
        for i, (c, a) in enumerate(zip(coords, axes)):
            shape = [1, 1, 1]
            shape[i] = -1
            if not np.allclose(c, a.reshape(shape), rtol=1e-10, atol=0):
                raise GridError(
                    "The coordinate arrays do not form a rectilinear grid. Create them with "
                    "np.meshgrid(x, y, z, indexing='ij'), or pass the axes x, y, z directly."
                )
    else:
        raise GridError(
            "The grid must be given as three 1D axes (x, y, z) or as three 3D coordinate "
            "arrays from np.meshgrid(x, y, z, indexing='ij')."
        )

    for name, a in zip("xyz", axes):
        if not np.all(np.isfinite(a)):
            raise GridError(f"The {name}-axis of the grid contains non-finite values.")
        steps = np.diff(a)
        if np.any(steps == 0):
            raise GridError(
                f"Grid spacing is zero along the {name}-axis. Create the coordinate arrays with "
                "np.meshgrid(..., indexing='ij'), or pass the axes x, y, z directly."
            )
        if not (np.all(steps > 0) or np.all(steps < 0)):
            raise GridError(
                f"The coordinates along the {name}-axis must be strictly increasing or "
                "decreasing."
            )

    return rho, Grid(axes)


def simpson_weights(x):
    """Weights w such that sum(w * f) is the composite Simpson integral of f over x.

    The points may be unevenly spaced, and ascending or descending. For an odd number of
    intervals, the last interval is integrated with a parabola through the last three points
    (Cartwright's correction), as in scipy.integrate.simpson since SciPy 1.11.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    h = np.abs(np.diff(x))
    w = np.zeros(n)
    if n == 2:
        w[:] = h[0] / 2
        return w

    n_pairs = (n - 1) // 2
    for k in range(n_pairs):
        h0, h1 = h[2 * k], h[2 * k + 1]
        s = h0 + h1
        w[2 * k] += s / 6 * (2 - h1 / h0)
        w[2 * k + 1] += s / 6 * s**2 / (h0 * h1)
        w[2 * k + 2] += s / 6 * (2 - h0 / h1)

    if (n - 1) % 2:
        # Integral over the last interval of the parabola through the last three points
        h0, h1 = h[-2], h[-1]
        w[-1] += (2 * h1**2 + 3 * h0 * h1) / (6 * (h0 + h1))
        w[-2] += (h1**2 + 3 * h0 * h1) / (6 * h0)
        w[-3] -= h1**3 / (6 * h0 * (h0 + h1))
    return w


def trapezoid_weights(x):
    """Weights w such that sum(w * f) is the trapezoidal integral of f over x."""
    h = np.abs(np.diff(np.asarray(x, dtype=float)))
    w = np.zeros(h.size + 1)
    w[:-1] += h / 2
    w[1:] += h / 2
    return w
