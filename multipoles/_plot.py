"""Plotting with matplotlib, which is an optional dependency."""

from __future__ import annotations

import numpy as np

_AXES = {"x": 0, "y": 1, "z": 2}


def plot_slice(expansion, normal, offset, extent, resolution, field, ax, l_max, **kwargs):
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise ImportError(
            "Plotting needs matplotlib: pip install matplotlib, or pip install multipoles[plot]"
        ) from None

    try:
        k = _AXES[normal]
    except KeyError:
        raise ValueError("'normal' must be 'x', 'y' or 'z'.") from None
    i, j = [a for a in range(3) if a != k]
    center = expansion.center

    if extent is None:
        radius = expansion.validity_radius
        if not np.isfinite(radius) or radius == 0:
            radius = 1.0
        extent = 4 * radius if expansion.exterior else radius
    if offset is None:
        offset = center[k]

    u = np.linspace(center[i] - extent, center[i] + extent, resolution)
    v = np.linspace(center[j] - extent, center[j] + extent, resolution)
    U, V = np.meshgrid(u, v, indexing="xy")
    points = np.empty(U.shape + (3,))
    points[..., i], points[..., j], points[..., k] = U, V, offset

    valid = expansion.is_valid(points)
    potential = np.full(U.shape, np.nan)
    potential[valid] = expansion(points[valid], l_max=l_max)

    if ax is None:
        ax = plt.gca()
    kwargs.setdefault("shading", "auto")
    mesh = ax.pcolormesh(U, V, potential, **kwargs)
    ax.figure.colorbar(mesh, ax=ax, label="potential")

    if field:
        E = np.full(U.shape + (3,), np.nan)
        E[valid] = expansion.field(points[valid], l_max=l_max)
        ax.streamplot(u, v, E[..., i], E[..., j], color="k", linewidth=0.6, density=1.2)

    ax.set_xlabel("xyz"[i])
    ax.set_ylabel("xyz"[j])
    ax.set_aspect("equal")
    ax.set_title(f"{normal} = {offset:.4g}")
    return ax
