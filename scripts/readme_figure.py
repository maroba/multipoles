"""Create the figure at the top of the README, in a light and a dark variant.

Run from the repository root:

    python scripts/readme_figure.py

It writes docs/images/readme-light.png and docs/images/readme-dark.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm  # noqa: E402

from multipoles import MultipoleExpansion  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "diverging": ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f2b3ae", "#e34948", "#9e2422"],
        "field_lines": "#52514e",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "text_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "diverging": ["#cde2fb", "#6da7ec", "#1c5cab", "#383835", "#a8322f", "#e66767", "#f6c4c1"],
        "field_lines": "#c3c2b7",
    },
}


def charge_distribution():
    """A neutral cluster of positive and negative point charges, so that the dipole and
    higher moments shape the potential."""
    rng = np.random.default_rng(7)
    xyz = rng.normal(scale=0.8, size=(12, 3))
    xyz[:, 2] *= 0.3  # flattened, so that the slice through the center shows the structure
    xyz[:6, 0] += 0.6  # positive charges on the right, negative ones on the left
    xyz[6:, 0] -= 0.6
    q = np.where(np.arange(12) < 6, 1.0, -1.0) * rng.uniform(0.6, 1.4, 12)
    q -= q.mean()
    return q, xyz


def exact_potential(q, xyz, points):
    d = np.linalg.norm(points[:, None, :] - xyz[None, :, :], axis=-1)
    return np.sum(q / d, axis=1)


def draw(theme_name):
    t = THEMES[theme_name]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "text.color": t["text"],
            "axes.labelcolor": t["text_secondary"],
            "axes.edgecolor": t["axis"],
            "xtick.color": t["muted"],
            "ytick.color": t["muted"],
            "axes.facecolor": t["surface"],
            "figure.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
        }
    )

    q, xyz = charge_distribution()
    phi = MultipoleExpansion.from_charges(q, xyz, l_max=10)
    R = phi.validity_radius
    c = phi.center

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})

    # --- Left: potential and field lines on the plane z = center_z -------------------
    extent = 3.2 * R
    u = np.linspace(c[0] - extent, c[0] + extent, 400)
    v = np.linspace(c[1] - extent, c[1] + extent, 400)
    U, V = np.meshgrid(u, v, indexing="xy")
    points = np.stack([U, V, np.full_like(U, c[2])], axis=-1)
    valid = phi.is_valid(points)

    potential = np.full(U.shape, np.nan)
    potential[valid] = phi(points[valid])
    E = np.full(U.shape + (3,), np.nan)
    E[valid] = phi.field(points[valid])

    vmax = np.nanmax(np.abs(potential))
    cmap = LinearSegmentedColormap.from_list("diverging", t["diverging"])
    cmap.set_bad(t["surface"])
    norm = SymLogNorm(linthresh=vmax / 30, vmin=-vmax, vmax=vmax)
    ax1.pcolormesh(U, V, potential, cmap=cmap, norm=norm, shading="auto", rasterized=True)
    ax1.streamplot(u, v, E[..., 0], E[..., 1], color=t["field_lines"], linewidth=0.6, density=1.1, arrowsize=0.7)

    circle = plt.Circle(c[:2], R, fill=False, linestyle=(0, (4, 3)), linewidth=1.2, color=t["muted"])
    ax1.add_patch(circle)
    for sign, color in [(1, t["diverging"][-2]), (-1, t["diverging"][1])]:
        sel = np.sign(q) == sign
        ax1.scatter(xyz[sel, 0], xyz[sel, 1], s=22, color=color, edgecolor=t["surface"], linewidth=1, zorder=3)
    ax1.text(c[0], c[1] + 1.12 * R, "validity radius", ha="center", va="bottom", color=t["text_secondary"],
             fontsize=10, bbox={"boxstyle": "round,pad=0.25", "facecolor": t["surface"], "edgecolor": "none"})

    ax1.set_xlim(u[0], u[-1])
    ax1.set_ylim(v[0], v[-1])
    ax1.set_aspect("equal")
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_title("Potential and field of 12 charges, total charge 0", loc="left", color=t["text"], fontsize=12, pad=10)

    # --- Right: error against the distance from the center -----------------------------
    r = np.logspace(np.log10(1.05 * R), np.log10(12 * R), 60)
    direction = np.array([0.6, 0.48, 0.64])
    line_points = c + r[:, None] * direction / np.linalg.norm(direction)
    exact = exact_potential(q, xyz, line_points)
    scale = np.sum(np.abs(q)) / r

    for l_max, color in zip([2, 5, 10], t["series"]):
        error = np.abs(phi(line_points, l_max=l_max) - exact) / scale
        bound = phi.error_bound(line_points, l_max=l_max) / scale
        ax2.loglog(r / R, bound, color=color, linewidth=1.2, linestyle=(0, (4, 3)), alpha=0.8)
        ax2.loglog(r / R, error, color=color, linewidth=2)
        ax2.annotate(f"l_max = {l_max}", xy=(r[-1] / R, error[-1]), xytext=(6, 0), textcoords="offset points",
                     va="center", color=t["text_secondary"], fontsize=10)

    ax2.plot([], [], color=t["text_secondary"], linewidth=2, label="actual error")
    ax2.plot([], [], color=t["text_secondary"], linewidth=1.2, linestyle=(0, (4, 3)), label="error_bound()")
    legend = ax2.legend(loc="lower left", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(t["text_secondary"])

    ax2.set_xlim(1, 12 * 1.9)
    ax2.set_ylim(1e-14, 3)
    ax2.grid(True, which="major", color=t["grid"], linewidth=0.8)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.set_xlabel("distance from the center / validity radius")
    ax2.set_ylabel("relative error of the potential")
    ax2.set_title("Accuracy grows quickly with l_max and distance", loc="left", color=t["text"], fontsize=12,
                  pad=10)

    fig.tight_layout(w_pad=3)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"readme-{theme_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for name in THEMES:
        print(draw(name))
