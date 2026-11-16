import numpy as np

from multipoles import MultipoleExpansion


def coulomb_potential(qs, xyzs, points):
    return np.array([np.sum(qs / np.linalg.norm(p - xyzs, axis=1)) for p in points])


def test_discrete_multipole_far_from_origin():
    # ten charges in a unit cube far away from the origin, evaluated along the diagonal
    rng = np.random.default_rng(42)
    qs = np.ones(10)
    qs[5:] = -1
    xyzs = rng.random((10, 3)) + np.array([215, 5, 23])

    phi = MultipoleExpansion.from_charges(qs, xyzs, l_max=30)

    ds = np.logspace(1, 3, 20)
    points = np.stack([ds, ds, ds], axis=1)
    expected = coulomb_potential(qs, xyzs, points)
    errors = np.abs((expected - phi(points)) / expected)

    assert np.max(errors) < 1e-11
