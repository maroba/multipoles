"""Property-based tests: the expansion against direct summation for random charge distributions."""

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra.numpy import arrays  # noqa: E402

from multipoles import MultipoleExpansion  # noqa: E402

finite = st.floats(-1, 1, allow_nan=False, allow_infinity=False)
nonzero_charge = st.floats(0.1, 2) | st.floats(-2, -0.1)


@st.composite
def charge_distributions(draw, min_size=1, max_size=8):
    n = draw(st.integers(min_size, max_size))
    q = draw(arrays(float, n, elements=nonzero_charge))
    xyz = draw(arrays(float, (n, 3), elements=finite))
    return q, xyz


@st.composite
def directions(draw):
    v = draw(arrays(float, 3, elements=st.floats(-1, 1)))
    norm = np.linalg.norm(v)
    if norm < 1e-3:
        v, norm = np.array([0.0, 0, 1]), 1.0
    return v / norm


def coulomb(q, xyz, p):
    d = p - xyz
    r = np.linalg.norm(d, axis=1)
    return np.sum(q / r), np.sum(q[:, None] * d / r[:, None] ** 3, axis=0)


SETTINGS = settings(max_examples=60, deadline=None)


@SETTINGS
@given(charge_distributions(), directions(), st.floats(2, 6))
def test_exterior_potential_and_field(dist, direction, distance_factor):
    q, xyz = dist
    mpe = MultipoleExpansion.from_charges(q, xyz, 20)
    r = distance_factor * max(mpe.validity_radius, 0.1)
    p = mpe.center + r * direction

    potential, field = coulomb(q, xyz, p)
    result = mpe.evaluate(*p)
    scale = np.sum(np.abs(q)) / r
    assert abs(result.potential - potential) <= mpe.error_bound(*p) + 1e-12 * scale
    assert np.allclose(result.field, field, atol=10 * mpe.error_bound(*p) / r + 1e-12 * scale / r)


@SETTINGS
@given(charge_distributions(), directions(), st.floats(0.05, 0.5))
def test_interior_potential(dist, direction, distance_factor):
    q, xyz = dist
    xyz = xyz + np.where(xyz >= 0, 2, -2)  # keep the charges away from the center
    mpe = MultipoleExpansion.from_charges(q, xyz, 20, interior=True, center=(0, 0, 0))
    p = distance_factor * mpe.validity_radius * direction

    potential, _ = coulomb(q, xyz, p)
    scale = np.sum(np.abs(q) / np.linalg.norm(xyz, axis=1))
    assert abs(mpe(*p) - potential) <= mpe.error_bound(*p) + 1e-12 * scale


@SETTINGS
@given(charge_distributions(), arrays(float, 3, elements=finite))
def test_shifting_moments_matches_direct_computation(dist, shift):
    q, xyz = dist
    mpe = MultipoleExpansion.from_charges(q, xyz, 6)
    new_center = mpe.center + shift
    translated = MultipoleExpansion.from_moments(mpe.multipole_moments, mpe.center).shifted(new_center)
    direct = MultipoleExpansion.from_charges(q, xyz, 6, center=new_center)
    scale = np.sum(np.abs(q)) * (1 + np.max(np.linalg.norm(xyz - new_center, axis=1))) ** np.arange(7)
    error = np.abs(translated.multipole_moments_array - direct.multipole_moments_array)
    assert np.all(error <= 1e-10 * scale[:, None])


@SETTINGS
@given(charge_distributions(min_size=2), st.floats(-3, 3))
def test_expansion_is_linear(dist, factor):
    q, xyz = dist
    half = len(q) // 2
    a = MultipoleExpansion.from_charges(q[:half], xyz[:half], 5, center=(0, 0, 0))
    b = MultipoleExpansion.from_charges(q[half:], xyz[half:], 5, center=(0, 0, 0))
    total = MultipoleExpansion.from_charges(q, xyz, 5, center=(0, 0, 0))
    combined = a + factor * b - (factor - 1) * b
    scale = np.sum(np.abs(q)) * (1 + np.max(np.linalg.norm(xyz, axis=1))) ** np.arange(6)
    error = np.abs(combined.multipole_moments_array - total.multipole_moments_array)
    assert np.all(error <= 1e-10 * (1 + abs(factor)) * scale[:, None])
