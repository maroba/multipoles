"""Spherical and solid harmonics in Racah normalization, and operations on expansions in them.

Coefficient sets are dicts {(l, m): c} that describe the function sum_lm c_lm S_lm, where
S_lm are either the regular solid harmonics R_lm = r^l C_lm or the irregular solid harmonics
I_lm = C_lm / r^(l+1), with the Racah-normalized spherical harmonics C_lm = sqrt(4 pi / (2l+1)) Y_lm.
"""

from __future__ import annotations

import math

import numpy as np


def racah_columns(l_max, theta, phi):
    """Spherical harmonics in Racah normalization, C_lm = sqrt(4 pi / (2l+1)) Y_lm.

    Yields (m, [C_mm, C_m+1,m, ..., C_l_max,m]) for m = 0, ..., l_max, computed with
    the standard three-term recurrence of the associated Legendre functions. The
    harmonics for negative m follow from C_l,-m = (-1)^m conj(C_lm).
    """
    x, s = np.cos(theta), np.sin(theta)
    e_iphi = np.exp(1j * np.asarray(phi))
    # sqrt((l-m)! / (l+m)!) P_lm(cos theta) for l = m, with Condon-Shortley phase
    c_mm = np.ones_like(x)
    e_imphi = np.ones_like(e_iphi)
    for m in range(l_max + 1):
        if m > 0:
            c_mm = -np.sqrt((2 * m - 1) / (2 * m)) * s * c_mm
            e_imphi = e_imphi * e_iphi
        column = [c_mm]
        if m < l_max:
            column.append(np.sqrt(2 * m + 1) * x * c_mm)
        for l in range(m + 2, l_max + 1):
            column.append(
                (
                    (2 * l - 1) * x * column[-1]
                    - np.sqrt((l - 1) ** 2 - m**2) * column[-2]
                )
                / np.sqrt(l**2 - m**2)
            )
        yield m, [c * e_imphi for c in column]


def cartesian_to_spherical(*coords):
    """Convert cartesian coordinates (x, y, z) to spherical coordinates (r, phi, theta)."""
    X, Y, Z = coords
    R = np.sqrt(X**2 + Y**2 + Z**2)
    R_xy = np.sqrt(X**2 + Y**2)

    # arctan2 is accurate everywhere, unlike arccos(Z / R) close to the z-axis
    Phi = np.where(R_xy == 0, 0.0, np.arctan2(Y, X))
    Theta = np.where(R == 0, np.pi / 2, np.arctan2(R_xy, Z))

    # np.where returns 0-d arrays for scalar input; turn them into scalars
    return R, Phi[()], Theta[()]


def spherical_to_cartesian(r, phi, theta):
    """Convert spherical coordinates (r, phi, theta) to cartesian coordinates (x, y, z)."""
    return (
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    )


def regular_solid_harmonics(v, l_max):
    """The regular solid harmonics R_lm(v) = |v|^l C_lm at a single point v, as dict."""
    r, phi, theta = cartesian_to_spherical(*np.asarray(v, dtype=float))
    harmonics = {}
    for m, column in racah_columns(l_max, theta, phi):
        for l, C_lm in zip(range(m, l_max + 1), column):
            R_lm = complex(r**l * C_lm)
            harmonics[(l, m)] = R_lm
            if m > 0:
                harmonics[(l, -m)] = (-1) ** m * R_lm.conjugate()
    return harmonics


def solid_sum(coeff_sets, r, phi, theta, regular, per_l=False):
    """Evaluate real functions given as coefficient sets of solid harmonics.

    The functions must be real, i.e. c_l,-m = (-1)^m conj(c_lm), so that only the
    coefficients with m >= 0 are needed.

    Args:
        coeff_sets: list of coefficient dicts {(l, m): c}.
        r, phi, theta: 1D arrays of spherical coordinates.
        regular: whether the coefficients refer to regular (r^l C_lm) or irregular
            (C_lm / r^(l+1)) solid harmonics.
        per_l: if True, return the contributions of each l separately.

    Returns:
        array of shape (len(coeff_sets), n) or, if per_l, (len(coeff_sets), l_top + 1, n),
        where l_top is the largest l in the coefficient sets.
    """
    n = np.size(r)
    l_top = max((l for coeffs in coeff_sets for (l, _) in coeffs), default=-1)
    if per_l:
        out = np.zeros((len(coeff_sets), l_top + 1, n))
    else:
        out = np.zeros((len(coeff_sets), n))
    if l_top < 0:
        return out

    # Radial factors r^l or 1 / r^(l+1)
    radial = []
    factor = np.ones_like(r) if regular else 1 / r
    for _ in range(l_top + 1):
        radial.append(factor)
        factor = factor * r if regular else factor / r

    for m, column in racah_columns(l_top, theta, phi):
        # The terms with m and -m add up to 2 Re(c_lm S_lm)
        weight = 1 if m == 0 else 2
        for l, C_lm in zip(range(m, l_top + 1), column):
            S_re = S_im = None
            for k, coeffs in enumerate(coeff_sets):
                c = coeffs.get((l, m), 0)
                if c == 0:
                    continue
                if S_re is None:
                    S = C_lm * radial[l]
                    S_re, S_im = S.real, S.imag
                c = weight * complex(c)
                term = c.real * S_re - c.imag * S_im
                if per_l:
                    out[k, l] += term
                else:
                    out[k] += term
    return out


def _ladder(coeffs, op, regular):
    """Apply d/dz (op 'z') or D+- = d/dx +- i d/dy (op '+', '-') to a coefficient set."""
    result = {}
    for (l, m), c in coeffs.items():
        if c == 0:
            continue
        if regular:
            if op == "z":
                target, f = (l - 1, m), (l + m) * (l - m)
            elif op == "+":
                target, f = (l - 1, m + 1), (l - m) * (l - m - 1)
            else:
                target, f = (l - 1, m - 1), -(l + m) * (l + m - 1)
        else:
            if op == "z":
                target, f = (l + 1, m), -(l + m + 1) * (l - m + 1)
            elif op == "+":
                target, f = (l + 1, m + 1), (l + m + 1) * (l + m + 2)
            else:
                target, f = (l + 1, m - 1), -(l - m + 1) * (l - m + 2)
        if f == 0 or target[0] < 0 or abs(target[1]) > target[0]:
            continue
        # f holds sign * (product under the square root)
        f = math.copysign(math.sqrt(abs(f)), f)
        result[target] = result.get(target, 0) + f * c
    return result


def partial_derivative(coeffs, axis, regular):
    """The coefficient set of the partial derivative d/dx_axis (axis 0, 1, 2 = x, y, z)."""
    if axis == 2:
        return _ladder(coeffs, "z", regular)
    plus = _ladder(coeffs, "+", regular)
    minus = _ladder(coeffs, "-", regular)
    keys = set(plus) | set(minus)
    if axis == 0:
        return {k: (plus.get(k, 0) + minus.get(k, 0)) / 2 for k in keys}
    return {k: (plus.get(k, 0) - minus.get(k, 0)) / 2j for k in keys}


def _addition_coefficient(l, m, lam, mu):
    """sqrt(binom(l+m, lam+mu) binom(l-m, lam-mu)) of the addition theorem
    R_lm(a + b) = sum_{lam, mu} coef * R_{lam,mu}(a) R_{l-lam,m-mu}(b)."""

    def binom(n, k):
        return math.comb(n, k) if 0 <= k <= n else 0

    return math.sqrt(binom(l + m, lam + mu) * binom(l - m, lam - mu))


def translate_exterior(moments, l_max, shift):
    """Exterior moments about a new center, given the moments about the old one.

    shift = old center - new center. Exact for all l <= l_max.
    """
    R = regular_solid_harmonics(shift, l_max)
    result = {}
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            total = 0
            for lam in range(l + 1):
                for mu in range(-lam, lam + 1):
                    j, k = l - lam, m - mu
                    if abs(k) > j:
                        continue
                    coef = _addition_coefficient(l, m, lam, mu)
                    if coef:
                        total += coef * moments[(lam, mu)] * R[(j, k)].conjugate()
            result[(l, m)] = total
    return result


def translate_interior(moments, l_max, shift):
    """Interior moments about a new center, given the moments about the old one.

    shift = new center - old center. The result describes exactly the same polynomial
    sum_{l <= l_max} I_lm R_lm as the original moments.
    """
    R = regular_solid_harmonics(shift, l_max)
    result = {}
    for lam in range(l_max + 1):
        for mu in range(-lam, lam + 1):
            total = 0
            for l in range(lam, l_max + 1):
                for m in range(-l, l + 1):
                    j, k = l - lam, m - mu
                    if abs(k) > j:
                        continue
                    coef = _addition_coefficient(l, m, lam, mu)
                    if coef:
                        total += coef * moments[(l, m)] * R[(j, k)]
            result[(lam, mu)] = total
    return result
