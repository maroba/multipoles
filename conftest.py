import numpy as np
import pytest

from multipoles import MultipoleExpansion


def gaussian(XYZ, xyz0, sigma):
    g = np.ones_like(XYZ[0])
    for k in range(3):
        g *= np.exp(-(XYZ[k] - xyz0[k]) ** 2 / sigma**2)
    return g * (sigma**2 * np.pi) ** -1.5


@pytest.fixture(autouse=True)
def add_doctest_namespace(doctest_namespace):
    """Names the docstring examples of the methods use, as defined in the examples of __init__."""
    x, y, z = [np.linspace(-8, 8, 81)] * 3
    XYZ = np.meshgrid(x, y, z, indexing="ij")
    rho = gaussian(XYZ, (0, 0, 1), 1.5) - gaussian(XYZ, (0, 0, -1), 1.5)

    doctest_namespace["np"] = np
    doctest_namespace["MultipoleExpansion"] = MultipoleExpansion
    doctest_namespace["gaussian"] = gaussian
    doctest_namespace["Phi"] = MultipoleExpansion({"discrete": False, "rho": rho, "xyz": XYZ}, l_max=2)
