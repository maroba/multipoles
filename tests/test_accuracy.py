#import matplotlib.pyplot as plt
import numpy as np

from multipoles.expansion import MultipoleExpansion


def calc_discrete_multipole(qs, xyzs, l_max=3):
    num_charges = len(qs)
    phi_mpe = MultipoleExpansion({
        "discrete": True,
        "charges": [
            {'q': qs[i], 'xyz': xyzs[i, :]} for i in range(num_charges)
        ]}, l_max=l_max)

    def phi_exact(x, y, z):
        phi = 0
        for i in range(num_charges):
            xi, yi, zi = xyzs[i, :]
            r = np.sqrt((x - xi) ** 2 + (y - yi) ** 2 + (z - zi) ** 2)
            phi += qs[i] / r
        return phi

    ds = np.logspace(1, 3, 20)
    expected = np.array([phi_exact(d, d, d) for d in ds])
    actual = np.array([phi_mpe(d, d, d) for d in ds])
    #plt.title("Discrete Multipole")
    errors = abs((expected - actual) / expected)
    # plt.loglog(ds, errors, "o-")
    # plt.xlabel("d")
    # plt.ylabel("Error")
    # plt.grid()
    # plt.show()

    return ds * np.sqrt(3), errors


def test_discrete_multipole():
    num_charges = 10
    np.random.seed(42)

    qs = np.ones(num_charges)
    qs[num_charges // 2:] *= -1

    approx_center_of_charge = np.array([215, 5, 23])

    xyzs = np.random.random((num_charges, 3)) + approx_center_of_charge
    ds, errors = calc_discrete_multipole(qs, xyzs, 30)

    assert np.max(errors) < 1.E-11


if __name__ == '__main__':
    test_discrete_multipole()
