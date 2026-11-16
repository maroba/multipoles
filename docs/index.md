# multipoles

*multipoles* computes multipole expansions of the solutions of the Poisson equation, such as electrostatic
or gravitational potentials. Instead of integrating over the whole charge or mass distribution for every point
where you need the potential, you compute a few multipole moments once and evaluate the expansion cheaply
anywhere outside (or, for the interior expansion, inside) the distribution.

--8<-- "README.md:features"

## Installation

```
pip install --upgrade multipoles
```

*multipoles* only needs NumPy. For plotting, install it with `pip install multipoles[plot]`.

--8<-- "README.md:quickstart"

--8<-- "README.md:units"

## Where to go next

- [Usage](usage.md): charge densities, evaluation on grids, the field and the Hessian, accuracy control,
  the interior expansion, the multipole moments, combining, saving and plotting expansions, units and gravity
- [Theory](theory.md): the formulas behind the expansion, where it converges and how accurate it is
- [Accuracy](accuracy.ipynb): a worked comparison with exact solutions
- [API Reference](reference.md)
- [Changelog](changelog.md)
