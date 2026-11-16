"""Multipole expansions of electrostatic and gravitational potentials."""

from .expansion import (
    EPSILON_0,
    GRAVITATIONAL_CONSTANT,
    Evaluation,
    ExpansionValidityWarning,
    InvalidChargeDistributionException,
    InvalidExpansionException,
    MultipoleExpansion,
)

__version__ = "0.6.0"

__all__ = [
    "MultipoleExpansion",
    "Evaluation",
    "ExpansionValidityWarning",
    "InvalidChargeDistributionException",
    "InvalidExpansionException",
    "EPSILON_0",
    "GRAVITATIONAL_CONSTANT",
]
