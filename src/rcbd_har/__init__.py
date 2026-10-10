"""RCBD-HAR public package."""

from .models.classifier import BilateralHARModel
from .models.rcbd import RoleCanonicalBilateralDecomposition

__all__ = ["BilateralHARModel", "RoleCanonicalBilateralDecomposition"]
__version__ = "0.1.0"
