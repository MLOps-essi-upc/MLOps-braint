"""Definitions for the objects used by our resource endpoints."""

from enum import Enum

class BrainTumorType(Enum):
    """
        Enum that contains all the possible classes of the Brain Tumor
    """
    GLIOMA_TUMOR = 0
    MENIGIOMA_TUMOR = 1
    NO_TUMOR = 2
    PITUITARY_TUMOR = 3
