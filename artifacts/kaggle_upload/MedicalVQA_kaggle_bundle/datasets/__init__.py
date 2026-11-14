"""Datasets package for MedicalVQA (minimal bundle for Kaggle inference).

Provides SLAKE dataset loader and bounding-box normalization utilities.
"""

from datasets.base import (
    BaseDatasetLoader,
    PatientIsolationReport,
    check_patient_leakage,
    denormalize_bounding_box,
    normalize_bounding_box,
)
from datasets.slake_loader import SlakeDatasetLoader

__all__ = [
    "BaseDatasetLoader",
    "normalize_bounding_box",
    "denormalize_bounding_box",
    "check_patient_leakage",
    "PatientIsolationReport",
    "SlakeDatasetLoader",
]
