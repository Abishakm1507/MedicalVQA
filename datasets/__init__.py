"""Datasets package for MedicalVQA.

Provides loaders, normalizers, and patient-level isolation validators for:
- SLAKE (English multimodal VQA benchmark)
- VQA-RAD (Radiology clinician QA baseline)
- RSNA Pneumonia Detection Challenge
- VinDr-CXR Thoracic Findings Dataset
"""

from datasets.base import (
    BaseDatasetLoader,
    check_patient_leakage,
    normalize_bounding_box,
)
from datasets.cxr_loader import RSNADatasetLoader
from datasets.slake_loader import SlakeDatasetLoader
from datasets.vindr_loader import VinDrCXRDatasetLoader
from datasets.vqarad_loader import VQARADDatasetLoader

__all__ = [
    "BaseDatasetLoader",
    "normalize_bounding_box",
    "check_patient_leakage",
    "SlakeDatasetLoader",
    "VQARADDatasetLoader",
    "RSNADatasetLoader",
    "VinDrCXRDatasetLoader",
]
