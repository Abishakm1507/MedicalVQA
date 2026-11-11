"""Datasets package for MedicalVQA.

Provides loaders, normalizers, and patient-level isolation validators for:
- SLAKE (English multimodal VQA benchmark)
- VQA-RAD (Radiology clinician QA baseline)
- RSNA Pneumonia Detection Challenge
- VinDr-CXR Thoracic Findings Dataset
"""

from datasets.base import (
    BaseDatasetLoader,
    PatientIsolationReport,
    check_patient_leakage,
    denormalize_bounding_box,
    normalize_bounding_box,
)
from datasets.cxr_loader import RSNADatasetLoader, VinDrCXRDatasetLoader
from datasets.slake_loader import SlakeDatasetLoader
from datasets.validate_dataset import validate_dataset_sample
from datasets.vqarad_loader import VQARADDatasetLoader

__all__ = [
    "BaseDatasetLoader",
    "normalize_bounding_box",
    "denormalize_bounding_box",
    "check_patient_leakage",
    "PatientIsolationReport",
    "SlakeDatasetLoader",
    "VQARADDatasetLoader",
    "RSNADatasetLoader",
    "VinDrCXRDatasetLoader",
    "validate_dataset_sample",
]
