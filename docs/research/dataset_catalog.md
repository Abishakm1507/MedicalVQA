# Medical VQA Dataset Catalog & Licensing Guide

This document catalogues all datasets used in the research project, verifies their open-access licenses, and defines their local storage and preprocessing protocols.

---

## 1. Summary of Benchmark Datasets

| Dataset | Modality | Annotations Available | Primary Role | License / Access Terms | Local Path |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SLAKE** | Multimodal (CXR, CT, MRI) | English/Chinese QA, Bounding Boxes, Pixel Masks, Knowledge Graphs | Primary grounding & VQA benchmark | Open Research / CC BY 4.0 | `D:/GitHub/honors-project/MedicalVQA/data/raw/slake` |
| **VQA-RAD** | Radiology (CXR, CT, MRI) | Clinician-generated English QA pairs (Closed & Open) | Answer accuracy & F1 baseline | Open Access (U.S. NLM / PhysioNet) | `D:/GitHub/honors-project/MedicalVQA/data/raw/vqarad` |
| **RSNA Pneumonia** | Chest Radiographs (DICOM/PNG) | Bounding boxes for lung opacity / pneumonia | Localizer detector training & evaluation | Open Research (Kaggle / RSNA) | `D:/GitHub/honors-project/MedicalVQA/data/raw/rsna` |
| **VinDr-CXR** | Chest Radiographs (DICOM/PNG) | Radiologist bounding boxes for 14 critical thoracic findings | Localizer detector evaluation & transfer | PhysioNet Credentialed / Open Subsplit | `D:/GitHub/honors-project/MedicalVQA/data/raw/vindr` |

---

## 2. Detailed Dataset Specifications

### 2.1 SLAKE (Semantically-Labeled Knowledge-Enhanced Dataset)
* **Citation**: Liu et al., *"SLAKE: A Semantically-Labeled Knowledge-Enhanced Dataset for Medical Visual Question Answering"*, ISBI 2021.
* **Volume**: 642 medical images, 14,028 QA pairs (7,033 English QA pairs).
* **Grounding Annotations**:
  - `mask/`: Pixel-level PNG segmentation masks.
  - `json/`: Bounding boxes in `[x, y, w, h]` format associated with organ and lesion entities.
* **Data Split**: Standard official train (70%), validation (15%), test (15%) splits.
* **Storage Footprint**: ~1.2 GB uncompressed.

### 2.2 VQA-RAD (Radiology Visual Question Answering)
* **Citation**: Lau et al., *"A dataset of clinically generated visual questions and answers about radiology images"*, Scientific Data 2018.
* **Volume**: 315 radiology images, 3,515 question-answer pairs.
* **Question Taxonomy**:
  - Closed questions: Yes/No, lateral ("left" vs. "right"), multi-choice (evaluated with Accuracy and Macro-F1).
  - Open questions: Descriptive answers (evaluated with BLEU/Token F1).
* **Role in Paper**: Serves as the ungrounded baseline comparison to verify answer fidelity across clinical QA styles.
* **Storage Footprint**: ~150 MB.

### 2.3 RSNA Pneumonia Detection Challenge
* **Citation**: Shih et al., *"Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia"*, Radiology: AI 2019.
* **Volume**: 26,684 frontal chest X-rays (subset with bounding boxes for pneumonia opacities).
* **Role in Paper**: Training and calibrating the candidate region bounding box detector ($b_i, \rho_i$) for thoracic opacities.
* **Storage Footprint**: ~3.5 GB (PNG converted format).

### 2.4 VinDr-CXR
* **Citation**: Nguyen et al., *"VinDr-CXR: An open dataset of chest X-rays with radiologist’s annotations"*, Scientific Data 2022.
* **Volume**: 18,000 chest radiographs labeled by 17 radiologists for 22 local critical findings and 6 global diagnoses.
* **Role in Paper**: Evaluating detector pointing accuracy on diverse chest pathologies (pleural effusion, cardiomegaly, pneumothorax, atelectasis, etc.).

---

## 3. Storage Hierarchy on Drive D:

All raw and preprocessed data must reside exclusively on the large storage drive (`D:`):

```
D:/GitHub/honors-project/MedicalVQA/
└── data/
    ├── raw/
    │   ├── slake/
    │   │   ├── imgs/
    │   │   ├── masks/
    │   │   └── train.json, validate.json, test.json
    │   ├── vqarad/
    │   │   ├── images/
    │   │   └── VQA_RAD_Dataset.json
    │   ├── rsna/
    │   └── vindr/
    └── processed/
        ├── slake_english_test.jsonl
        ├── vqarad_closed_test.jsonl
        └── detector_eval_cxr.jsonl
```

---

## 4. Ethical & Privacy Compliance
* All datasets utilized are publicly de-identified research benchmarks.
* No protected health information (PHI) or private patient data is involved.
* Ingestion loaders will enforce checksum validation (SHA-256) to ensure data integrity and full reproducibility.
