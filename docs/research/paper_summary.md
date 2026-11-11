# Research Paper Summary & Methodology

## 1. Bibliographic Metadata
* **Title**: Region-Grounded and Uncertainty-Aware Visual Question Answering on Medical Images Using General and Medical Vision-Language Models
* **Authors**: Abisha K M, Anusiya S, Bhavya Sree P R, Dirisala Kavya
* **Affiliation**: Department of Artificial Intelligence and Data Science, RMK Engineering College, Kavaraipettai, Tamil Nadu, India
* **Mentor**: Mr. Sathesh Abraham Leo E, Assistant Professor
* **Target Venue / Format**: IEEE Conference Format (4 pages)

---

## 2. Problem Statement
Vision-Language Models (VLMs) demonstrate impressive fluency in interpreting clinical images, but suffer from three critical failure modes:
1. **Hallucination / Factuality Gaps**: VLMs frequently claim findings or abnormalities not present in the scan.
2. **Lack of Spatial Grounding**: Plain-text responses provide no spatial link to the underlying image pixels, forcing clinicians to scour the whole image to verify claims.
3. **Overconfidence & Miscalibration**: Generative models rarely express uncertainty or doubt, answering even when evidence is ambiguous or missing.

---

## 3. Core Hypothesis & Proposed Solution
> **Guiding Axiom**: *"The system can only claim what it can point to."*

The authors propose a deterministic 6-stage grounded answering pipeline that constrains generative VLMs:
1. **Stage 1 (Input)**: Ingests medical image $I$ and clinical query $q$.
2. **Stage 2 (Localize)**: A region detector (YOLO or DETR) proposes $k$ candidate bounding boxes $b_i$ with detection scores $\rho_i$. MedSAM generates corresponding segmentation masks $m_i$. The number of regions $k$ is intentionally kept small (e.g. 3–7) to avoid visual confusion.
3. **Stage 3 (Label)**: Superimposes region IDs ($R_1 \dots R_k$) onto an image copy following Set-of-Mark visual prompting.
4. **Stage 4 (Answer)**: Prompts the VLM (Qwen2.5-VL-3B or MedGemma-4B) under a strict JSON contract requiring region ID citations for every clinical claim. $N$ answers are sampled.
5. **Stage 5 (Verify)**: Schema verification rejects malformed replies and strips claims citing nonexistent region IDs. During evaluation, cited regions are tested against ground-truth masks/boxes ($\text{IoU} \ge 0.5$).
6. **Stage 6 (Gate)**: Fuses majority agreement ($a$), semantic entropy ($H_n$), and region detection score ($\rho$) into a scalar confidence $c$. If $c \ge \tau$, output answer; otherwise, report "uncertain" (selective abstention).

---

## 4. Benchmark Datasets
* **SLAKE**: Main multimodal benchmark providing English questions, answers, bounding boxes, pixel masks, and visual knowledge graphs.
* **VQA-RAD**: Baseline radiology QA dataset (clinician-written questions and answers; no region annotations).
* **RSNA Pneumonia Detection**: Chest radiograph dataset with bounding boxes for lung opacity used to train/validate the localizer.
* **VinDr-CXR**: Large open chest X-ray dataset with radiologist bounding boxes for multiple local thoracic findings.

---

## 5. Planned Evaluation Dimensions
1. **Answer Quality**: Accuracy and F1 score on closed questions.
2. **Grounding Validity**: Unsupported-Claim Rate (USR) and Pointing Accuracy ($\text{IoU} \ge 0.5$).
3. **Calibration & Abstention**: Expected Calibration Error (ECE), AUROC of confidence score $c$, and accuracy-vs-coverage curve.
4. **Robustness**: Performance delta under visual perturbations (noise, contrast, rotation, flips) and question paraphrasing.
5. **Masked-Region Ablation**: Erasing the cited region and re-evaluating the model to prove the prediction relied on the visual evidence.
6. **Blinded Review**: Clinical rubric scoring with Cohen's $\kappa$ (or disclosed LLM-as-a-judge alternative with bootstrap confidence intervals).
