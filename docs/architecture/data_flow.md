# End-to-End Data Flow Specification

This document details the sequence of operations, data structures, and state transitions during an inference request.

---

## 1. Sequence Diagram: Clinical Query Processing

```mermaid
sequenceDiagram
    autonumber
    actor Clinician as Clinician / User
    participant Web as React Web App
    participant API as FastAPI Backend (/analyze)
    participant Localizer as YOLO + MedSAM Engine
    participant VLM as Remote VLM Worker
    participant Gate as Verification & Confidence Gate

    Clinician->>Web: Uploads Chest X-ray + Asks "Is there a pleural effusion?"
    Web->>API: POST /api/v1/analyze (multipart/form-data: image, question, model_name)
    
    API->>Localizer: Propose candidate regions(image, max_k=5)
    Localizer-->>API: Returns regions R = {R1, R2, R3} with boxes, masks, and scores ρ
    
    API->>Localizer: Render visual overlay(image, R)
    Localizer-->>API: Returns marked_image with R1..Rk pills

    API->>VLM: Query VLM(marked_image, question, schema, N=3)
    VLM-->>API: Returns 3 JSON response samples
    
    API->>Gate: Verify schema & claims(samples, candidate_regions=R)
    Gate-->>API: Strips invalid region claims
    
    API->>Gate: Calculate confidence c = w1*a + w2*(1-Hn) + w3*ρ
    alt c >= tau (Confident)
        Gate-->>API: Status = ACCEPT, answer = "Yes, right lower zone", cited_region = "R2"
    else c < tau (Uncertain)
        Gate-->>API: Status = ABSTAIN, answer = "uncertain", alert = "Low confidence deferral"
    end

    API-->>Web: JSON payload (answer, status, confidence, regions, masks)
    Web-->>Clinician: Renders marked image on Canvas + Confidence Badge + Findings
```

---

## 2. API Contract & Data Schemas

### 2.1 Request Payload (`POST /api/v1/analyze`)
* `image`: Multipart file upload (JPEG/PNG/DICOM).
* `question`: String (e.g., *"Is there cardiomegaly?"*).
* `model_name`: Enum (`"qwen2.5-vl-3b"`, `"medgemma-4b"`, `"both"`).
* `sampling_n`: Integer (default: 3; range: 1–10).

### 2.2 Internal VLM Structured Prompt & Response Schema
```json
{
  "answer": "Yes, right lower zone",
  "findings": [
    {
      "claim": "pleural effusion",
      "region_id": "R2",
      "confidence": "high"
    }
  ]
}
```

### 2.3 Response Payload (`POST /api/v1/analyze`)
```json
{
  "request_id": "req-987a-4c2b",
  "question": "Is there a pleural effusion?",
  "model": "qwen2.5-vl-3b",
  "status": "ACCEPTED",
  "final_answer": "Yes, right lower zone",
  "is_abstaining": false,
  "confidence_score": 0.842,
  "confidence_breakdown": {
    "agreement": 1.0,
    "normalized_entropy": 0.0,
    "localizer_score": 0.89
  },
  "regions": [
    {
      "region_id": "R1",
      "box_2d": [120, 140, 260, 310],
      "score": 0.92,
      "mask_rle": "..."
    },
    {
      "region_id": "R2",
      "box_2d": [340, 420, 480, 560],
      "score": 0.89,
      "mask_rle": "..."
    }
  ],
  "findings": [
    {
      "claim": "pleural effusion",
      "region_id": "R2",
      "is_grounded": true,
      "localizer_score": 0.89
    }
  ]
}
```
If $c < \tau$, `"status"` becomes `"ABSTAINED"`, `"final_answer"` becomes `"uncertain"`, and `"is_abstaining"` is `true`.
