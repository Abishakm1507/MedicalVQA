# Comprehensive Testing & Verification Strategy

This document establishes the testing levels, mock strategies, coverage criteria, and mathematical verification tests required for all code in this repository.

---

## 1. Testing Pyramid & Principles

```
           / \
          / E2E \         (Playwright: Web UI + API end-to-end flow)
         /-------\
        /  Integ  \       (FastAPI TestClient: POST /analyze, Claim Gate)
       /-----------\
      /    Unit     \     (Pytest: Schema, IoU, Formulas, Token Entropy)
     /---------------\
```

1. **Hardware-Decoupled Testing**: Unit and integration tests must run on CPU without requiring an NVIDIA GPU or external cloud connection.
2. **Deterministic Mocking**: All VLM generation is mocked using `MockVLMClient` returning pre-recorded fixture responses with controlled temperatures and seeds.
3. **Strict Coverage Floor**: Codebase must maintain $\ge 85\%$ line coverage reported via `pytest-cov`.
4. **Never Bypass Failures**: No test may be disabled, deleted, or marked `@pytest.mark.skip` without explicit ADR justification.

---

## 2. Test Categories & Execution

### 2.1 Unit Tests (`tests/unit/`)
* **Schemas & Contracts (`test_schemas.py`)**: Validates that Pydantic properly accepts compliant JSON and rejects malformed VLM outputs.
* **Mathematical Metrics (`test_metrics.py`)**:
  - IoU computation on synthetic overlapping/non-overlapping bounding boxes.
  - USR calculation before and after claim filtering.
  - Normalized entropy $H_n$ on unanimous ($H_n = 0$) and uniform ($H_n = 1$) sample distributions.
  - Confidence gate calculation $c = w_1 a + w_2 (1 - H_n) + w_3 \rho$ under edge conditions ($\rho = 0$, $a = 1$).
  - ECE calculation on perfectly calibrated and miscalibrated toy bins.
* **Claim Verifier (`test_claim_verifier.py`)**:
  - Correct filtering when VLM claims `R99` which does not exist in $R = \{R_1, R_2\}$.
  - Correct handling of negative findings citing `NONE`.

### 2.2 Integration Tests (`tests/integration/`)
* **API Ingestion (`test_api_analyze.py`)**:
  - Uses FastAPI `TestClient` to submit sample CXR images and queries to `POST /api/v1/analyze`.
  - Verifies response headers, status codes, Pydantic response payloads, and abstention behavior when confidence is low.
* **Localizer & Marker Pipeline (`test_localization_pipeline.py`)**:
  - Tests image loading, bounding box annotation, and Set-of-Mark drawing into PIL Image objects.

### 2.3 End-to-End Tests (`tests/e2e/`)
* **Frontend Web Testing (Playwright)**:
  - Uploads a test radiograph in the web UI.
  - Types a question.
  - Verifies that the Canvas overlay draws region bounding boxes, the confidence badge renders, and the answer is displayed.

---

## 3. Test Runner Commands
```bash
# Run all unit and integration tests with coverage
pytest --cov=core --cov=verification --cov=evaluation --cov=api --cov-report=term-missing

# Run only mathematical metric validation tests
pytest tests/unit/test_metrics.py -v

# Run linting and style checks
ruff check .
```
