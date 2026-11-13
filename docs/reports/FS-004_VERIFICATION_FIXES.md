# FS-004 Verification Fixes Report

**Scope:** Fixes for two failing unit tests and removal of unused pytest configuration.
**Modules:** `tests/unit/test_slake_loader_actual.py`, `tests/unit/test_ollama_client.py`, `pyproject.toml`
**Production code changed:** None.
**FS-002 / FS-003 implementation:** Not modified (no genuine defect demonstrated).

---

## 1. Root causes

### 1.1 `test_slake_loader_inline_boxes_still_work` (SLAKE inline boxes)
- **Root cause:** The fixture used `[50, 50, 100, 100]` as an XYWH box on a 100×100 image.
- The SLAKE loader parses inline `boxes` as `[x, y, w, h]` (`slake_loader.py:172-178`) and normalizes via `normalize_bounding_box(..., input_format="xywh")` (`base.py:56-58`): `xmax = x + w = 150`, `ymax = y + h = 150`.
- That exceeds the 100×100 image bounds, so production bounds validation (`base.py:77-95`) correctly rejects the box, yielding 0 GT regions instead of the expected 1.
- Production validation was behaving correctly; the test fixture was invalid.

### 1.2 `test_metadata_recorded_after_generate` (Ollama metadata timing)
- **Root cause:** `_call_generate` measures `wall_time = time.perf_counter() - t0` around a mocked `urllib.request.urlopen` (`ollama_client.py:133-142`). The mocked call completes within a single clock tick, so `wall_time` rounds to `0.0`, failing `assert meta["wall_time_s"] > 0`.
- The assertion itself was valid; the test lacked deterministic clock control.

### 1.3 Pytest configuration warnings
- **Root cause:** `pyproject.toml` declared `asyncio_mode = "auto"` and `asyncio_default_fixture_loop_scope = "function"`, but `pytest-asyncio` is not installed (`ModuleNotFoundError`) and the suite contains no async tests. pytest 9.2.0 therefore emits `PytestConfigWarning: Unknown config option` for both keys.

---

## 2. Exact test changes

### `tests/unit/test_slake_loader_actual.py`
1. `test_slake_loader_inline_boxes_still_work`: changed inline box fixture from `[50, 50, 100, 100]` to `[10, 10, 40, 40]` (XYWH), which on a 100×100 image yields `xmax=50, ymax=50` — valid, normalized box `(0.1, 0.1, 0.4, 0.4)`. Purpose preserved: verifying valid inline boxes are loaded into `ground_truth_regions`.
2. Added `test_slake_loader_out_of_bounds_inline_boxes_are_rejected`: uses the original `[50, 50, 100, 100]` box on a 100×100 image and asserts `len(ground_truth_regions) == 0`, confirming genuinely out-of-bounds boxes are rejected by production bounds validation (validation not weakened).

### `tests/unit/test_ollama_client.py`
1. `test_metadata_recorded_after_generate`: added `patch("vlm.ollama_client.time.perf_counter", side_effect=[1000.0, 1002.5])` so the elapsed-time calculation is deterministic; changed `assert meta["wall_time_s"] > 0` to `assert meta["wall_time_s"] == 2.5`. No `sleep` added; assertion not weakened to `>= 0`.

### `pyproject.toml`
1. Removed `asyncio_mode = "auto"` and `asyncio_default_fixture_loop_scope = "function"` from `[tool.pytest.ini_options]` (pytest-asyncio not installed; no async tests in suite). No dependencies added.

---

## 3. Production code changes
None. All fixes are test-fixture and configuration changes only. `normalize_bounding_box` bounds validation in `datasets/base.py` and the timing measurement in `vlm/ollama_client.py` were left unchanged.

---

## 4. Verification commands and results

Command:
```
.venv\Scripts\python.exe -m pytest tests/unit/test_slake_loader_actual.py tests/unit/test_ollama_client.py tests/unit/test_segmenter.py tests/unit/test_medsam_segmenter.py tests/unit/test_medsam_integration.py tests/unit/test_slake_loader.py tests/unit/test_detector.py tests/unit/test_rle.py tests/unit/test_schemas.py tests/unit/test_patient_isolation.py tests/unit/test_validate_dataset.py -v
```

Results:
- `tests/unit/test_slake_loader_actual.py`: 7 passed (including the new out-of-bounds rejection test)
- `tests/unit/test_ollama_client.py`: 10 passed
- `tests/unit/test_segmenter.py` (FS-003): 46 passed
- `tests/unit/test_medsam_segmenter.py` (FS-003): 32 passed
- Combined target + FS-003 + affected unit tests: **191 passed, 1 skipped, 14 failed**

The 14 failures (`test_detector.py` × 11, `test_medsam_integration.py` × 3) are **pre-existing and unrelated to this change**: they fail at `torch` import with `RuntimeError: Only a single TORCH_LIBRARY can be used to register the namespace triton` (a broken torch/triton installation in this venv). Verified via `git stash` — they fail identically with my changes reverted. They are not hidden; they are reported here.

Warnings: `asyncio_mode` / `asyncio_default_fixture_loop_scope` warnings eliminated (config removed).

---

## 5. Remaining issues
- `test_detector.py` and `test_medsam_integration.py` fail due to a broken `torch`/`torchvision` installation (triton library registration error at import), not due to any code change. Requires environment repair (reinstall CPU-only torch/torchvision per `pyproject.toml` ml extra note) before FS-002/FS-003 detector inference tests can run locally.
- No FS-005/FS-006 work begun, per instructions.
