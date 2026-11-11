# Contributing to MedicalVQA

Thank you for your interest in contributing to the MedicalVQA research engineering project.

## 1. Principles & Scientific Integrity
- **Research Faithfulness**: All implementations must strictly correspond to the methodology described in our research paper. Do not alter experimental conditions, formulas, or schemas without an accepted Architecture Decision Record (ADR).
- **Zero-Cost Constraint**: We do not accept dependencies that require paid subscriptions, proprietary APIs, or credit card registrations.
- **Reproducibility**: All experiments, random seeds, and evaluation routines must be deterministic and fully reproducible across local CPU and free Colab/Kaggle environments.
- **No Fabricated Data**: Never invent test metrics, benchmark scores, or calibration figures.

---

## 2. Development Workflow

### 2.1 Branching Strategy
- Branch from `develop`:
  ```bash
  git checkout -b feat/FS-XXX-feature-name
  ```
- Reference the Feature Specification ID (`FS-XXX`) in the branch name, commit messages, and PR titles.

### 2.2 Coding Standards
- **Python**: Strictly Python 3.11+. Follow PEP 8 guidelines enforced by `ruff`.
  ```bash
  ruff check .
  ruff format .
  ```
- **Type Annotations**: All public functions and classes must include Python type hints verified by `mypy`.
  ```bash
  mypy core/ api/ verification/ evaluation/
  ```
- **Testing**: Follow Test-Driven Development (TDD). Add comprehensive unit tests in `tests/` before writing implementation code.
  ```bash
  pytest --cov=. --cov-report=term-missing
  ```
  Minimum unit test coverage threshold is **85%**.

---

## 3. Pull Request Requirements
Before submitting a PR, verify:
1. All unit tests pass locally without CUDA dependencies.
2. Ruff linter reports 0 errors and 0 warnings.
3. Relevant documentation and Feature Specification (`docs/specifications/FS-XXX.md`) updated.
4. No hardcoded credentials, secret keys, or local absolute paths.
5. Large datasets (`data/raw/`) or model weights (`models/weights/`) are excluded from Git commits.
