# AI Agent Skills Plan & Customization Architecture

This document outlines the specialized skills, tools, and execution guidelines for AI agents working within Google Antigravity on the MedicalVQA project.

---

## 1. Google Antigravity Skill Ecosystem & Format

In Google Antigravity, skills are modular packages providing domain-specific workflows and scripts. Each skill resides in a directory with a required `SKILL.md` containing YAML frontmatter (`name`, `description`) and markdown instructions.

Project-level instructions and skills are loaded from:
- Project instructions: `AGENTS.md` (root directory)
- Project-specific skills: `.gemini/skills/<skill-name>/SKILL.md` or `.agents/skills/<skill-name>/SKILL.md`

---

## 2. Recommended Agent Skills by Engineering Domain

| Domain | Skill Capability | Target Tooling (100% Free / Open-Source) | Implementation in Repository |
| :--- | :--- | :--- | :--- |
| **Architecture Planning** | Feature Spec adherence & ADR tracking | ADR Markdown templates & traceability matrices | `docs/architecture/adr/`, `docs/specifications/` |
| **Python Development** | PEP 8, typing, modern 3.11 syntax | `ruff format`, `ruff check`, `mypy --strict` | `pyproject.toml` |
| **Machine Learning** | Quantization, Hugging Face pipelines | `transformers`, `accelerate`, `bitsandbytes` (4-bit) | `vlm/`, `api/remote_worker.py` |
| **Computer Vision** | Bounding box conversion, Set-of-Mark | `OpenCV`, `Pillow`, `Albumentations` | `localization/`, `evaluation/robustness.py` |
| **Systematic Debugging** | Traceback diagnosis, fixture isolation | `pytest -vv --tb=short -l` | `tests/` |
| **Test-Driven Dev (TDD)**| Unit & integration test generation | `pytest`, `pytest-cov`, `pytest-asyncio` | `tests/unit/`, `tests/integration/` |
| **Security Auditing** | AST security checks, secret scanning | `bandit -r`, regex token detectors | Pre-commit quality gate |
| **Code Quality Audit** | Complexity & style enforcement | `ruff`, `flake8-bugbear` | `ruff check .` |
| **Dead Code Detection** | Unused functions, dead imports | `vulture`, `ruff check --select F401` | `vulture core/ api/ verification/` |
| **Duplicate Code** | Duplicate logic identification | `flake8-duplicate`, AST similarity checkers | Quality review checklist |
| **Dependency Auditing** | Vulnerability check for installed packages | `pip-audit`, `safety` (free DB) | `pip-audit -r requirements-dev.txt` |
| **Documentation** | API docs & LaTeX generation | Sphinx / MkDocs / FastAPI Swagger | FastAPI `/docs`, `README.md` |

---

## 3. Project-Specific Custom Skills to Establish

To automate repetitive research engineering tasks, we establish custom skills within `.agents/skills/`:

### 3.1 Skill: `eval-runner` (`.agents/skills/eval-runner/SKILL.md`)
* **Purpose**: Orchestrates standard evaluation runs across the 3 experimental conditions on SLAKE or VQA-RAD test splits, calculating USR, pointing accuracy, ECE, and AUROC with bootstrap resampling.
* **Usage**: Triggered when validating a new model checkpoint or prompt modification.

### 3.2 Skill: `grounding-verifier` (`.agents/skills/grounding-verifier/SKILL.md`)
* **Purpose**: Tests visual bounding box extraction, MedSAM mask conversion, and Set-of-Mark label placement against ground-truth coordinate fixtures.

### 3.3 Skill: `security-quality-audit` (`.agents/skills/security-quality-audit/SKILL.md`)
* **Purpose**: Executes `ruff check`, `mypy`, `bandit`, and `pip-audit` in a single pass to ensure pull requests strictly comply with our zero-defect quality gate.
