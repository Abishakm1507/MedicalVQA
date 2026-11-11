# Security & Privacy Guidelines

This document outlines mandatory security protocols, static analysis configurations, and health-data privacy safeguards.

---

## 1. Zero Secret Leakage & Credential Management
* **No Hardcoded Tokens**: Never commit API keys, Hugging Face tokens, tunnel credentials, or private keys to the repository.
* **Environment Variables**: All external configuration is loaded through `.env` files parsed via `pydantic-settings`.
* **Git Pre-Commit Hygiene**: `.gitignore` strictly blocks `.env`, `*.secret`, `*.pem`, and `*.key`.

---

## 2. Medical Image Upload Security & Sanitization
Because users can upload medical scans (PNG, JPEG, DICOM):
1. **File Size Enforcement**: Hard limit of 15 MB per upload to prevent memory exhaustion / denial of service.
2. **MIME-Type & Magic Byte Validation**: Prevent extension spoofing by validating image headers using `python-magic` or Pillow byte inspection rather than trusting the file extension.
3. **Path Traversal Defense**: Never write uploaded files using client-supplied filenames. All temporary files are hashed using UUIDv4 (`{uuid4()}.png`) and saved inside safe scratch directories.
4. **Metadata & De-identification**: DICOM images, if accepted, must be stripped of all Protected Health Information (PHI) tags (Patient Name, Patient ID, DOB, Institution, Physician) using `pydicom` before downstream processing.

---

## 3. Remote GPU Worker Security
* **Reverse Tunnel Authentication**: The remote Colab/Kaggle worker exposes an HTTPS endpoint via Cloudflare Tunnels or ngrok. Access must require an `X-Worker-Auth-Key` HTTP header configured in both the local backend `.env` and the Colab notebook.
* **Payload Validation**: The remote worker validates all incoming requests using strict Pydantic models to prevent command injection or arbitrary code execution.

---

## 4. Automated Static Security & Dependency Scanning
As part of the mandatory CI/CD quality gate:
* **Static Analysis**: `bandit -r core/ api/ verification/ evaluation/ -ll` scans for common vulnerabilities (e.g., insecure random generators, shell injection).
* **Dependency Auditing**: `pip-audit` checks installed packages against known CVE vulnerability databases.
