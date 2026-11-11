# ADR 0001: Selection of Python 3.11 as Primary Runtime

## Status
Accepted

## Context
The host development system has Python 3.13.5 installed by default. However, essential machine learning, computer vision, and scientific packages—including PyTorch, torchvision, bitsandbytes, ONNX runtime, and MedSAM dependencies—have limited, experimental, or missing pre-compiled Windows binary wheels for Python 3.13. Attempting to install these under Python 3.13 leads to C++ compiler toolchain errors, binary incompatibilities, and broken dependencies.

Python 3.11.0 is also installed on the system (`C:\Users\yuvar\AppData\Local\Programs\Python\Python311\python.exe`).

## Decision
We select **Python 3.11** as the standardized project runtime across all environments (local laptop venv, Docker containers, and Google Colab workers). The project virtual environment is created strictly using Python 3.11 on the large storage drive (`D:\GitHub\honors-project\MedicalVQA\.venv`).

## Consequences
- **Positive**: 100% pre-compiled binary wheel availability for all ML, CV, and scientific libraries without requiring Visual Studio C++ build tools.
- **Positive**: Complete alignment with Google Colab and Kaggle runtime Python versions.
- **Negative**: Developers must ensure `py -3.11` or `.venv/Scripts/python.exe` is explicitly targeted when executing commands.
