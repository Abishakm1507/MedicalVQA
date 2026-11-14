"""FS-005 Phase 5 Stage 2: Confidence Calibration Script.

Calibrates confidence weights (w1, w2, w3) and abstention threshold (tau)
using held-out SLAKE validation data via ConfidenceCalibrator grid search.

Per ADR 0005:
- Grid search over step size 0.05
- Constraint: w1 + w2 + w3 = 1, w_i >= 0
- PRIMARY objective: minimize selective risk at operating coverage
- SECONDARY objective: AUROC (diagnostic only)
- CALIBRATION: ECE (diagnostic only)
- Coverage constraint prevents trivial all-abstain solution
- Selected weights and tau are frozen before touching the test set

This script uses SYNTHETIC validation data (no real model inference executed)
to demonstrate the calibration pipeline. In production, replace synthetic
data with actual VLM outputs from the SLAKE validation split.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on the path for package imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verification.confidence_gate import (
    ConfidenceCalibrator,
    ConfidenceConfig,
    compute_confidence,
)


def generate_synthetic_validation_data(n_samples: int = 100, seed: int = 42):
    """Generate synthetic validation data for calibration demonstration.

    In production, replace this with actual VLM outputs from the SLAKE
    validation split (1,053 English samples, 970 with GT regions).

    Returns:
        Tuple of (agreements, normalized_entropies, localizer_scores,
        evidence_states, correctness) lists.
    """
    import random
    random.seed(seed)

    agreements = []
    normalized_entropies = []
    localizer_scores = []
    evidence_states = []
    correctness = []

    for _i in range(n_samples):
        # Simulate varying quality of predictions
        is_correct = random.random() < 0.7  # ~70% accuracy

        if is_correct:
            # Correct predictions tend to have higher agreement, lower entropy
            agreement = random.uniform(0.6, 1.0)
            normalized_entropy = random.uniform(0.0, 0.4)
            localizer_score = random.uniform(0.6, 1.0)
            evidence_state = "present_grounded"
        else:
            # Incorrect predictions tend to have lower agreement, higher entropy
            agreement = random.uniform(0.1, 0.6)
            normalized_entropy = random.uniform(0.4, 1.0)
            localizer_score = random.uniform(0.0, 0.5)
            evidence_state = "present_grounded"

        # Occasionally include absence and uncertain claims
        r = random.random()
        if r < 0.15:
            evidence_state = "absent"
            localizer_score = None
            agreement = random.uniform(0.3, 0.8)
            normalized_entropy = random.uniform(0.0, 0.5)
        elif r < 0.25:
            evidence_state = "uncertain"
            localizer_score = None
            agreement = random.uniform(0.3, 0.7)
            normalized_entropy = random.uniform(0.5, 1.0)
        elif r < 0.35:
            evidence_state = "present_ungrounded"
            localizer_score = None

        agreements.append(agreement)
        normalized_entropies.append(normalized_entropy)
        localizer_scores.append(localizer_score)
        evidence_states.append(evidence_state)
        correctness.append(is_correct)

    return agreements, normalized_entropies, localizer_scores, evidence_states, correctness


def main():
    """Run calibration on synthetic validation data."""
    print("FS-005 Phase 5 Stage 2: Confidence Calibration")
    print("=" * 60)

    # Generate synthetic validation data
    agreements, normalized_entropies, localizer_scores, evidence_states, correctness = \
        generate_synthetic_validation_data(n_samples=200, seed=42)

    print(f"Validation samples: {len(agreements)}")
    print(f"Correct: {sum(correctness)} ({sum(correctness)/len(correctness):.1%})")
    print(f"Evidence states: "
          f"present_grounded={evidence_states.count('present_grounded')}, "
          f"absent={evidence_states.count('absent')}, "
          f"uncertain={evidence_states.count('uncertain')}, "
          f"present_ungrounded={evidence_states.count('present_ungrounded')}")
    print()

    # Run calibration
    calibrator = ConfidenceCalibrator(grid_step=0.05, min_coverage=0.1)
    result = calibrator.calibrate(
        agreements=agreements,
        normalized_entropies=normalized_entropies,
        localizer_scores=localizer_scores,
        evidence_states=evidence_states,
        correctness=correctness,
    )

    print("Calibration Results:")
    print(f"  w1 (agreement weight):     {result.w1:.3f}")
    print(f"  w2 (entropy weight):       {result.w2:.3f}")
    print(f"  w3 (localizer weight):     {result.w3:.3f}")
    print(f"  tau (abstention threshold): {result.tau:.3f}")
    print(f"  Selective risk:            {result.selective_risk:.4f}")
    print(f"  Coverage:                  {result.coverage:.4f}")
    print(f"  AUROC:                     {result.auroc:.4f}")
    print(f"  ECE:                       {result.ece:.4f}")
    print(f"  Validation samples:        {result.n_validation}")
    print()

    # Verify weight constraint
    weight_sum = result.w1 + result.w2 + result.w3
    print(f"Weight sum constraint (w1+w2+w3=1): {weight_sum:.6f} (should be 1.0)")
    print(f"All weights non-negative: {result.w1 >= 0 and result.w2 >= 0 and result.w3 >= 0}")
    print()

    # Demonstrate confidence calculation with fitted weights
    config = ConfidenceConfig(
        w1=result.w1, w2=result.w2, w3=result.w3, tau=result.tau,
        n_samples=5, temperature=0.7, random_seed=42
    )

    print("Sample confidence calculations with fitted weights:")
    test_cases = [
        ("present_grounded, high agreement", 0.9, 0.1, 0.9, "present_grounded"),
        ("present_grounded, low agreement", 0.3, 0.8, 0.3, "present_grounded"),
        ("absent claim", 0.8, 0.2, None, "absent"),
        ("uncertain claim", 0.5, 0.5, None, "uncertain"),
        ("present_ungrounded", 0.6, 0.4, None, "present_ungrounded"),
    ]
    for label, a, h, rho, es in test_cases:
        c = compute_confidence(a, h, rho, config, es, sampling_valid=True)
        print(f"  {label}: c={c:.4f}")

    print()

    # Save calibration artifacts
    output_dir = Path("experiments/runs/fs005_phase5/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_artifacts = {
        "experiment": "FS-005 Phase 5 Stage 2: Confidence Calibration",
        "timestamp": datetime.now().isoformat(),
        "validation_samples": result.n_validation,
        "calibration_config": {
            "w1": result.w1,
            "w2": result.w2,
            "w3": result.w3,
            "tau": result.tau,
            "n_samples": 5,
            "temperature": 0.7,
            "grid_step": 0.05,
            "min_coverage": 0.1,
        },
        "metrics": {
            "selective_risk": result.selective_risk,
            "coverage": result.coverage,
            "auroc": result.auroc,
            "ece": result.ece,
        },
        "coverage_curve": [
            {"tau": t, "coverage": c, "selective_accuracy": a}
            for t, c, a in result.coverage_curve
        ],
        "note": "Calibration performed on synthetic validation data. "
                "Replace with actual SLAKE validation split outputs for production.",
    }

    with open(output_dir / "calibration_result.json", "w") as f:
        json.dump(calibration_artifacts, f, indent=2)

    print(f"Calibration artifacts saved to: {output_dir}")
    print()
    print("WARNING: This calibration used SYNTHETIC validation data. "
          "For production, replace with actual SLAKE validation split outputs.")


if __name__ == "__main__":
    main()
