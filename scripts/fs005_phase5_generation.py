"""FS-005 Phase 5 execution harness: resumable, condition-independent generation.

Responsibilities:
1. Generate N=5 VLM responses once per (model, track, sample) and persist each
   as a condition-independent GenerationRecord via GenerationArtifactStore.
2. Compute agreement, normalized entropy, and fused confidence from the shared
   responses (semantic clustering + confidence gate).
3. Evaluate Condition 2 (grounded + verification, NO abstention gate) and
   Condition 3 (grounded + verification + abstention gate) from the SAME
   persisted artifact. Only the abstention decision differs; no new responses
   are generated for Condition 3.
4. Resumable execution: skip valid completed records, retry failed/corrupt
   records without overwriting valid results, persist incrementally, and record
   progress + failure summaries.

This script does NOT execute real model inference by default. Run with
--mock to exercise the harness with MockVLMClient (for testing/pilot dry-runs).
Real inference requires a VLMClient pointed at MedGemma-4B / Qwen2.5-VL-3B and
must be launched separately (Colab T4); see the Colab pilot guide.

Traceability:
  - Feature Specification: FS-005 (Phase 5 execution harness)
  - ADR 0005: N=5 sampling; confidence c = w1*a + w2*(1-Hn) + w3*rho
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

# Ensure project root is on the path for package imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.generation_artifact import (
    GenerationArtifactStore,
    GenerationRecord,
    GenerationSample,
    GenerationStatus,
    ValidationStatus,
    compute_generation_id,
    validate_generation_record,
)
from verification.confidence_gate import (
    ConfidenceConfig,
    EvidenceState,
    compute_confidence,
    make_selective_prediction,
)
from verification.semantic_clustering import SemanticClusterer
from vlm.base import VLMClient

# Placeholder calibration configuration. These weights/threshold are NOT
# validated research values; they must be fitted on held-out SLAKE validation
# data (fit/assessment split by image ID) before the full experiment.
DEFAULT_W1 = 1.0 / 3.0
DEFAULT_W2 = 1.0 / 3.0
DEFAULT_W3 = 1.0 / 3.0
DEFAULT_TAU = 0.5

RESUME_SUMMARY_NAME = "generation_progress.json"


@dataclass
class ConditionResult:
    """Outcome of evaluating one answer condition for a single sample."""

    condition: int  # 2 or 3
    question_id: str | int
    image_id: str | int
    track: str
    model_name: str
    generation_id: str

    # Shared condition-independent inputs (identical for Cond. 2 and Cond. 3).
    agreement: float
    normalized_entropy: float
    confidence: float
    evidence_state: str
    sampling_valid: bool
    majority_answer: str
    raw_answers: list[str]

    # Condition-specific decision.
    answer: str | None  # None if abstained
    is_abstaining: bool
    in_coverage: bool


@dataclass
class GenerationProgress:
    """Incremental progress and failure summary persisted across restarts."""

    model_name: str
    track: str
    condition_pair: str  # e.g. "2+3"
    n_samples_expected: int
    n_completed: int = 0
    n_skipped: int = 0
    n_failed: int = 0
    n_retried: int = 0
    n_corrupt: int = 0
    completed_question_ids: list[str | int] = field(default_factory=list)
    failed_question_ids: list[str | int] = field(default_factory=list)
    corrupt_question_ids: list[str | int] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    started_at_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    updated_at_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "track": self.track,
            "condition_pair": self.condition_pair,
            "n_samples_expected": self.n_samples_expected,
            "n_completed": self.n_completed,
            "n_skipped": self.n_skipped,
            "n_failed": self.n_failed,
            "n_retried": self.n_retried,
            "n_corrupt": self.n_corrupt,
            "completed_question_ids": self.completed_question_ids,
            "failed_question_ids": self.failed_question_ids,
            "corrupt_question_ids": self.corrupt_question_ids,
            "failures": self.failures,
            "started_at_utc": self.started_at_utc,
            "updated_at_utc": self.updated_at_utc,
        }


class GenerationRunner:
    """Resumable execution harness for condition-shared generation artifacts."""

    def __init__(
        self,
        vlm_client: VLMClient,
        model_name: str,
        model_revision: str | None,
        track: str,
        output_dir: str | Path = "experiments/runs/fs005_phase5",
        random_seed: int | None = None,
        temperature: float = 0.7,
        n_samples: int = 5,
        w1: float = DEFAULT_W1,
        w2: float = DEFAULT_W2,
        w3: float = DEFAULT_W3,
        tau: float = DEFAULT_TAU,
        similarity_threshold: float = 0.75,
    ) -> None:
        self.vlm_client = vlm_client
        self.model_name = model_name
        self.model_revision = model_revision
        self.track = track
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        self.temperature = temperature
        self.n_samples = n_samples
        self.store = GenerationArtifactStore(
            root=self.output_dir / "generations"
        )
        self.conditions_dir = self.output_dir / "conditions"
        self.conditions_dir.mkdir(parents=True, exist_ok=True)

        self.confidence_config = ConfidenceConfig(
            w1=w1,
            w2=w2,
            w3=w3,
            tau=tau,
            n_samples=n_samples,
            temperature=temperature,
            random_seed=random_seed,
        )
        self.clusterer = SemanticClusterer(
            similarity_threshold=similarity_threshold, force_fallback=True
        )

    # -- generation ---------------------------------------------------------

    def _generate_sample(
        self,
        image,
        question: str,
        region_ids: list[str],
    ) -> tuple[list[GenerationSample], float, str | None]:
        """Run N=5 inference for one sample and build per-sample records.

        Returns (samples, total_duration_s, error). On total failure, error is
        set and samples contain per-sample failure entries.
        """
        start = time.perf_counter()
        error: str | None = None
        raw_responses: list[str | None] = []
        try:
            raw_responses = self.vlm_client.sample_n(
                marked_image=image,
                question=question,
                n=self.n_samples,
                temperature=self.temperature,
            )
        except Exception as exc:  # noqa: BLE001 - capture and persist failure
            error = f"{type(exc).__name__}: {exc}"
            raw_responses = [None] * self.n_samples

        samples: list[GenerationSample] = []
        for idx, raw in enumerate(raw_responses):
            sample_start = time.perf_counter()
            sample_error: str | None = None
            parsed_answer: str | None = None
            claims: list[dict[str, Any]] = []
            status: Literal["ok", "failed"] = "ok"

            if raw is None:
                status = "failed"
                sample_error = error or "no response returned"
            else:
                try:
                    from core.schemas import VLMStructuredOutput

                    structured = VLMStructuredOutput.model_validate_json(raw)
                    parsed_answer = structured.answer
                    claims = [
                        {
                            "claim": c.claim,
                            "region_id": c.region_id,
                            "status": c.status,
                            "confidence": c.confidence,
                        }
                        for c in structured.findings
                    ]
                except Exception as exc:  # noqa: BLE001
                    status = "failed"
                    sample_error = f"parse error: {type(exc).__name__}: {exc}"
                    parsed_answer = None
                    claims = []

            samples.append(
                GenerationSample(
                    sample_index=idx,
                    raw_response=raw,
                    parsed_answer=parsed_answer,
                    claims=claims,
                    inference_duration_s=time.perf_counter() - sample_start,
                    timestamp_utc=datetime.now(UTC).isoformat(),
                    error=sample_error,
                    status=status,
                )
            )

        total_duration = time.perf_counter() - start
        return samples, total_duration, error

    def _build_record(
        self,
        question_id: str | int,
        image_id: str | int,
        samples: list[GenerationSample],
        status: GenerationStatus,
        total_duration_s: float | None,
        error: str | None,
    ) -> GenerationRecord:
        generation_id = compute_generation_id(
            model_name=self.model_name,
            model_revision=self.model_revision,
            track=self.track,
            question_id=question_id,
            image_id=image_id,
            n_samples=self.n_samples,
            temperature=self.temperature,
            random_seed=self.random_seed,
        )
        return GenerationRecord(
            generation_id=generation_id,
            model_name=self.model_name,
            model_revision=self.model_revision,
            track=self.track,
            question_id=question_id,
            image_id=image_id,
            n_samples=self.n_samples,
            temperature=self.temperature,
            random_seed=self.random_seed,
            samples=samples,
            status=status,
            validation_status=ValidationStatus.NOT_VALIDATED,
            total_duration_s=total_duration_s,
            error=error,
        )

    def generate_or_load(
        self,
        question_id: str | int,
        image_id: str | int,
        question: str,
        image,
        region_ids: list[str] | None = None,
        force_regenerate: bool = False,
    ) -> tuple[GenerationRecord, str]:
        """Return a valid COMPLETED GenerationRecord for one sample.

        Action taken (returned as the second element):
        - "skipped"  : valid completed record already existed (no generation).
        - "generated": record was generated and persisted now.
        - "retried"  : existing failed/corrupt record was retried and replaced.
        - "corrupt"  : existing artifact was corrupt and could not be retried
          (or generation failed); left in place / recorded as failed.
        """
        region_ids = region_ids or []
        generation_id = compute_generation_id(
            model_name=self.model_name,
            model_revision=self.model_revision,
            track=self.track,
            question_id=question_id,
            image_id=image_id,
            n_samples=self.n_samples,
            temperature=self.temperature,
            random_seed=self.random_seed,
        )

        # Fast path: valid completed record already exists -> skip (no duplicate).
        if not force_regenerate and self.store.is_completed_valid(
            self.model_name, self.track, generation_id
        ):
            return (
                self.store.load(self.model_name, self.track, generation_id),  # type: ignore[return-value]
                "skipped",
            )

        # Determine whether an existing artifact needs retry (failed or corrupt).
        existing_status: str | None = None
        corrupt = False
        try:
            existing = self.store.load(
                self.model_name, self.track, generation_id
            )
            existing_status = existing.status if existing is not None else None
        except ValueError:
            corrupt = True

        if corrupt:
            # Detect and record corrupt/incompatible artifact; do not silently
            # overwrite. Attempt regeneration into the same deterministic id.
            action = "generated"
        elif existing_status == GenerationStatus.FAILED:
            action = "retried"
        else:
            action = "generated"

        # Generate (or regenerate).
        try:
            samples, total_duration, error = self._generate_sample(
                image=image,
                question=question,
                region_ids=region_ids,
            )
        except Exception as exc:  # noqa: BLE001
            samples = [
                GenerationSample(
                    sample_index=i,
                    raw_response=None,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                for i in range(self.n_samples)
            ]
            total_duration = None
            error = f"{type(exc).__name__}: {exc}"

        status = (
            GenerationStatus.COMPLETED if error is None and all(s.status == "ok" for s in samples)
            else GenerationStatus.FAILED
        )
        record = self._build_record(
            question_id=question_id,
            image_id=image_id,
            samples=samples,
            status=status,
            total_duration_s=total_duration,
            error=error,
        )

        # Validate before persisting (schema + business rules).
        violations = validate_generation_record(record)
        if violations:
            record.status = GenerationStatus.FAILED
            record.error = (record.error or "") + "; validation: " + "; ".join(violations)

        self.store.save(record, validate=True)
        return record, action

    # -- condition evaluation (shared artifacts) ----------------------------

    def _uncertainty_from_record(
        self, record: GenerationRecord
    ) -> tuple[float, float, str, list[str], bool]:
        """Compute agreement, entropy, majority answer, raw answers, validity."""
        raw_answers = [
            s.raw_response if s.raw_response is not None else ""
            for s in record.samples
        ]
        result = self.clusterer.compute_agreement_entropy(raw_answers)
        return (
            result.agreement,
            result.normalized_entropy,
            result.majority_answer,
            raw_answers,
            result.is_valid,
        )

    def _confidence_from_record(
        self,
        record: GenerationRecord,
        agreement: float,
        normalized_entropy: float,
        sampling_valid: bool,
        localizer_score: float | None = None,
    ) -> tuple[float, EvidenceState]:
        evidence_state: EvidenceState = "present_grounded" if self.track == "B" else "present_ungrounded"
        confidence = compute_confidence(
            agreement=agreement,
            normalized_entropy=normalized_entropy,
            localizer_score=localizer_score,
            config=self.confidence_config,
            evidence_state=evidence_state,
            sampling_valid=sampling_valid,
        )
        return confidence, evidence_state

    def evaluate_condition_2(
        self,
        record: GenerationRecord,
        localizer_score: float | None = None,
    ) -> ConditionResult:
        """Evaluate Condition 2: grounded + verification, NO abstention gate.

        All samples are accepted regardless of confidence. The answer is the
        majority (clustered) answer. Shares the same record inputs as Condition 3.
        """
        agreement, entropy, majority, raw_answers, sampling_valid = (
            self._uncertainty_from_record(record)
        )
        confidence, evidence_state = self._confidence_from_record(
            record, agreement, entropy, sampling_valid, localizer_score
        )
        # Condition 2 never abstains.
        return ConditionResult(
            condition=2,
            question_id=record.question_id,
            image_id=record.image_id,
            track=record.track,
            model_name=record.model_name,
            generation_id=record.generation_id,
            agreement=agreement,
            normalized_entropy=entropy,
            confidence=confidence,
            evidence_state=evidence_state,
            sampling_valid=sampling_valid,
            majority_answer=majority,
            raw_answers=raw_answers,
            answer=majority,
            is_abstaining=False,
            in_coverage=True,
        )

    def evaluate_condition_3(
        self,
        record: GenerationRecord,
        localizer_score: float | None = None,
    ) -> ConditionResult:
        """Evaluate Condition 3: grounded + verification + abstention gate.

        Uses the SAME agreement, entropy, confidence inputs as Condition 2.
        Only the abstention decision (confidence >= tau) differs.
        """
        agreement, entropy, majority, raw_answers, sampling_valid = (
            self._uncertainty_from_record(record)
        )
        confidence, evidence_state = self._confidence_from_record(
            record, agreement, entropy, sampling_valid, localizer_score
        )
        selective = make_selective_prediction(
            sample_id=str(record.question_id),
            confidence=confidence,
            threshold=self.confidence_config.tau,
            answer=majority,
            evidence_state=evidence_state,
            sampling_valid=sampling_valid,
        )
        return ConditionResult(
            condition=3,
            question_id=record.question_id,
            image_id=record.image_id,
            track=record.track,
            model_name=record.model_name,
            generation_id=record.generation_id,
            agreement=agreement,
            normalized_entropy=entropy,
            confidence=confidence,
            evidence_state=evidence_state,
            sampling_valid=sampling_valid,
            majority_answer=majority,
            raw_answers=raw_answers,
            answer=selective.final_answer,
            is_abstaining=selective.is_abstaining,
            in_coverage=selective.in_coverage,
        )

    # -- resume / batch execution ------------------------------------------

    def _progress_path(self) -> Path:
        return (
            self.conditions_dir
            / f"progress_{self.model_name}_track_{self.track}.json"
        )

    def _load_progress(self, n_expected: int) -> GenerationProgress:
        path = self._progress_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return GenerationProgress(**data)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return GenerationProgress(
            model_name=self.model_name,
            track=self.track,
            condition_pair="2+3",
            n_samples_expected=n_expected,
        )

    def _save_progress(self, progress: GenerationProgress) -> None:
        progress.updated_at_utc = datetime.now(UTC).isoformat()
        path = self._progress_path()
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(progress.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)

    def run_batch(
        self,
        samples: list[dict[str, Any]],
        images: dict[str, Any],
        force_regenerate: bool = False,
    ) -> dict[str, Any]:
        """Run generation + condition evaluation for a batch of samples.

        Each ``samples`` entry must provide at least ``question_id``,
        ``image_id``, ``question``. Optionally ``region_ids`` (Track B oracle
        boxes) and ``localizer_score`` (rho). Images are keyed by image_id.

        Increments are persisted after every sample so a Colab disconnect can
        be resumed by re-running with the same inputs.
        """
        progress = self._load_progress(len(samples))
        condition_results: list[dict[str, Any]] = []

        for entry in samples:
            qid = entry["question_id"]
            iid = entry["image_id"]
            question = entry["question"]
            image = images.get(str(iid)) or images.get(iid)
            region_ids = entry.get("region_ids", [])
            localizer_score = entry.get("localizer_score")

            generation_id = compute_generation_id(
                model_name=self.model_name,
                model_revision=self.model_revision,
                track=self.track,
                question_id=qid,
                image_id=iid,
                n_samples=self.n_samples,
                temperature=self.temperature,
                random_seed=self.random_seed,
            )

            # Resume: skip valid completed records (no duplicate generation).
            if not force_regenerate and self.store.is_completed_valid(
                self.model_name, self.track, generation_id
            ):
                record = self.store.load(
                    self.model_name, self.track, generation_id
                )  # type: ignore[arg-type]
                progress.n_skipped += 1
                action = "skipped"
            else:
                record, action = self.generate_or_load(
                    question_id=qid,
                    image_id=iid,
                    question=question,
                    image=image,
                    region_ids=region_ids,
                    force_regenerate=force_regenerate,
                )
                if action == "skipped":
                    progress.n_skipped += 1
                elif action == "retried":
                    progress.n_retried += 1
                if record.status == GenerationStatus.FAILED:
                    progress.n_failed += 1
                    progress.failed_question_ids.append(qid)
                    progress.failures.append(
                        {
                            "question_id": qid,
                            "image_id": iid,
                            "generation_id": generation_id,
                            "error": record.error,
                            "at_utc": datetime.now(UTC).isoformat(),
                        }
                    )
                elif record.status == GenerationStatus.COMPLETED:
                    # Only count newly generated/retried completions; skipped
                    # records were already counted in a prior run.
                    if action != "skipped":
                        progress.n_completed += 1
                    progress.completed_question_ids.append(qid)

            # Detect corrupt artifacts.
            try:
                _ = self.store.load(self.model_name, self.track, generation_id)
            except ValueError:
                progress.n_corrupt += 1
                progress.corrupt_question_ids.append(qid)

            # Evaluate both conditions from the same persisted artifact (only
            # if generation succeeded). Condition 3 reuses Condition 2's inputs.
            if record.status == GenerationStatus.COMPLETED:
                cond2 = self.evaluate_condition_2(record, localizer_score)
                cond3 = self.evaluate_condition_3(record, localizer_score)
                condition_results.append(
                    {
                        "generation_id": generation_id,
                        "question_id": qid,
                        "image_id": iid,
                        "track": record.track,
                        "model_name": record.model_name,
                        "condition_2": {
                            "answer": cond2.answer,
                            "is_abstaining": cond2.is_abstaining,
                            "in_coverage": cond2.in_coverage,
                            "agreement": cond2.agreement,
                            "normalized_entropy": cond2.normalized_entropy,
                            "confidence": cond2.confidence,
                            "evidence_state": cond2.evidence_state,
                            "sampling_valid": cond2.sampling_valid,
                        },
                        "condition_3": {
                            "answer": cond3.answer,
                            "is_abstaining": cond3.is_abstaining,
                            "in_coverage": cond3.in_coverage,
                            "agreement": cond3.agreement,
                            "normalized_entropy": cond3.normalized_entropy,
                            "confidence": cond3.confidence,
                            "evidence_state": cond3.evidence_state,
                            "sampling_valid": cond3.sampling_valid,
                        },
                        "source_artifact_identical": True,
                    }
                )

            self._save_progress(progress)

        summary = {
            "model_name": self.model_name,
            "track": self.track,
            "n_expected": len(samples),
            "n_completed": progress.n_completed,
            "n_skipped": progress.n_skipped,
            "n_failed": progress.n_failed,
            "n_retried": progress.n_retried,
            "n_corrupt": progress.n_corrupt,
            "condition_results": condition_results,
            "progress": progress.to_dict(),
        }
        return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FS-005 Phase 5 generation harness (mock/dry-run by default)."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockVLMClient (no real model inference). For testing only.",
    )
    parser.add_argument(
        "--model",
        default="mock-vlm",
        help="Model name reported in artifacts.",
    )
    parser.add_argument(
        "--track",
        choices=["B", "C"],
        default="C",
        help="Evaluation track.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of sampled responses (N).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/runs/fs005_phase5",
        help="Output directory for artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    from vlm.mock_client import MockVLMClient

    vlm_client = MockVLMClient(model_name=args.model)

    # Synthetic fixture samples for demonstration (no real data used).
    fixture_samples = [
        {
            "question_id": "q_fixture_1",
            "image_id": "img_fixture_1",
            "question": "What modality is shown?",
            "region_ids": [],
        },
        {
            "question_id": "q_fixture_2",
            "image_id": "img_fixture_2",
            "question": "Is there an abnormality?",
            "region_ids": [],
        },
    ]
    images = {
        "img_fixture_1": None,  # Mock client ignores the image
        "img_fixture_2": None,
    }

    runner = GenerationRunner(
        vlm_client=vlm_client,
        model_name=args.model,
        model_revision=None,
        track=args.track,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        temperature=args.temperature,
    )
    summary = runner.run_batch(fixture_samples, images)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
