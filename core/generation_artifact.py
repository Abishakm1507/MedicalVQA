"""Condition-independent generation-artifact system for FS-005 Phase 5.

Persists VLM generated responses independently of evaluation conditions so
that Conditions 2 and 3 can share the same N=5 generated responses. Each
generation record carries a deterministic, condition-independent identifier
(derived from model, track, question, image, sampling parameters, and the Track
B input mode; the answer condition is intentionally excluded) so completed
records can be recognized reliably across restarts and Colab disconnects.

Records are validated before they are allowed into evaluation. The store
persists completed samples incrementally, skips valid completed records on
restart, retries failed/corrupt records without overwriting valid results,
and prevents accidental duplicate generation.

Traceability:
  - Feature Specification: FS-005 (Phase 5 execution harness)
  - ADR 0005: N=5 sampling standard; confidence weights/threshold configurable
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator


class GenerationStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class ValidationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    NOT_VALIDATED = "not_validated"


GroundingMode = Literal["oracle_gt_xywh_overlay_v1", "original_unmarked_image"]


def _grounding_mode_for_track(track: str) -> GroundingMode:
    if track == "B":
        return "oracle_gt_xywh_overlay_v1"
    if track == "C":
        return "original_unmarked_image"
    raise ValueError(f"Unsupported generation track: {track!r}")


class GenerationSample(BaseModel):
    """A single sampled response within a generation record."""

    sample_index: int = Field(ge=0, description="Zero-based index of the sample within N.")
    raw_response: str | None = Field(
        default=None, description="Raw model response string (may be None on failure)."
    )
    parsed_answer: str | None = Field(
        default=None, description="Parsed answer text extracted from the raw response."
    )
    claims: list[dict[str, Any]] = Field(
        default_factory=list, description="Extracted/verified claims with region grounding."
    )
    inference_duration_s: float | None = Field(
        default=None, ge=0, description="Wall-clock inference duration for this sample in seconds."
    )
    timestamp_utc: str | None = Field(
        default=None, description="ISO-8601 UTC timestamp when this sample was generated."
    )
    error: str | None = Field(
        default=None, description="Error message if this individual sample failed."
    )
    status: Literal["ok", "failed"] = Field(
        default="ok", description="Per-sample generation status."
    )


class GenerationRecord(BaseModel):
    """A condition-independent generation artifact for one (model, track, question).

    The ``generation_id`` is deterministic and excludes the answer condition,
    so Conditions 2 and 3 (and Condition 1 when N matches) share the same
    underlying generated responses.
    """

    schema_version: str = Field(default="1.0", description="Artifact schema version.")
    generation_id: str = Field(
        description="Deterministic, condition-independent generation identifier."
    )
    model_name: str = Field(description="Model name (e.g. 'medgemma-4b', 'qwen2.5-vl-3b').")
    model_revision: str | None = Field(
        default=None, description="Checkpoint / revision identifier (e.g. HF revision)."
    )
    track: Literal["B", "C"] = Field(description="Evaluation track identifier (B or C).")
    grounding_mode: GroundingMode | None = Field(
        default=None,
        description="Input grounding presentation used for this generation.",
    )
    question_id: str | int = Field(description="Question ID from the cohort.")
    image_id: str | int = Field(description="Image ID the question belongs to.")
    n_samples: int = Field(ge=1, description="Number of sampled responses (N).")
    temperature: float = Field(ge=0.0, le=1.0, description="Sampling temperature.")
    random_seed: int | None = Field(
        default=None, description="Random seed used for sampling (recorded for reproducibility)."
    )
    samples: list[GenerationSample] = Field(
        description="Individual raw responses, parsed answers, and claims for each sample."
    )
    status: GenerationStatus = Field(
        default=GenerationStatus.PENDING, description="Overall generation status."
    )
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.NOT_VALIDATED,
        description="Schema/business-rule validation status.",
    )
    total_duration_s: float | None = Field(
        default=None, ge=0, description="Total wall-clock duration for all N samples in seconds."
    )
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 UTC creation timestamp.",
    )
    updated_at_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 UTC last-update timestamp.",
    )
    error: str | None = Field(
        default=None, description="Error message if the overall generation failed."
    )

    @model_validator(mode="after")
    def _validate_sample_count_matches_n(self) -> GenerationRecord:
        expected_grounding_mode = _grounding_mode_for_track(self.track)
        if self.grounding_mode is None:
            self.grounding_mode = expected_grounding_mode
        elif self.grounding_mode != expected_grounding_mode:
            raise ValueError(
                f"grounding_mode {self.grounding_mode!r} does not match track "
                f"{self.track!r} (expected {expected_grounding_mode!r})"
            )
        if len(self.samples) != self.n_samples:
            raise ValueError(
                f"samples length ({len(self.samples)}) must equal n_samples ({self.n_samples})"
            )
        if self.status == GenerationStatus.COMPLETED and self.error is not None:
            raise ValueError("A completed record must not carry a generation error.")
        if self.status == GenerationStatus.FAILED and self.error is None:
            raise ValueError("A failed record must carry an error message.")
        return self


# ---------------------------------------------------------------------------
# Deterministic identifier
# ---------------------------------------------------------------------------


def compute_generation_id(
    model_name: str,
    model_revision: str | None,
    track: str,
    question_id: str | int,
    image_id: str | int,
    n_samples: int,
    temperature: float,
    random_seed: int | None,
) -> str:
    """Compute a deterministic, condition-independent generation identifier.

    The identifier is a SHA-256 digest over the canonical JSON of the
    identifying fields and, for Track B, the oracle-overlay input version.
    The answer condition is intentionally excluded so Conditions 2 and 3
    reuse the same generation artifact.
    """
    payload: dict[str, Any] = {
        "model_name": model_name,
        "model_revision": model_revision,
        "track": track,
        "question_id": str(question_id),
        "image_id": str(image_id),
        "n_samples": n_samples,
        "temperature": temperature,
        "random_seed": random_seed,
    }
    if track == "B":
        payload["grounding_mode"] = _grounding_mode_for_track(track)
    elif track != "C":
        raise ValueError(f"Unsupported generation track: {track!r}")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_generation_record(record: GenerationRecord) -> list[str]:
    """Validate a generation record against business rules.

    Returns a list of human-readable violation messages. An empty list means
    the record is valid and may be admitted into evaluation.

    Rules checked:
    - schema_version is supported
    - sample count equals n_samples
    - every sample has a unique, contiguous sample_index starting at 0
    - completed records have no error; failed records have an error
    - timestamps are parseable ISO-8601 UTC
    - durations are non-negative when present
    """
    violations: list[str] = []

    if record.schema_version != "1.0":
        violations.append(f"Unsupported schema_version: {record.schema_version!r}")

    if len(record.samples) != record.n_samples:
        violations.append(
            f"samples length ({len(record.samples)}) != n_samples ({record.n_samples})"
        )

    expected_indices = list(range(record.n_samples))
    actual_indices = [s.sample_index for s in record.samples]
    if sorted(actual_indices) != expected_indices:
        violations.append(
            f"sample_index values {actual_indices} are not contiguous 0..{record.n_samples - 1}"
        )
    if len(actual_indices) != len(set(actual_indices)):
        violations.append("Duplicate sample_index values present")

    for s in record.samples:
        if s.status == "failed" and s.error is None:
            violations.append(f"Sample {s.sample_index} marked failed with no error")
        if s.status == "ok" and s.raw_response is None:
            violations.append(f"Sample {s.sample_index} marked ok with no raw_response")
        if s.inference_duration_s is not None and s.inference_duration_s < 0:
            violations.append(f"Sample {s.sample_index} has negative duration")

    if record.status == GenerationStatus.COMPLETED and record.error is not None:
        violations.append("Completed record carries a generation error")
    if record.status == GenerationStatus.FAILED and record.error is None:
        violations.append("Failed record carries no error message")

    for field_name in ("created_at_utc", "updated_at_utc"):
        value = getattr(record, field_name)
        try:
            datetime.fromisoformat(value)
        except (TypeError, ValueError):
            violations.append(f"{field_name} is not parseable ISO-8601: {value!r}")

    if record.total_duration_s is not None and record.total_duration_s < 0:
        violations.append("total_duration_s is negative")

    return violations


# ---------------------------------------------------------------------------
# Persistence store
# ---------------------------------------------------------------------------


class GenerationArtifactStore:
    """File-system store for generation artifacts.

    Layout:
        <root>/<model_name>/<track>/<generation_id>.json

    One JSON file per generation record. Records are written atomically
    (temp file + replace) so partial writes cannot corrupt an existing record.
    """

    def __init__(self, root: str | Path = "experiments/runs/fs005_phase5/generations") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- path helpers -------------------------------------------------------

    def _record_dir(self, model_name: str, track: str) -> Path:
        d = self.root / model_name / track
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _record_path(self, model_name: str, track: str, generation_id: str) -> Path:
        return self._record_dir(model_name, track) / f"{generation_id}.json"

    # -- CRUD ---------------------------------------------------------------

    def save(self, record: GenerationRecord, validate: bool = True) -> Path:
        """Persist a generation record.

        Raises ValueError if validation fails and ``validate`` is True. The
        record's ``updated_at_utc`` is refreshed before writing. An existing
        valid COMPLETED record is never overwritten (call ``exists`` first to
        check).
        """
        if validate:
            violations = validate_generation_record(record)
            if violations:
                raise ValueError("Invalid generation record: " + "; ".join(violations))

        # Mark the record as validated before persisting so that
        # is_completed_valid recognizes it on resume.
        record.validation_status = ValidationStatus.VALID
        record.updated_at_utc = datetime.now(UTC).isoformat()
        path = self._record_path(record.model_name, record.track, record.generation_id)
        tmp = path.with_suffix(".json.tmp")
        payload = record.model_dump(mode="json")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)

        # Verify-after-write: re-read and re-validate the persisted artifact so
        # a truncated or unreadable write is detected immediately (critical for
        # persistent Drive storage where a resume must trust prior writes).
        self._verify_artifact(path)
        return path

    def _verify_artifact(self, path: Path) -> None:
        """Re-read and validate a persisted artifact; raise ValueError on any
        corruption or incompatibility."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        record = GenerationRecord.model_validate(raw)
        violations = validate_generation_record(record)
        if violations:
            raise ValueError(
                f"Persisted artifact {path} failed validation: "
                + "; ".join(violations)
            )

    def load(
        self, model_name: str, track: str, generation_id: str
    ) -> GenerationRecord | None:
        """Load and validate a generation record.

        Returns None if the file does not exist. Raises ValueError if the
        file exists but is corrupt, unparseable, or fails validation.
        """
        path = self._record_path(model_name, track, generation_id)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Corrupt or unparseable artifact {path}: {exc}") from exc
        try:
            record = GenerationRecord.model_validate(raw)
        except ValidationError as exc:
            raise ValueError(f"Artifact {path} failed schema validation: {exc}") from exc
        violations = validate_generation_record(record)
        if violations:
            raise ValueError(
                f"Artifact {path} failed business validation: " + "; ".join(violations)
            )
        return record

    def exists(self, model_name: str, track: str, generation_id: str) -> bool:
        """Return True if an artifact file exists for the given key."""
        return self._record_path(model_name, track, generation_id).exists()

    def list_records(self, model_name: str, track: str) -> list[str]:
        """Return sorted generation IDs stored for a model/track."""
        d = self._record_dir(model_name, track)
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.json"))

    # -- status queries -----------------------------------------------------

    def status(self, model_name: str, track: str, generation_id: str) -> str | None:
        """Return the record status, or None if no artifact exists."""
        record = self.load(model_name, track, generation_id)
        return record.status if record is not None else None

    def is_completed_valid(
        self, model_name: str, track: str, generation_id: str
    ) -> bool:
        """Return True if a valid COMPLETED artifact exists for the key."""
        try:
            record = self.load(model_name, track, generation_id)
        except ValueError:
            return False
        return (
            record is not None
            and record.status == GenerationStatus.COMPLETED
            and record.validation_status == ValidationStatus.VALID
        )
