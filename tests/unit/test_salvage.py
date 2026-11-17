"""Salvage rule for samples that failed the strict JSON contract."""

from core.generation_artifact import GenerationRecord, GenerationSample, GenerationStatus
from scripts.fs005_phase5_salvage import SALVAGE_STATS, salvage_record, salvage_response

OK = '```json\n{"answer": "Yes", "findings": [{"claim": "a", "region_id": "R1", "status": "present", "confidence": "high"}]}\n```'
TRUNC = (
    '```json\n{\n  "answer": "The liver",\n  "findings": [\n'
    '    {"claim": "liver", "region_id": "R1", "status": "present", "confidence": "high"},\n'
    '    {"claim": "spl'
)
POSSIBLE = (
    '{"answer": "Cardiomegaly", "findings": ['
    '{"claim": "a", "region_id": "R1", "status": "possible", "confidence": "high"},'
    '{"claim": "b", "region_id": "R9", "status": "present", "confidence": "low"}]}'
)


def test_truncated_reply_keeps_answer_and_complete_findings_only() -> None:
    r = salvage_response(TRUNC, {"R1", "NONE"})
    assert r["answer"] == "The liver"
    assert [c["claim"] for c in r["claims"]] == ["liver"]


def test_out_of_schema_status_is_dropped_not_remapped() -> None:
    r = salvage_response(POSSIBLE, {"R1", "NONE"})
    assert r["answer"] == "Cardiomegaly"
    assert r["dropped_findings"] == 1
    assert [(c["claim"], c["region_id"]) for c in r["claims"]] == [("b", "NONE")]


def test_no_answer_is_unusable() -> None:
    assert salvage_response("Sure! It looks like a lung.") is None
    assert salvage_response(None) is None
    assert salvage_response('{"answer": ""}') is None


def _rec(raws: list[str], n: int = 5) -> GenerationRecord:
    samples = []
    for i, raw in enumerate(raws):
        good = raw == OK
        samples.append(
            GenerationSample(
                sample_index=i,
                raw_response=raw,
                parsed_answer="Yes" if good else None,
                claims=[],
                status="ok" if good else "failed",
                error=None if good else "parse error",
            )
        )
    return GenerationRecord(
        generation_id="g", model_name="m", track="C", question_id=1, image_id=1,
        n_samples=n, temperature=0.7, samples=samples,
        status=GenerationStatus.FAILED, error="x",
    )


def test_record_kept_when_three_of_five_usable_else_excluded() -> None:
    SALVAGE_STATS.clear()
    kept = salvage_record(_rec([OK, OK, TRUNC, "garbage", "garbage"]), {"NONE", "R1"})
    assert kept is not None and kept.status == GenerationStatus.COMPLETED
    assert [s.parsed_answer for s in kept.samples] == ["Yes", "Yes", "The liver"]
    assert salvage_record(_rec([OK, TRUNC, "garbage", "garbage", "garbage"])) is None
    assert SALVAGE_STATS["records_excluded"] == 1 and SALVAGE_STATS["records_salvaged"] == 1


def test_greedy_single_sample_needs_one_usable() -> None:
    assert salvage_record(_rec([TRUNC], n=1)) is not None
    assert salvage_record(_rec(["garbage"], n=1)) is None
