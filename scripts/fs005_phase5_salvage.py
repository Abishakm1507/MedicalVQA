"""Offline salvage of generation samples that failed the strict JSON contract.

Why this exists
---------------
The generation harness marks a sample ``failed`` if its reply does not validate
against ``VLMStructuredOutput`` and marks the whole question ``failed`` if any of
its samples failed. In the paper run this dropped ~8-10 % of questions per arm,
and the dropped questions are not random: replies cut off by the 256-token cap
(long answers) and replies using an out-of-schema ``status`` (e.g. "possible").
Dropping them would bias accuracy, agreement and calibration.

The raw replies are stored in the artifacts, so this module re-reads them on the
CPU with one fixed rule, applied identically to every model, track and split
(validation and test), with no model calls and no per-question judgement:

1. A reply is usable if it contains a top-level ``"answer"`` string, even when
   the JSON is cut off afterwards.
2. Findings are kept only if they are complete JSON objects with a valid
   ``status`` (present|absent|uncertain), ``confidence`` (high|medium|low) and a
   string ``claim``. Invalid or cut-off findings are DROPPED, never remapped or
   completed. A region id outside the question's candidate set becomes NONE,
   exactly as the harness does.
3. A question is kept if at least ``MIN_VALID`` samples (3 of 5; 1 of 1 for the
   greedy run) are usable; otherwise it stays excluded and is counted.

Samples that cannot be salvaged are removed from the question's sample list, so
agreement/entropy are computed over the usable samples only.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from core.generation_artifact import GenerationRecord, GenerationStatus

STATUSES = {"present", "absent", "uncertain"}
CONFIDENCES = {"high", "medium", "low"}
MIN_VALID_SAMPLES = 3

# Module-level counters so every stage can report how many records were touched.
SALVAGE_STATS: Counter = Counter()

_ANSWER_RE = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"')
_FINDINGS_RE = re.compile(r'"findings"\s*:\s*\[')


def salvage_response(raw: str | None, valid_region_ids: set[str] | None = None) -> dict | None:
    """Return {"answer", "claims", "dropped_findings"} or None if unusable."""
    if not raw:
        return None
    m = _ANSWER_RE.search(raw)
    if not m:
        return None
    try:
        answer = json.loads('"' + m.group(1) + '"')
    except json.JSONDecodeError:
        return None
    if not answer.strip():
        return None

    claims: list[dict[str, Any]] = []
    dropped = 0
    fm = _FINDINGS_RE.search(raw, m.end()) or _FINDINGS_RE.search(raw)
    if fm:
        decoder = json.JSONDecoder()
        i = fm.end()
        while True:
            j = raw.find("{", i)
            if j == -1:
                break
            try:
                obj, end = decoder.raw_decode(raw, j)
            except json.JSONDecodeError:
                break  # cut-off tail: nothing more can be read safely
            i = end
            if (
                isinstance(obj, dict)
                and isinstance(obj.get("claim"), str)
                and obj.get("status") in STATUSES
                and obj.get("confidence") in CONFIDENCES
            ):
                rid = str(obj.get("region_id", "NONE"))
                if valid_region_ids is not None and rid not in valid_region_ids:
                    rid = "NONE"
                claims.append(
                    {
                        "claim": obj["claim"],
                        "region_id": rid,
                        "status": obj["status"],
                        "confidence": obj["confidence"],
                    }
                )
            else:
                dropped += 1
    return {"answer": answer, "claims": claims, "dropped_findings": dropped}


def salvage_record(
    record: GenerationRecord, valid_region_ids: set[str] | None = None
) -> GenerationRecord | None:
    """Return a usable COMPLETED copy of ``record`` or None if too few samples survive."""
    if record.status == GenerationStatus.COMPLETED:
        return record
    kept = []
    n_salvaged = 0
    for s in record.samples:
        if s.status == "ok":
            kept.append(s)
            continue
        res = salvage_response(s.raw_response, valid_region_ids)
        if res is None:
            continue
        n_salvaged += 1
        kept.append(
            s.model_copy(
                update={
                    "status": "ok",
                    "parsed_answer": res["answer"],
                    "claims": res["claims"],
                    "error": None,
                }
            )
        )
    need = min(MIN_VALID_SAMPLES, record.n_samples)
    SALVAGE_STATS["records_with_failures"] += 1
    if len(kept) < need:
        SALVAGE_STATS["records_excluded"] += 1
        return None
    SALVAGE_STATS["records_salvaged"] += 1
    SALVAGE_STATS["samples_salvaged"] += n_salvaged
    SALVAGE_STATS["samples_dropped"] += len(record.samples) - len(kept)
    return record.model_copy(
        update={"samples": kept, "status": GenerationStatus.COMPLETED, "error": None}
    )
