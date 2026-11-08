"""Inventory a captured GSPC board without promoting it to a new Kaggle result.

Receipt validation belongs to the caller's independently pinned verifier. This
module never grades a response, admits a board cell, or validates a signature.
"""

from collections import Counter
from collections.abc import Callable, Mapping
from datetime import datetime
import hashlib
from typing import Any
from urllib.parse import urlsplit

from .artifacts import strict_json

BOARD_SCHEMA = "csoai.gspc-axes/0.5"
KINDS = frozenset({"model-comparison", "deterministic-facts", "measured-subject"})
EXECUTION_STATES = frozenset({"RUNTIME_OBSERVED", "REPRODUCED"})
DEFAULT_SOURCE = "https://councilof.ai/api/gspc"


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _execution(axis: str, evidence: Any, supported: bool,
               validator: Callable[[str, Mapping], bool] | None) -> dict:
    result = {"measurement_status": "UNMEASURED", "execution_status": "NOT_RUN",
              "reported_status": None, "receipt_id": None,
              "receipt_validation": "NOT_PROVIDED"}
    if evidence is None:
        return result
    if not isinstance(evidence, Mapping):
        raise ValueError(f"execution evidence for {axis!r} must be an object")
    result.update(execution_status="REPORTED_ONLY",
                  reported_status=_text(evidence.get("status")),
                  receipt_id=_text(evidence.get("receipt_id")),
                  receipt_validation="NOT_VALIDATED")
    receipt = evidence.get("receipt")
    if not supported:
        result["receipt_validation"] = "UNSUPPORTED_AXIS_KIND"
    elif result["reported_status"] not in EXECUTION_STATES:
        result["receipt_validation"] = "UNSUPPORTED_EXECUTION_STATUS"
    elif not isinstance(receipt, Mapping) or not receipt:
        result["receipt_validation"] = "MISSING_RECEIPT"
    elif validator is not None:
        try:
            valid = validator(axis, evidence)
        except Exception:
            result["receipt_validation"] = "UNCHECKABLE"
        else:
            result["receipt_validation"] = "VALIDATED" if valid is True else "REJECTED"
            if valid is True:
                result["execution_status"] = result["reported_status"]
    return result


def build_coverage(captured_payload: bytes, *, observed_at: str,
                   source_url: str = DEFAULT_SOURCE,
                   execution_evidence: Mapping | None = None,
                   validate_receipt: Callable[[str, Mapping], bool] | None = None) -> dict:
    """Return one row per captured axis; all measurement states start UNMEASURED.

    ``observed_at`` is caller-supplied capture time, not board measurement time.
    The SHA-256 binds the exact supplied bytes, not a reserialized JSON object.
    A validator must return the literal bool True after verifying the receipt's
    axis, subject, method, inputs, outputs and claimed execution state. Even a
    validated replay does not constitute independent measurement admission.
    """
    if not isinstance(captured_payload, bytes) or len(captured_payload) > 5_000_000:
        raise ValueError("capture must be bytes of at most 5 MB")
    try:
        stamp = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("observed_at must be an ISO 8601 timestamp") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("observed_at must include a timezone")
    parsed_url = urlsplit(source_url)
    if parsed_url.scheme != "https" or not parsed_url.netloc or parsed_url.username:
        raise ValueError("source_url must be an HTTPS URL without credentials")
    payload = strict_json(captured_payload)
    if not isinstance(payload, dict) or payload.get("schema") != BOARD_SCHEMA:
        raise ValueError("unsupported board schema")
    axes = payload.get("axes")
    if not isinstance(axes, list):
        raise ValueError("board axes must be a list")
    evidence = {} if execution_evidence is None else execution_evidence
    if not isinstance(evidence, Mapping) or any(not isinstance(k, str) for k in evidence):
        raise ValueError("execution_evidence must map axis IDs to objects")
    rows, seen = [], set()
    for axis in axes:
        if not isinstance(axis, dict) or not _text(axis.get("axis")):
            raise ValueError("every axis requires a nonempty axis ID")
        axis_id = axis["axis"]
        if axis_id in seen:
            raise ValueError(f"duplicate axis ID: {axis_id}")
        seen.add(axis_id)
        kind = _text(axis.get("kind"))
        supported = kind in KINDS
        rows.append({"axis": axis_id, "family": _text(axis.get("family")),
                     "declared_kind": kind,
                     "execution_family": kind if supported else "UNSUPPORTED",
                     "benchmark": _text(axis.get("bench")),
                     "board_status": _text(axis.get("status")),
                     "this_run": _execution(axis_id, evidence.get(axis_id),
                                            supported, validate_receipt)})
    return {"schema": "meok.kaggle-coverage/1", "board_schema": BOARD_SCHEMA,
            "source_url": source_url, "source_sha256": hashlib.sha256(captured_payload).hexdigest(),
            "observed_at": observed_at, "axis_count": len(rows),
            "counts_by_execution_family": dict(sorted(Counter(
                row["execution_family"] for row in rows).items())),
            "unused_evidence_axes": sorted(set(evidence) - seen), "axes": rows,
            "limitations": ["Inventory of a captured board, not a live fetch.",
                            "Board scores are not results of this run.",
                            "Validated replay does not admit or sign a GSPC measurement."]}
