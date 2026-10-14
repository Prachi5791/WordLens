

from __future__ import annotations

from html import unescape

from backend.app.core.errors import WordLensError
from backend.app.repositories.terminology import TerminologyRepository, normalize
from scripts.dynamic_knowledge import resolve_term, validate_contextual_usage


def _usable_definition(value) -> bool:
    return len(unescape(str(value or "")).strip()) >= 30


def _record_knowledge(record) -> dict:
    return {
        "term_id": record.term_id,
        "term": record.term,
        "canonical_term": record.term,
        "aliases": record.aliases or [],
        "domain": record.domain,
        "subdomain": record.subdomain,
        "definition": record.definition,
        "source": record.source,
        "source_url": record.source_url,
        "confidence": record.confidence,
        "knowledge_status": "local",
    }


def lookup_term(
    repository: TerminologyRepository,
    requested_term: str,
    context: str | None = None,
) -> tuple[object | None, str, dict]:
    requested_term = (requested_term or "").strip()
    if not requested_term:
        raise WordLensError(error_code="INVALID_TERM", message="Term cannot be empty.", status_code=422)
    if len(requested_term) > 512:
        raise WordLensError(error_code="INVALID_TERM", message="Term is too long.", status_code=422)

    normalized = normalize(requested_term)
    if not normalized:
        raise WordLensError(error_code="INVALID_TERM", message="Term cannot be empty after normalization.", status_code=422)

    record = repository.get_exact(normalized)
    matched_by = "canonical"
    if record is None:
        record = repository.get_by_alias(normalized)
        matched_by = "alias"

    # Preserve the old reliable behavior for ordinary /term lookups and for
    # known terms without PDF context. Context validation is only applied when
    # the caller actually supplied paper context.
    if record is not None and _usable_definition(record.definition):
        if not context:
            return record, matched_by, {
                "technical_status": "technical",
                "decision": "resolved",
                "reason_code": "LOCAL_RESOLVED",
                "reason": "The term was resolved from the local verified knowledge base.",
                "knowledge_status": "local",
                "persisted": False,
                "created": False,
            }

        validation = validate_contextual_usage(
            requested_term,
            context,
            _record_knowledge(record),
        )
        if validation.get("valid"):
            return record, matched_by, {
                "technical_status": "technical",
                "decision": "resolved",
                "reason_code": validation.get("reason_code", "LOCAL_RESOLVED"),
                "reason": validation.get("reason", "The local definition matches the PDF context."),
                "knowledge_status": "local",
                "persisted": False,
                "created": False,
            }

    # Context-specific dynamic resolution is allowed to find a better canonical
    # page (e.g. Kafka -> Apache Kafka) rather than blindly returning an unrelated
    # local one-word record.
    dynamic = resolve_term(
        requested_term,
        context=context,
        persist=True,
    )

    if dynamic.get("decision") == "resolved":
        dynamic_term = normalize(str(dynamic.get("term") or requested_term))
        persisted = repository.get_exact(dynamic_term) or repository.get_by_alias(dynamic_term)
        if persisted is not None and _usable_definition(persisted.definition):
            return persisted, "external" if record is None else "enriched", dynamic

        # The dynamic resolver may have committed through another session.
        return None, "external", dynamic

    if record is not None and not context:
        return record, matched_by, dynamic

    return None, "none", dynamic


def serialize_term(record, matched_by: str, requested_term: str, decision: dict | None = None) -> dict:
    decision = decision or {}
    definition = record.definition if record is not None else decision.get("definition")
    if definition:
        definition = unescape(str(definition)).strip()
    return {
        "term_id": record.term_id if record is not None else decision.get("term_id"),
        "term": record.term if record is not None else decision.get("term", requested_term),
        "aliases": (record.aliases or []) if record is not None else decision.get("aliases", []),
        "domain": record.domain if record is not None else decision.get("domain"),
        "subdomain": record.subdomain if record is not None else decision.get("subdomain"),
        "definition": definition,
        "simplified_definition": record.simplified_definition if record is not None else decision.get("simplified_definition"),
        "confidence": record.confidence if record is not None else decision.get("confidence"),
        "source": record.source if record is not None else decision.get("source"),
        "source_url": record.source_url if record is not None else decision.get("source_url"),
        "matched_by": matched_by,
        "requested_term": requested_term,
        "technical_status": decision.get("technical_status", "technical" if record is not None else "unknown"),
        "decision": decision.get("decision", "resolved" if record is not None else "unresolved"),
        "reason_code": decision.get("reason_code"),
        "reason": decision.get("reason"),
        "knowledge_status": decision.get("knowledge_status"),
        "persisted": decision.get("persisted", False),
        "created": decision.get("created", False),
    }
