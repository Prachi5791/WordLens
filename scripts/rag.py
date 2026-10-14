
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from html import unescape
from typing import Any, Optional

from scripts.dynamic_knowledge import validate_contextual_usage
from scripts.retrieval import retrieve
from scripts.scientific_term_discovery import classify_rejection


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", unescape(str(value or ""))).strip()


def _context_sentence(term: str, context: str | None = None) -> str | None:
    value = _clean(context)
    return value or None


def _definition_sentence(definition: str | None) -> str:
    text = _clean(definition)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    for part in parts:
        if len(part.strip()) >= 20:
            return part.strip().rstrip(".!?") + "."
    return text.rstrip(".!?") + "."


def _definition_subject_body(definition: str, canonical: str) -> tuple[str, str]:
    sentence = _definition_sentence(definition).rstrip(".")
    if not sentence:
        return canonical, ""
    match = re.match(
        rf"^{re.escape(canonical)}\s+(?:is|are|refers to|means|denotes)\s+(.+)$",
        sentence,
        flags=re.I,
    )
    if match:
        return canonical, match.group(1).strip()
    return canonical, sentence


def _contextual_explanation(selected_term: str, context: str | None, definition: str, canonical_term: str | None = None) -> str:
    requested = _clean(selected_term)
    canonical = _clean(canonical_term) or requested
    definition = _clean(definition)
    context = _clean(context)
    if not definition:
        return ""

    _, body = _definition_subject_body(definition, canonical)
    body = body.rstrip(".!?")
    if not context:
        return f"{canonical} refers to {body}." if body else _definition_sentence(definition)

    escaped = re.escape(requested).replace(r"\ ", r"\s+")
    match = re.search(rf"(?<!\w){escaped}(?!\w)", context, flags=re.I)
    if not match and canonical:
        escaped_canonical = re.escape(canonical).replace(r"\ ", r"\s+")
        match = re.search(rf"(?<!\w){escaped_canonical}(?!\w)", context, flags=re.I)

    if not match:
        return ""

    # Preserve the actual paper sentence as the usage statement, then attach only
    # the verified canonical meaning. This avoids the previous 'is described as is'
    # grammar bug and avoids inventing a new claim about the paper.
    sentence = context.rstrip(".!?")
    if canonical.casefold() != requested.casefold():
        return f'In this paper, “{requested}” refers to {canonical}. The paper uses it in: “{sentence}.”'

    return f'In this paper, “{requested}” refers to {body}. It is used in: “{sentence}.”' if body else f'In this paper, “{requested}” is used in: “{sentence}.”'


def build_sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for item in results:
        key = (item.get("term_id"), item.get("source_url"))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "term_id": item.get("term_id"),
            "term": item.get("term"),
            "source": item.get("source"),
            "source_url": item.get("source_url"),
            "rank": item.get("rank"),
            "final_score": item.get("final_score"),
        })
    return sources


def _base_response(selected_term: str | None, context: str | None, detection: dict | None, minimum_score: float) -> dict[str, Any]:
    return {
        "request_id": uuid.uuid4().hex[:16],
        "selected_term": selected_term,
        "context": context,
        "detection_id": (detection or {}).get("detection_id"),
        "retrieval_query": None,
        "retrieval_status": "not_requested",
        "grounding_status": "insufficient_evidence",
        "generation_method": "none",
        "generated_explanation": None,
        "canonical_definition": None,
        "retrieved_evidence": [],
        "sources": [],
        "top_score": None,
        "retrieval_threshold": minimum_score,
        "warnings": [],
        "created_at": utc_timestamp(),
    }


def _abstain(selected_term: str, context: str | None, result: dict[str, Any], minimum_score: float, reason: str, detection: dict | None = None) -> dict[str, Any]:
    response = _base_response(selected_term, context, detection, minimum_score)
    evidence = result.get("results", [])
    response.update({
        "retrieval_query": result.get("retrieval_query"),
        "retrieval_status": result.get("retrieval_status", "success"),
        "grounding_status": result.get("grounding_status", "insufficient_evidence"),
        "retrieved_evidence": evidence,
        "sources": build_sources(evidence),
        "top_score": evidence[0].get("final_score") if evidence else None,
        "warnings": [reason],
        "reason_code": "INSUFFICIENT_EVIDENCE",
        "reason": reason,
    })
    return response


def _knowledge_evidence(knowledge: dict[str, Any]) -> dict[str, Any]:
    return {
        "faiss_id": None,
        "term_id": knowledge.get("term_id"),
        "term": knowledge.get("term"),
        "aliases": knowledge.get("aliases", []),
        "domain": knowledge.get("domain"),
        "subdomain": knowledge.get("subdomain"),
        "definition": knowledge.get("definition"),
        "source": knowledge.get("source"),
        "source_url": knowledge.get("source_url"),
        "semantic_score": 1.0,
        "lexical_boost": 0.0,
        "final_score": 1.0,
        "matched_selected_term": True,
        "lexical_match_type": knowledge.get("knowledge_status", "local"),
        "rank": 1,
        "knowledge_resolution": knowledge.get("knowledge_status", "local"),
    }


def _knowledge_is_contextual(term: str, context: str | None, knowledge: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    validation = validate_contextual_usage(term, context, knowledge)
    return bool(validation.get("valid")), validation


def run_rag(
    selected_term: Optional[str],
    context: Optional[str] = None,
    top_k: int = 5,
    minimum_score: float = 0.45,
    detection: Optional[dict[str, Any]] = None,
    knowledge: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    term = _clean(selected_term)
    context_clean = _context_sentence(term, context)
    if not term:
        return _abstain("", context_clean, {"results": []}, minimum_score, "No technical term was provided.", detection)

    # A verified local/dynamic record is the first source of truth. Do not let the
    # generic-word classifier destroy an already verified technical term.
    if knowledge and knowledge.get("definition") and len(_clean(knowledge.get("definition"))) >= 30:
        valid, validation = _knowledge_is_contextual(term, context_clean, knowledge)
        if valid:
            evidence = _knowledge_evidence(knowledge)
            canonical = _clean(knowledge.get("term")) or term
            explanation = _contextual_explanation(term, context_clean, evidence["definition"], canonical)
            if explanation or not context_clean:
                response = _base_response(term, context_clean, detection, minimum_score)
                response.update({
                    "retrieval_query": f"Selected term: {term}\nResearch context: {context_clean or 'Not provided'}",
                    "retrieval_status": "knowledge_base",
                    "grounding_status": "grounded" if context_clean else "grounded",
                    "generation_method": "context_evidence_template",
                    "generated_explanation": explanation or f"{canonical} refers to {_definition_sentence(evidence['definition'])}",
                    "canonical_definition": evidence["definition"],
                    "retrieved_evidence": [evidence],
                    "sources": build_sources([evidence]),
                    "top_score": 1.0,
                    "warnings": [],
                    "reason_code": "KNOWLEDGE_RESOLVED",
                    "reason": validation.get("reason", "Verified terminology knowledge was compatible with the PDF context."),
                })
                return response

    # Do not allow a generic-word rejection to erase a retrieval path. Retrieval
    # already combines lexical and semantic evidence and is authoritative for RAG.
    try:
        retrieval = retrieve(
            selected_term=term,
            context=context_clean,
            top_k=top_k,
            minimum_score=minimum_score,
            detection=detection,
        )
    except Exception as exc:
        response = _base_response(term, context_clean, detection, minimum_score)
        response.update({
            "retrieval_status": "error",
            "grounding_status": "retrieval_error",
            "warnings": [str(exc)],
            "reason_code": "RETRIEVAL_ERROR",
            "reason": str(exc),
        })
        return response

    evidence = retrieval.get("results", [])
    if retrieval.get("retrieval_status") == "error":
        return _abstain(term, context_clean, retrieval, minimum_score, retrieval.get("error", "Retrieval failed."), detection)

    if retrieval.get("grounding_status") != "grounded" or not evidence:
        return _abstain(term, context_clean, retrieval, minimum_score, retrieval.get("grounding_reason", "Retrieved evidence was insufficient to generate a grounded explanation."), detection)

    primary = evidence[0]
    definition = _clean(primary.get("definition"))
    if len(definition) < 30:
        return _abstain(term, context_clean, retrieval, minimum_score, "The retrieved record has no usable definition.", detection)

    # Retrieval evidence must also be compatible with the actual paper context.
    valid, validation = _knowledge_is_contextual(term, context_clean, primary)
    if not valid:
        return _abstain(term, context_clean, retrieval, minimum_score, validation.get("reason", "Retrieved evidence does not match the paper context."), detection)

    explanation = _contextual_explanation(term, context_clean, definition, _clean(primary.get("term")) or term)
    if not explanation:
        return _abstain(term, context_clean, retrieval, minimum_score, "A grounded explanation could not be constructed from the supplied PDF context.", detection)

    response = _base_response(term, context_clean, detection, minimum_score)
    response.update({
        "retrieval_query": retrieval.get("retrieval_query"),
        "retrieval_status": retrieval.get("retrieval_status", "success"),
        "grounding_status": "grounded",
        "generation_method": "context_evidence_template",
        "generated_explanation": explanation,
        "canonical_definition": definition,
        "retrieved_evidence": evidence,
        "sources": build_sources(evidence),
        "top_score": primary.get("final_score"),
        "warnings": retrieval.get("warnings", []),
        "reason_code": "GROUNDED",
        "reason": validation.get("reason") or retrieval.get("grounding_reason"),
    })
    return response
