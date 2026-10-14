
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, text

from backend.app.db.database import SessionLocal
from backend.app.models.terminology import TechnicalTerm
from scripts.scientific_term_discovery import (
    classify_rejection,
    generate_candidates,
    normalize_lookup,
)

NIST_URL = "https://csrc.nist.gov/glossary/term/{term}"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "WordLens/1.0 (scientific terminology resolver)"
MAX_DEFINITION_LENGTH = 12000
HTTP_TIMEOUT = 8
DEFAULT_MAX_DEPTH = 2
DEFAULT_MAX_TERMS_PER_DEFINITION = 20


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()


def _valid_definition(value: Any) -> bool:
    text_value = _clean_text(value)
    return 30 <= len(text_value) <= MAX_DEFINITION_LENGTH


def dynamic_term_id(term: str, source: str) -> str:
    normalized = normalize_lookup(term)
    source = _clean_text(source)
    digest = hashlib.sha256(
        f"dynamic:{normalized}:{source}".encode("utf-8")
    ).hexdigest()
    return f"dyn_{digest[:24]}"


def _request_json(url: str, params: dict[str, Any], timeout: int = HTTP_TIMEOUT) -> dict[str, Any] | None:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def _request_text(url: str, timeout: int = HTTP_TIMEOUT) -> str | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def _clean_html(value: str) -> str:
    return _clean_text(re.sub(r"<[^>]+>", " ", value or ""))


def _lookup_variants(term: str) -> list[str]:
    """Conservative lookup normalization only; never a detection gate."""
    term = _clean_text(term)
    if not term:
        return []

    variants = [term]
    normalized = normalize_lookup(term)
    if " " not in normalized and "-" not in normalized:
        if normalized.endswith("ies") and len(normalized) > 4:
            variants.append(normalized[:-3] + "y")
        elif normalized.endswith(("ches", "shes", "xes", "zes", "ses")) and len(normalized) > 4:
            variants.append(normalized[:-2])
        elif normalized.endswith("s") and not normalized.endswith("ss") and len(normalized) > 3:
            variants.append(normalized[:-1])

    output: list[str] = []
    seen: set[str] = set()
    for value in variants:
        key = normalize_lookup(value)
        if key and key not in seen:
            seen.add(key)
            output.append(value)
    return output


def _context_tokens(context: str | None) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9_-]{1,}", _clean_text(context).lower()))


def _context_window(term: str, context: str | None) -> str:
    context = _clean_text(context)
    if not context:
        return ""
    tokens = context.split()
    term_tokens = normalize_lookup(term).split()
    if not term_tokens:
        return context[:800]

    lowered = [normalize_lookup(token) for token in tokens]
    for index in range(len(lowered)):
        if lowered[index:index + len(term_tokens)] == term_tokens:
            start = max(0, index - 12)
            end = min(len(tokens), index + len(term_tokens) + 12)
            return " ".join(tokens[start:end])
    return context[:800]


def _is_disambiguation(title: str, definition: str) -> bool:
    combined = f"{title} {definition}".lower()
    markers = (
        "may refer to:",
        "can refer to:",
        "disambiguation page",
        "is a disambiguation",
    )
    return any(marker in combined for marker in markers)


def _score_wikipedia_result(term: str, title: str, snippet: str, context: str | None) -> float:
    requested = normalize_lookup(term)
    candidate = normalize_lookup(title)
    requested_tokens = set(requested.split())
    candidate_tokens = set(candidate.split())
    context_tokens = _context_tokens(context)
    score = 0.0

    if candidate == requested:
        score += 1.00
    elif requested and requested in candidate:
        score += 0.78
    elif candidate and candidate in requested:
        score += 0.62

    if requested_tokens and candidate_tokens:
        score += 0.25 * (len(requested_tokens & candidate_tokens) / len(requested_tokens))

    if context_tokens:
        score += min(len(candidate_tokens & context_tokens) * 0.16, 0.48)
        snippet_tokens = set(re.findall(r"[a-z][a-z0-9_-]{1,}", snippet.lower()))
        score += min(len(snippet_tokens & context_tokens) * 0.025, 0.20)

    if len(requested_tokens) == 1 and len(candidate_tokens) > 1:
        score += 0.08
    return score


def _wikipedia_search_queries(term: str, context: str | None) -> list[str]:
    queries = [term]
    window = _context_window(term, context)
    if window:
        context_tokens = list(_context_tokens(window))[:10]
        if context_tokens:
            queries.append(f"{term} {' '.join(context_tokens)}")
    spaced = term.replace("-", " ")
    if spaced.casefold() != term.casefold():
        queries.append(spaced)
    return list(dict.fromkeys(q for q in queries if q.strip()))


def lookup_nist(term: str) -> dict[str, Any] | None:
    term = _clean_text(term)
    if not term:
        return None
    encoded = urllib.parse.quote(term, safe="")
    url = NIST_URL.format(term=encoded)
    page = _request_text(url)
    if not page:
        return None

    title_match = re.search(r"<title>(.*?)</title>", page, flags=re.I | re.S)
    if not title_match:
        return None
    title = _clean_html(title_match.group(1))
    requested = normalize_lookup(term)
    normalized_title = normalize_lookup(title)
    if not requested or requested not in normalized_title:
        return None

    definition_match = re.search(
        r"(?:Definitions?|Definition).{0,2000}?<p[^>]*>(.*?)</p>",
        page,
        flags=re.I | re.S,
    )
    if not definition_match:
        return None
    definition = _clean_html(definition_match.group(1))
    if not _valid_definition(definition):
        return None

    return {
        "term": term,
        "canonical_term": term,
        "requested_term": term,
        "aliases": [],
        "definition": definition[:MAX_DEFINITION_LENGTH],
        "source": "NIST CSRC Glossary",
        "source_url": url,
        "confidence": 0.92,
    }


def lookup_wikipedia(term: str, context: str | None = None) -> dict[str, Any] | None:
    term = _clean_text(term)
    normalized_term = normalize_lookup(term)
    if not normalized_term:
        return None

    candidates: dict[str, dict[str, Any]] = {}
    for query in _wikipedia_search_queries(term, context):
        search = _request_json(
            WIKIPEDIA_API,
            {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srnamespace": 0,
                "srlimit": 10,
                "srprop": "snippet|titlesnippet|redirecttitle",
            },
        )
        if not search:
            continue

        for result in search.get("query", {}).get("search", []):
            title = _clean_text(result.get("title"))
            if not title:
                continue
            key = normalize_lookup(title)
            score = _score_wikipedia_result(
                term,
                title,
                _clean_text(result.get("snippet")),
                context,
            )
            if key not in candidates or score > candidates[key]["score"]:
                candidates[key] = {"result": result, "title": title, "score": score}

    if not candidates:
        return None

    ranked = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
    for candidate in ranked[:5]:
        title = candidate["title"]
        page = _request_json(
            WIKIPEDIA_API,
            {
                "action": "query",
                "format": "json",
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "redirects": 1,
                "inprop": "url",
                "titles": title,
            },
        )
        if not page:
            continue
        pages = page.get("query", {}).get("pages", {})
        if not pages:
            continue
        page_data = next(iter(pages.values()), {})
        definition = _clean_text(page_data.get("extract"))
        actual_title = _clean_text(page_data.get("title")) or title
        if not _valid_definition(definition) or _is_disambiguation(actual_title, definition):
            continue

        source_url = page_data.get("fullurl") or (
            "https://en.wikipedia.org/wiki/"
            + urllib.parse.quote(actual_title.replace(" ", "_"))
        )
        canonical = actual_title
        aliases = []
        if normalize_lookup(canonical) != normalized_term:
            aliases.append(term)

        return {
            "term": canonical,
            "canonical_term": canonical,
            "requested_term": term,
            "aliases": aliases,
            "definition": definition[:MAX_DEFINITION_LENGTH],
            "source": "Wikipedia",
            "source_url": source_url,
            "confidence": round(min(0.92, 0.62 + max(candidate["score"], 0.0) * 0.16), 3),
        }
    return None


def _abbreviation_expansions(term: str, context: str | None) -> list[str]:
    if not context:
        return []
    term_clean = _clean_text(term)
    if not re.fullmatch(r"[A-Z][A-Z0-9&./-]{1,15}", term_clean):
        return []

    patterns = [
        re.compile(rf"(?i)([A-Za-z][A-Za-z0-9 /_-]{{3,100}}?)\s*\(\s*{re.escape(term_clean)}\s*\)"),
        re.compile(rf"(?i){re.escape(term_clean)}\s*\(\s*([A-Za-z][A-Za-z0-9 /_-]{{3,100}}?)\s*\)"),
    ]
    values: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(context):
            values.append(_clean_text(match.group(1)))
    return list(dict.fromkeys(v for v in values if v))


def external_lookup(term: str, context: str | None = None) -> dict[str, Any] | None:
    term = _clean_text(term)
    if not term:
        return None

    # Explicit paper-local abbreviation evidence gets first priority.
    for expansion in _abbreviation_expansions(term, context):
        result = lookup_nist(expansion) or lookup_wikipedia(expansion, context)
        if result:
            aliases = list(result.get("aliases", []) or [])
            if normalize_lookup(term) != normalize_lookup(result.get("term")) and term not in aliases:
                aliases.append(term)
            result["aliases"] = list(dict.fromkeys(aliases))
            result["requested_term"] = term
            return result

    for variant in _lookup_variants(term):
        result = lookup_nist(variant)
        if result:
            if normalize_lookup(variant) != normalize_lookup(term):
                result["aliases"] = [term]
            result["requested_term"] = term
            return result

    for variant in _lookup_variants(term):
        result = lookup_wikipedia(variant, context)
        if result:
            aliases = list(result.get("aliases", []) or [])
            if normalize_lookup(variant) != normalize_lookup(term) and term not in aliases:
                aliases.append(term)
            result["aliases"] = list(dict.fromkeys(aliases))
            result["requested_term"] = term
            return result
    return None


def _definition_tokens(definition: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9-]{2,}", definition.lower()))


def validate_contextual_usage(term: str, context: str | None, knowledge: dict[str, Any]) -> dict[str, Any]:
    """Validate that terminology evidence is compatible with the PDF context."""
    definition = _clean_text(knowledge.get("definition"))
    title = _clean_text(knowledge.get("canonical_term") or knowledge.get("term") or term)
    source = _clean_text(knowledge.get("source"))

    if not _valid_definition(definition):
        return {"valid": False, "reason_code": "INVALID_DEFINITION", "reason": "The source did not provide a usable definition."}

    context = _clean_text(context)
    if not context:
        if source == "Wikipedia" and len(normalize_lookup(term).split()) == 1 and normalize_lookup(title) != normalize_lookup(term):
            return {"valid": False, "reason_code": "AMBIGUOUS_EXTERNAL_MATCH", "reason": "The one-word request matched an expanded Wikipedia title without paper context."}
        return {"valid": True, "reason_code": "CONTEXT_NOT_REQUIRED", "reason": "The terminology evidence is sufficiently specific."}

    context_norm = normalize_lookup(context)
    term_norm = normalize_lookup(term)
    title_norm = normalize_lookup(title)
    context_tokens = _context_tokens(context)
    title_tokens = set(title_norm.split())
    definition_tokens = _definition_tokens(definition)
    title_overlap = len(title_tokens & context_tokens)
    definition_overlap = len(definition_tokens & context_tokens)

    # The selected term itself must actually occur in the supplied paper context.
    term_tokens = set(term_norm.split())
    if term_tokens and not term_tokens.issubset(context_tokens):
        return {"valid": False, "reason_code": "TERM_NOT_IN_CONTEXT", "reason": "The selected term does not occur in the supplied PDF context."}

    # One-word Wikipedia pages are ambiguous unless the paper context supports them.
    if source == "Wikipedia" and len(term_norm.split()) == 1 and title_norm != term_norm:
        if title_tokens.issubset(context_tokens) and definition_overlap >= 1:
            return {"valid": True, "reason_code": "CONTEXT_VALIDATED", "reason": "The expanded Wikipedia title is explicitly supported by the paper context."}
        return {"valid": False, "reason_code": "AMBIGUOUS_EXTERNAL_MATCH", "reason": "The external result expanded a one-word term without enough context support."}

    # A heading such as SYSTEM ARCHITECTURE must not be enough to accept an unrelated glossary page.
    if len(context.split()) <= 6 and context_norm != term_norm and definition_overlap < 2:
        return {"valid": False, "reason_code": "CONTEXT_TOO_WEAK", "reason": "The supplied PDF context is too short to validate the external meaning."}

    if title_norm == term_norm and definition_overlap >= 1:
        return {"valid": True, "reason_code": "CONTEXT_VALIDATED", "reason": "The canonical evidence is compatible with the paper context."}

    if title_overlap >= 1 and definition_overlap >= 1:
        return {"valid": True, "reason_code": "CONTEXT_VALIDATED", "reason": "The evidence title and definition are compatible with the paper context."}

    if source != "Wikipedia" and title_norm == term_norm and len(context.split()) >= 8:
        return {"valid": True, "reason_code": "CONTEXT_VALIDATED", "reason": "The exact approved terminology entry is used in a sufficiently descriptive paper context."}

    return {"valid": False, "reason_code": "CONTEXT_MISMATCH", "reason": "The retrieved definition is not sufficiently supported by the PDF context."}


def local_lookup(session, term: str) -> tuple[TechnicalTerm | None, str | None]:
    normalized = normalize_lookup(term)
    if not normalized:
        return None, None

    canonical = (
        session.query(TechnicalTerm)
        .filter(
            TechnicalTerm.is_active.is_(True),
            TechnicalTerm.quality_status == "accepted",
            func.lower(TechnicalTerm.normalized_term) == normalized,
        )
        .first()
    )
    if canonical is not None:
        return canonical, "canonical"

    records = (
        session.query(TechnicalTerm)
        .filter(
            TechnicalTerm.is_active.is_(True),
            TechnicalTerm.quality_status == "accepted",
        )
        .all()
    )
    for record in records:
        for alias in record.aliases or []:
            if normalize_lookup(str(alias)) == normalized:
                return record, "alias"
    return None, None


def ensure_dataset_hash_schema(session) -> None:
    try:
        session.execute(text("ALTER TABLE technical_terms MODIFY COLUMN dataset_hash VARCHAR(64) NOT NULL"))
        session.commit()
    except Exception:
        session.rollback()


def _confidence_for_source(source: str | None) -> float:
    if source == "NIST CSRC Glossary":
        return 0.92
    if source == "Wikipedia":
        return 0.78
    return 0.70


def persist_external(session, knowledge: dict[str, Any]) -> tuple[TechnicalTerm, bool]:
    canonical = _clean_text(knowledge.get("canonical_term") or knowledge.get("term"))
    requested = _clean_text(knowledge.get("requested_term"))
    definition = _clean_text(knowledge.get("definition"))
    source = _clean_text(knowledge.get("source"))
    source_url = _clean_text(knowledge.get("source_url"))

    if not canonical or not _valid_definition(definition) or not source:
        raise ValueError("External knowledge is incomplete or invalid.")

    normalized = normalize_lookup(canonical)
    existing, _ = local_lookup(session, canonical)
    now = utc_now()

    aliases = list(existing.aliases or []) if existing is not None else []
    for alias in knowledge.get("aliases", []) or []:
        alias_text = _clean_text(alias)
        if alias_text and normalize_lookup(alias_text) != normalized and alias_text not in aliases:
            aliases.append(alias_text)
    if requested and normalize_lookup(requested) != normalized and requested not in aliases:
        aliases.append(requested)

    if existing is not None:
        # Never replace a complete verified definition with a different source.
        if not _valid_definition(existing.definition):
            existing.definition = definition
            existing.source = source
            existing.source_url = source_url or existing.source_url
            existing.confidence = float(knowledge.get("confidence") or _confidence_for_source(source))
            existing.quality_status = "accepted"
            existing.is_active = True
        existing.aliases = aliases
        existing.updated_at = now
        if not existing.imported_at:
            existing.imported_at = now
        session.commit()
        session.refresh(existing)
        return existing, False

    term_id = dynamic_term_id(canonical, source)
    existing_by_id = session.query(TechnicalTerm).filter(TechnicalTerm.term_id == term_id).first()
    if existing_by_id is not None:
        return existing_by_id, False

    record = TechnicalTerm(
        term_id=term_id,
        term=canonical,
        normalized_term=normalized,
        aliases=aliases,
        domain=None,
        subdomain=None,
        definition=definition,
        simplified_definition=None,
        source=source,
        source_url=source_url or None,
        confidence=float(knowledge.get("confidence") or _confidence_for_source(source)),
        quality_status="accepted",
        is_active=True,
        dataset_hash=hashlib.sha256(f"dynamic:{normalized}:{source}".encode("utf-8")).hexdigest(),
        imported_at=now,
        created_at=now,
        updated_at=now,
    )
    session.add(record)
    try:
        session.commit()
    except Exception:
        session.rollback()
        existing, _ = local_lookup(session, canonical)
        if existing is not None:
            return existing, False
        existing_by_id = session.query(TechnicalTerm).filter(TechnicalTerm.term_id == term_id).first()
        if existing_by_id is not None:
            return existing_by_id, False
        raise
    session.refresh(record)
    return record, True


def _record_to_result(record: TechnicalTerm, matched_by: str, status: str = "local") -> dict[str, Any]:
    return {
        "term": record.term,
        "canonical_term": record.term,
        "term_id": record.term_id,
        "aliases": record.aliases or [],
        "definition": _clean_text(record.definition),
        "source": record.source,
        "source_url": record.source_url,
        "confidence": record.confidence,
        "technical_status": "technical",
        "decision": "resolved",
        "reason_code": "LOCAL_RESOLVED",
        "reason": "The term was resolved from the local verified knowledge base.",
        "matched_by": matched_by,
        "knowledge_status": status,
        "persisted": False,
        "created": False,
    }


def resolve_term(term: str, context: str | None = None, persist: bool = True) -> dict[str, Any]:
    term = _clean_text(term)
    if not term:
        return {"term": "", "technical_status": "unknown", "decision": "unresolved", "reason_code": "EMPTY", "reason": "Term is empty.", "knowledge_status": "unknown", "persisted": False, "created": False}

    rejection = classify_rejection(term, context)
    hard_rejections = {"EMPTY", "PUNCTUATION", "URL", "QUANTITY", "STOPWORD"}
    if rejection.get("decision") == "rejected" and rejection.get("reason_code") in hard_rejections:
        return {"term": term, **rejection, "knowledge_status": "not_requested"}

    session = SessionLocal()
    try:
        local, matched_by = local_lookup(session, term)
        if local is not None and _valid_definition(local.definition):
            validation = validate_contextual_usage(
                term,
                context,
                {
                    "term": local.term,
                    "canonical_term": local.term,
                    "definition": local.definition,
                    "source": local.source,
                },
            )
            if validation.get("valid"):
                result = _record_to_result(local, matched_by)
                result["reason"] = validation.get("reason", result["reason"])
                result["reason_code"] = validation.get("reason_code", result["reason_code"])
                return result

        evidence = external_lookup(term, context)
        if evidence is not None:
            validation = validate_contextual_usage(term, context, evidence)
            if validation.get("valid"):
                if persist:
                    record, created = persist_external(session, evidence)
                    return {
                        **evidence,
                        "term": record.term,
                        "canonical_term": record.term,
                        "term_id": record.term_id,
                        "aliases": record.aliases or [],
                        "definition": _clean_text(record.definition),
                        "source": record.source,
                        "source_url": record.source_url,
                        "confidence": record.confidence,
                        "technical_status": "technical",
                        "decision": "resolved",
                        "reason_code": "EXTERNAL_RESOLVED",
                        "reason": validation.get("reason"),
                        "knowledge_status": "external",
                        "matched_by": "external" if created else "canonical",
                        "persisted": True,
                        "created": created,
                    }
                return {
                    **evidence,
                    "technical_status": "technical",
                    "decision": "resolved",
                    "reason_code": "EXTERNAL_RESOLVED",
                    "reason": validation.get("reason"),
                    "knowledge_status": "external",
                    "persisted": False,
                    "created": False,
                }

        if local is not None:
            return {
                **_record_to_result(local, matched_by),
                "decision": "unresolved_context",
                "reason_code": "CONTEXT_MISMATCH",
                "reason": "A local definition exists, but it could not be validated against the supplied PDF context.",
                "knowledge_status": "local_context_mismatch",
            }

        return {
            "term": term,
            "definition": None,
            "source": None,
            "source_url": None,
            "confidence": 0.0,
            "technical_status": "unknown",
            "decision": "unresolved",
            "reason_code": "EXTERNAL_NOT_FOUND",
            "reason": "No verified local or approved external definition was found for this context.",
            "knowledge_status": "unknown",
            "persisted": False,
            "created": False,
        }
    finally:
        session.close()


def enrich_definition(root_term: str, definition: str, depth: int = 0, max_depth: int = DEFAULT_MAX_DEPTH, max_terms: int = DEFAULT_MAX_TERMS_PER_DEFINITION) -> dict[str, Any]:
    if depth >= max_depth:
        return {"root_term": root_term, "depth": depth, "discovered": [], "stopped": True}
    discovered = []
    for candidate in generate_candidates(definition, top_n=max_terms):
        result = resolve_term(candidate.text, context=definition, persist=True)
        if result.get("decision") == "resolved":
            discovered.append({"term": candidate.text, "score": candidate.score, "source": candidate.source, "result": result})
    return {"root_term": root_term, "depth": depth, "discovered": discovered, "stopped": False}


def enrich_database(max_depth: int = DEFAULT_MAX_DEPTH, max_root_terms: int | None = None, max_terms_per_definition: int = DEFAULT_MAX_TERMS_PER_DEFINITION, rebuild_index: bool = True) -> dict[str, Any]:
    visited: set[str] = set()
    queue: list[tuple[str, str, int]] = []
    session = SessionLocal()
    try:
        query = session.query(TechnicalTerm).filter(TechnicalTerm.is_active.is_(True), TechnicalTerm.quality_status == "accepted").order_by(TechnicalTerm.id.asc())
        if max_root_terms is not None:
            query = query.limit(max_root_terms)
        for record in query.all():
            if record.definition:
                queue.append((record.term, record.definition, 0))
    finally:
        session.close()

    discovered_total = 0
    processed = 0
    while queue:
        term, definition, depth = queue.pop(0)
        normalized = normalize_lookup(term)
        if not normalized or normalized in visited:
            continue
        visited.add(normalized)
        processed += 1
        if depth >= max_depth:
            continue
        result = enrich_definition(term, definition, depth, max_depth, max_terms_per_definition)
        for item in result["discovered"]:
            discovered_total += 1
            child = item["result"]
            if child.get("term") and child.get("definition") and depth + 1 < max_depth:
                queue.append((child["term"], child["definition"], depth + 1))

    index_status = "not_requested"
    if rebuild_index:
        try:
            from scripts.retrieval import build_vectorstore
            build_vectorstore()
            index_status = "rebuilt"
        except Exception as exc:
            index_status = f"rebuild_failed: {exc}"
    return {"processed_definitions": processed, "resolved_concepts": discovered_total, "rejected_or_unresolved_candidates": 0, "max_depth": max_depth, "index_status": index_status}


def main() -> None:
    parser = argparse.ArgumentParser(description="WordLens dynamic scientific knowledge resolver")
    subparsers = parser.add_subparsers(dest="command", required=True)
    resolve_parser = subparsers.add_parser("resolve")
    resolve_parser.add_argument("term")
    resolve_parser.add_argument("--context", default=None)
    resolve_parser.add_argument("--no-persist", action="store_true")
    enrich_parser = subparsers.add_parser("enrich-db")
    enrich_parser.add_argument("--max-depth", type=int, default=2)
    enrich_parser.add_argument("--max-root-terms", type=int, default=None)
    enrich_parser.add_argument("--no-rebuild-index", action="store_true")
    enrich_parser.add_argument("--max-terms-per-definition", type=int, default=20)
    args = parser.parse_args()
    if args.command == "resolve":
        result = resolve_term(args.term, context=args.context, persist=not args.no_persist)
    else:
        result = enrich_database(max_depth=args.max_depth, max_root_terms=args.max_root_terms, max_terms_per_definition=args.max_terms_per_definition, rebuild_index=not args.no_rebuild_index)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
