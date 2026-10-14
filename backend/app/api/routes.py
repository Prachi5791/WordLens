

from __future__ import annotations

import logging
import uuid
from html import unescape
from datetime import datetime, timezone
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Request,
    UploadFile,
)
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.errors import WordLensError
from backend.app.db.database import (
    check_database_connection,
    get_db,
)
from backend.app.repositories.terminology import (
    TerminologyRepository,
)
from backend.app.services.document import process_pdf
from backend.app.services.detector import detect_text
from backend.app.services.rag import explain
from backend.app.services.terminology import (
    lookup_term,
    serialize_term,
)


logger = logging.getLogger(__name__)

router = APIRouter()


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


class ExplainRequest(BaseModel):
    term: str = Field(
        min_length=1,
        max_length=512,
    )

    context: str | None = Field(
        default=None,
        max_length=10000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )

    include_related_terms: bool = False

    include_retrieval_evidence: bool = True


class ExplainSelectionRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=20000,
    )

    context: str | None = Field(
        default=None,
        max_length=10000,
    )

    top_k_per_term: int = Field(
        default=3,
        ge=1,
        le=10,
    )

    max_terms: int | None = Field(
        default=None,
        ge=1,
        le=50,
    )

    include_retrieval_evidence: bool = False


def request_id(request: Request) -> str:
    return getattr(
        request.state,
        "request_id",
        str(uuid.uuid4()),
    )


def build_evidence(
    result: dict,
) -> list[dict]:

    output = []

    for item in result.get(
        "retrieved_evidence",
        [],
    ):

        output.append(
            {
                "term_id": item.get(
                    "term_id"
                ),
                "term": item.get(
                    "term"
                ),
                "domain": item.get(
                    "domain"
                ),
                "subdomain": item.get(
                    "subdomain"
                ),
                "definition": item.get(
                    "definition"
                ),
                "source": item.get(
                    "source"
                ),
                "source_url": item.get(
                    "source_url"
                ),
                "rank": item.get(
                    "rank"
                ),
                "semantic_score": item.get(
                    "semantic_score"
                ),
                "lexical_boost": item.get(
                    "lexical_boost"
                ),
                "final_score": item.get(
                    "final_score"
                ),
            }
        )

    return output


@router.get("/health")
def health(request: Request):

    settings = get_settings()

    db_status = "unavailable"

    try:
        check_database_connection()
        db_status = "connected"
    except Exception:
        db_status = "unavailable"

    vectorstore_dir = (
        settings.absolute_vectorstore_dir
    )

    required_vectorstore_files = [
        "index.faiss",
        "metadata.json",
        "records.json",
        "embeddings.npy",
    ]

    vectorstore_available = all(
        (
            vectorstore_dir / filename
        ).exists()
        for filename in required_vectorstore_files
    )

    vectorstore_status = (
        "available"
        if vectorstore_available
        else "unavailable"
    )

    detector_status = "lazy"
    generation_status = "evidence_template"

    required_ready = (
        db_status == "connected"
        and vectorstore_available
        and settings.absolute_dataset_path.exists()
    )

    payload = {
        "status": (
            "ready"
            if required_ready
            else "unavailable"
        ),
        "environment": settings.app_env,
        "database": db_status,
        "vectorstore": vectorstore_status,
        "count": None,
        "model": "BAAI/bge-small-en-v1.5",
        "dataset_version": None,
        "detector": detector_status,
        "generation": generation_status,
        "timestamp": utc_now(),
    }

    if not required_ready:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content=payload,
            headers={
                "X-Request-ID": request_id(request),
            },
        )

    return payload


@router.post("/upload")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
):

    settings = get_settings()

    content = await file.read()

    if not file.filename:
        raise WordLensError(
            error_code="INVALID_FILENAME",
            message="Uploaded file has no filename.",
            status_code=400,
        )

    result = process_pdf(
        content=content,
        original_filename=file.filename,
        content_type=file.content_type,
        upload_dir=settings.absolute_upload_dir,
        max_upload_bytes=settings.max_upload_bytes,
    )

    result["created_at"] = utc_now()

    return result


@router.get("/term/{term}")
def get_term(
    term: str,
    request: Request,
    db: Session = Depends(get_db),
):
    repository = TerminologyRepository(db)

    record, matched_by, decision = lookup_term(
        repository,
        term,
        context=None,
    )

    if record is None and decision.get("decision") != "resolved":
        raise WordLensError(
            error_code=decision.get("reason_code", "TERM_NOT_FOUND"),
            message=decision.get("reason", "Technical term could not be verified."),
            status_code=404,
            details={"requested_term": term, "decision": decision},
        )

    return serialize_term(
        record=record,
        matched_by=matched_by,
        requested_term=term,
        decision=decision,
    )


@router.post("/explain")
def explain_term(
    payload: ExplainRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    context = payload.context.strip() if payload.context else None
    if context == "":
        context = None

    repository = TerminologyRepository(db)
    record, matched_by, decision = lookup_term(
        repository,
        payload.term,
        context=context,
    )

    knowledge = None
    if record is not None and record.definition:
        knowledge = {
            "term_id": record.term_id,
            "term": record.term,
            "aliases": record.aliases or [],
            "domain": record.domain,
            "subdomain": record.subdomain,
            "definition": record.definition,
            "source": record.source,
            "source_url": record.source_url,
            "confidence": record.confidence,
            "knowledge_status": decision.get("knowledge_status", "local"),
        }
    elif decision.get("definition"):
        knowledge = dict(decision)

    rag_result = explain(
        selected_term=payload.term.strip(),
        context=context,
        top_k=payload.top_k,
        knowledge=knowledge,
    )

    return {
        "requested_term": payload.term,
        "resolved_term": record.term if record is not None else rag_result.get("selected_term"),
        "canonical_definition": unescape(str(record.definition)).strip() if record is not None and record.definition else rag_result.get("canonical_definition"),
        "simplified_definition": record.simplified_definition if record is not None else decision.get("simplified_definition"),
        "context": context,
        "confidence": record.confidence if record is not None else decision.get("confidence"),
        "grounding_status": rag_result.get("grounding_status"),
        "generation_method": rag_result.get("generation_method"),
        "generated_explanation": rag_result.get("generated_explanation"),
        "retrieved_evidence": build_evidence(rag_result) if payload.include_retrieval_evidence else [],
        "sources": rag_result.get("sources", []),
        "warnings": rag_result.get("warnings", []),
        "reason_code": rag_result.get("reason_code"),
        "reason": rag_result.get("reason"),
        "knowledge_status": decision.get("knowledge_status"),
        "matched_by": matched_by,
        "request_id": request_id(request),
        "created_at": utc_now(),
    }


@router.post("/explain-selection")
def explain_selection(
    payload: ExplainSelectionRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    detections = detect_text(payload.text)
    if payload.max_terms is not None:
        detections = detections[:payload.max_terms]

    repository = TerminologyRepository(db)
    explanations = []
    seen_terms: set[str] = set()

    for detection in detections:
        term = str(detection.get("term") or "").strip()
        if not term:
            continue
        key = term.casefold()
        if key in seen_terms:
            continue
        seen_terms.add(key)

        context = (payload.context or detection.get("sentence") or "").strip() or None
        record, matched_by, decision = lookup_term(repository, term, context=context)
        knowledge = None
        if record is not None and record.definition:
            knowledge = {
                "term_id": record.term_id,
                "term": record.term,
                "aliases": record.aliases or [],
                "domain": record.domain,
                "subdomain": record.subdomain,
                "definition": record.definition,
                "source": record.source,
                "source_url": record.source_url,
                "confidence": record.confidence,
                "knowledge_status": decision.get("knowledge_status", "local"),
            }

        result = explain(
            selected_term=term,
            context=context,
            top_k=payload.top_k_per_term,
            detection=detection,
            knowledge=knowledge,
        )

        explanations.append({
            "term_id": record.term_id if record is not None else detection.get("term_id"),
            "term": record.term if record is not None else term,
            "canonical_definition": result.get("canonical_definition"),
            "simplified_definition": record.simplified_definition if record is not None else None,
            "grounding_status": result.get("grounding_status"),
            "generation_method": result.get("generation_method"),
            "generated_explanation": result.get("generated_explanation"),
            "top_score": result.get("top_score"),
            "evidence": build_evidence(result) if payload.include_retrieval_evidence else [],
            "sources": result.get("sources", []),
            "warnings": result.get("warnings", []),
            "reason_code": result.get("reason_code"),
            "reason": result.get("reason"),
            "matched_by": matched_by,
        })

    return {
        "request_id": request_id(request),
        "input_text": payload.text,
        "detection_count": len(detections),
        "explanation_count": len(explanations),
        "detections": detections,
        "explanations": explanations,
        "warnings": [],
        "created_at": utc_now(),
    }

