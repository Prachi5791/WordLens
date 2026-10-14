from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

import fitz

from backend.app.core.errors import WordLensError
from backend.app.services.detector import detect_pdf

logger = logging.getLogger(__name__)


PDF_MAGIC = b"%PDF"


def validate_pdf_bytes(
    content: bytes,
    filename: str,
    content_type: str | None,
    max_bytes: int,
) -> None:

    if not filename.lower().endswith(".pdf"):
        raise WordLensError(
            error_code="INVALID_FILE_TYPE",
            message="Only PDF files are supported.",
            status_code=400,
        )

    if content_type not in (
        None,
        "application/pdf",
    ):
        raise WordLensError(
            error_code="INVALID_CONTENT_TYPE",
            message="Uploaded file must have content type application/pdf.",
            status_code=400,
        )

    if len(content) > max_bytes:
        raise WordLensError(
            error_code="FILE_TOO_LARGE",
            message="Uploaded PDF exceeds the configured size limit.",
            status_code=413,
        )

    if not content.startswith(PDF_MAGIC):
        raise WordLensError(
            error_code="INVALID_PDF",
            message="File does not contain a valid PDF signature.",
            status_code=400,
        )


def safe_filename(
    filename: str,
    document_id: str,
) -> str:

    original = Path(filename).name

    original = re.sub(
        r"[^A-Za-z0-9._-]",
        "_",
        original,
    )

    if not original.lower().endswith(".pdf"):
        original += ".pdf"

    return (
        f"{document_id}_{original}"
    )


def extract_pdf_statistics(
    path: Path,
) -> dict:

    try:
        document = fitz.open(
            str(path)
        )
    except Exception as exc:
        raise WordLensError(
            error_code="INVALID_PDF",
            message="The uploaded PDF could not be opened.",
            status_code=400,
        ) from exc

    if document.is_encrypted:
        document.close()

        raise WordLensError(
            error_code="ENCRYPTED_PDF",
            message="Encrypted PDFs are not supported.",
            status_code=400,
        )

    page_count = len(document)

    text_parts = []

    for page in document:
        text_parts.append(
            page.get_text("text")
        )

    document.close()

    text = "\n".join(
        text_parts
    ).strip()

    if not text:
        raise WordLensError(
            error_code="SCANNED_PDF",
            message=(
                "No extractable text was found. "
                "Scanned PDFs without a text layer are not supported."
            ),
            status_code=422,
        )

    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            text,
        )
        if sentence.strip()
    ]

    return {
        "page_count": page_count,
        "text": text,
        "extracted_character_count": len(text),
        "sentence_count": len(sentences),
    }


def process_pdf(
    content: bytes,
    original_filename: str,
    content_type: str | None,
    upload_dir: Path,
    max_upload_bytes: int,
) -> dict:

    validate_pdf_bytes(
        content=content,
        filename=original_filename,
        content_type=content_type,
        max_bytes=max_upload_bytes,
    )

    document_id = uuid.uuid4().hex

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = safe_filename(
        original_filename,
        document_id,
    )

    destination = (
        upload_dir / filename
    ).resolve()

    upload_root = upload_dir.resolve()

    if upload_root not in destination.parents:
        raise WordLensError(
            error_code="INVALID_FILENAME",
            message="Unsafe filename.",
            status_code=400,
        )

    destination.write_bytes(
        content
    )

    try:
        statistics = extract_pdf_statistics(
            destination
        )

        detections = detect_pdf(
            destination
        )

    except Exception:
        try:
            destination.unlink()
        except OSError:
            pass

        raise

    summaries = []

    for detection in detections:
        summaries.append(
            {
                "detection_id": detection.get(
                    "detection_id"
                ),
                "term_id": detection.get(
                    "term_id"
                ),
                "term": detection.get(
                    "term"
                ),
                "matched_text": detection.get(
                    "matched_text"
                ),
                "page_number": detection.get(
                    "page_number"
                ),
                "sentence": detection.get(
                    "sentence"
                ),
                "confidence": detection.get(
                    "confidence"
                ),
            }
        )

    return {
        "document_id": document_id,
        "original_filename": original_filename,
        "content_type": "application/pdf",
        "byte_size": len(content),
        "page_count": statistics[
            "page_count"
        ],
        "extracted_character_count": statistics[
            "extracted_character_count"
        ],
        "sentence_count": statistics[
            "sentence_count"
        ],
        "detected_term_count": len(
            detections
        ),
        "detected_terms": summaries,
        "processing_status": "completed",
        "warnings": [],
    }
