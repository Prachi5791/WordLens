

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from scripts.technical_term_detector import TechnicalTermDetector

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_detector() -> TechnicalTermDetector:
    logger.info("Initializing WordLens Day-2 detector.")
    return TechnicalTermDetector(
        use_semantic=False,
        use_external=False,
    )


def detect_text(text: str) -> list[dict]:
    if not text or not text.strip():
        return []
    return [item.to_dict() for item in get_detector().detect_text(text, use_semantic=False)]


def detect_pdf(pdf_path: str | Path) -> list[dict]:
    return [item.to_dict() for item in get_detector().detect_pdf(pdf_path)]
