from __future__ import annotations

from scripts.retrieval import retrieve


def retrieve_terms(
    selected_term: str | None,
    context: str | None,
    top_k: int,
    minimum_score: float = 0.45,
    detection: dict | None = None,
) -> dict:

    return retrieve(
        selected_term=selected_term,
        context=context,
        top_k=top_k,
        minimum_score=minimum_score,
        detection=detection,
    )
