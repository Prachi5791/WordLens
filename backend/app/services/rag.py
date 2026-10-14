

from __future__ import annotations

from scripts.rag import run_rag


def explain(
    selected_term: str,
    context: str | None,
    top_k: int,
    detection: dict | None = None,
    knowledge: dict | None = None,
) -> dict:
    return run_rag(
        selected_term=selected_term,
        context=context,
        top_k=top_k,
        minimum_score=0.45,
        detection=detection,
        knowledge=knowledge,
    )
