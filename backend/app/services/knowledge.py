from __future__ import annotations

from sqlalchemy.orm import Session

from scripts.dynamic_knowledge import resolve_or_enrich


def resolve_term(session: Session, term: str, allow_external: bool = True):
    return resolve_or_enrich(session, term, allow_external=allow_external)
