from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.terminology import TechnicalTerm


ACTIVE_STATUS = "accepted"


class TerminologyRepository:

    def __init__(self, db: Session):
        self.db = db

    def get_exact(
        self,
        normalized_term: str,
    ) -> TechnicalTerm | None:

        statement = (
            select(TechnicalTerm)
            .where(
                TechnicalTerm.normalized_term
                == normalized_term,
                TechnicalTerm.is_active.is_(True),
                TechnicalTerm.quality_status
                == ACTIVE_STATUS,
            )
            .limit(1)
        )

        return self.db.execute(
            statement
        ).scalar_one_or_none()

    def get_by_alias(
        self,
        normalized_requested_term: str,
    ) -> TechnicalTerm | None:

        statement = (
            select(TechnicalTerm)
            .where(
                TechnicalTerm.is_active.is_(True),
                TechnicalTerm.quality_status
                == ACTIVE_STATUS,
            )
        )

        records = self.db.execute(
            statement
        ).scalars().all()

        for record in records:
            aliases = record.aliases or []

            for alias in aliases:
                if normalize(alias) == normalized_requested_term:
                    return record

        return None

    def count_active(self) -> int:
        statement = (
            select(TechnicalTerm)
            .where(
                TechnicalTerm.is_active.is_(True),
                TechnicalTerm.quality_status
                == ACTIVE_STATUS,
            )
        )

        return len(
            self.db.execute(statement).scalars().all()
        )


def normalize(value: str) -> str:
    value = value.strip().lower()

    value = (
        value
        .replace("-", " ")
        .replace("_", " ")
    )

    return " ".join(value.split())
