from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, Text
from backend.app.db.database import Base


class TechnicalTerm(Base):
    __tablename__ = "technical_terms"

    id = Column(Integer, primary_key=True, autoincrement=True)

    term_id = Column(String(100), unique=True, nullable=False, index=True)
    term = Column(String(255), nullable=False)
    normalized_term = Column(String(255), nullable=False, index=True)

    aliases = Column(JSON, nullable=True)

    domain = Column(String(100), nullable=True, index=True)
    subdomain = Column(String(100), nullable=True, index=True)

    definition = Column(Text, nullable=False)
    simplified_definition = Column(Text, nullable=True)

    source = Column(String(255), nullable=True)
    source_url = Column(Text, nullable=True)

    confidence = Column(Float, nullable=True)

    quality_status = Column(
        String(50),
        nullable=False,
        index=True,
    )

    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )

    dataset_hash = Column(String(64), nullable=False)

    imported_at = Column(DateTime(timezone=True), nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )