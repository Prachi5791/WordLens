from __future__ import annotations

from typing import Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.app.core.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()

_engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=3600,
)

SessionLocal = sessionmaker(
    bind=_engine,
    autocommit=False,
    autoflush=False,
)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """
    Create database tables if they do not already exist.

    This does not import or seed terminology data.
    """

    from backend.app.models.terminology import TechnicalTerm  # noqa: F401

    Base.metadata.create_all(bind=_engine)


def initialize_database(
    database_url: Optional[str] = None,
) -> None:
    """
    Initialize the database schema.

    `database_url` is accepted because the FastAPI lifespan passes the
    configured database URL during startup.

    The existing configured engine is reused when no URL is supplied.
    """

    if database_url is not None and database_url != settings.database_url:
        raise ValueError(
            "Database URL passed to initialize_database does not match "
            "the configured database URL."
        )

    create_tables()


def check_database_connection() -> bool:
    """
    Check whether the configured MySQL database is reachable.
    """

    try:
        with _engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return True

    except Exception:
        return False


def get_engine():
    """
    Return the configured SQLAlchemy engine.
    """

    return _engine