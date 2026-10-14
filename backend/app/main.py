from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.core.config import get_settings
from backend.app.core.errors import (
    WordLensError,
    unexpected_error_handler,
    validation_error_handler,
    wordlens_error_handler,
)
from backend.app.core.logging_config import configure_logging
from backend.app.db.database import (
    create_tables,
    initialize_database,
)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    configure_logging(settings.log_level)

    logger.info("Starting WordLens backend.")

    settings.absolute_upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not settings.absolute_dataset_path.exists():
        raise RuntimeError(
            "Day-1 dataset was not found."
        )

    vectorstore_dir = settings.absolute_vectorstore_dir

    required_vectorstore_files = [
        "index.faiss",
        "metadata.json",
        "records.json",
        "embeddings.npy",
    ]

    missing = [
        filename
        for filename in required_vectorstore_files
        if not (vectorstore_dir / filename).exists()
    ]

    if missing:
        raise RuntimeError(
            "Persisted Day-3 vectorstore is incomplete: "
            + ", ".join(missing)
        )

    initialize_database(settings.database_url)

    # Create only the Day-4 serving tables.
    # This does NOT modify the Day-1 dataset.
    create_tables()

    logger.info(
        "WordLens backend startup validation completed."
    )

    yield

    logger.info(
        "WordLens backend shutting down."
    )


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="WordLens API",
        description=(
            "Technical terminology detection, "
            "retrieval and grounded explanation API."
        ),
        version="0.4.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=[
            "GET",
            "POST",
        ],
        allow_headers=[
            "Content-Type",
            "Authorization",
            "X-Request-ID",
        ],
        expose_headers=[
            "X-Request-ID",
        ],
    )

    @app.middleware("http")
    async def request_id_middleware(
        request: Request,
        call_next,
    ):
        incoming = request.headers.get(
            "X-Request-ID"
        )

        request_id = (
            incoming.strip()
            if incoming
            else str(uuid.uuid4())
        )

        request.state.request_id = request_id

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id

        return response

    app.add_exception_handler(
        WordLensError,
        wordlens_error_handler,
    )

    app.add_exception_handler(
        RequestValidationError,
        validation_error_handler,
    )

    app.add_exception_handler(
        Exception,
        unexpected_error_handler,
    )

    # Day-4 API endpoints are intentionally exposed
    # at the root paths required by the project SRS.
    app.include_router(router)

    return app


app = create_app()