from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(
        default="development",
        alias="WORDLENS_APP_ENV",
    )

    host: str = Field(
        default="127.0.0.1",
        alias="WORDLENS_HOST",
    )

    port: int = Field(
        default=8000,
        alias="WORDLENS_PORT",
    )

    database_url: str = Field(
        default="mysql+pymysql://wordlens:CHANGE_ME@127.0.0.1:3306/wordlens",
        alias="WORDLENS_DATABASE_URL",
    )

    upload_dir: str = Field(
        default="data/uploads",
        alias="WORDLENS_UPLOAD_DIR",
    )

    max_upload_bytes: int = Field(
        default=10 * 1024 * 1024,
        alias="WORDLENS_MAX_UPLOAD_BYTES",
    )

    vectorstore_dir: str = Field(
        default="vectorstore",
        alias="WORDLENS_VECTORSTORE_DIR",
    )

    dataset_path: str = Field(
        default="data/processed/technical_terms.json",
        alias="WORDLENS_DATASET_PATH",
    )

    log_level: str = Field(
        default="INFO",
        alias="WORDLENS_LOG_LEVEL",
    )

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        alias="WORDLENS_CORS_ORIGINS",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            return [
                item.strip()
                for item in value.split(",")
                if item.strip()
            ]

        return value

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def absolute_upload_dir(self) -> Path:
        path = Path(self.upload_dir)

        if not path.is_absolute():
            path = self.project_root / path

        return path.resolve()

    @property
    def absolute_vectorstore_dir(self) -> Path:
        path = Path(self.vectorstore_dir)

        if not path.is_absolute():
            path = self.project_root / path

        return path.resolve()

    @property
    def absolute_dataset_path(self) -> Path:
        path = Path(self.dataset_path)

        if not path.is_absolute():
            path = self.project_root / path

        return path.resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
