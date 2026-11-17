"""Application settings loaded from environment variables / the .env file.

Every tunable value lives here so the rest of the code never reads os.environ
directly. Relative paths are resolved against the project root, so the app
behaves the same no matter which directory it is launched from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALID_LLM_PROVIDERS = ("openai", "anthropic")
VALID_EMBEDDING_PROVIDERS = ("openai", "local")

# Values copied unchanged from .env.example are treated as "not set".
_PLACEHOLDER_KEYS = {"sk-your-openai-key-here", "sk-ant-your-anthropic-key-here"}


class ConfigError(ValueError):
    """Raised when a required setting is missing or invalid."""


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _api_key(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value if value and value not in _PLACEHOLDER_KEYS else None


def _int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a whole number, got {raw!r}.") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, got {value}.")
    return value


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}.") from exc


def _choice(name: str, default: str, valid: tuple[str, ...]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in valid:
        raise ConfigError(f"{name} must be one of {', '.join(valid)}; got {value!r}.")
    return value


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-5-5"
    embedding_provider: str = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    chroma_persist_dir: Path = PROJECT_ROOT / "chroma_db"
    chroma_collection: str = "study_materials"
    log_file: Path = PROJECT_ROOT / "system.log"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    chunk_size_words: int = 500
    chunk_overlap_words: int = 50
    retrieval_top_k: int = 5
    questions_per_topic: int = 3
    llm_temperature: float = 0.2
    max_upload_mb: int = 50

    @property
    def collection_name(self) -> str:
        # Vectors from different embedding models are not comparable, so each
        # embedding provider gets its own collection.
        return f"{self.chroma_collection}_{self.embedding_provider}"

    def require_embeddings(self) -> None:
        """Fail early with a clear message if embeddings cannot be created."""
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ConfigError(
                "EMBEDDING_PROVIDER is 'openai' but OPENAI_API_KEY is not set. "
                "Add your key to .env, or set EMBEDDING_PROVIDER=local."
            )

    def require_llm(self) -> None:
        """Fail early with a clear message if the chat model cannot be called."""
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ConfigError("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set in .env.")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ConfigError("LLM_PROVIDER is 'anthropic' but ANTHROPIC_API_KEY is not set in .env.")


def load_settings(env_file: Path | None = None) -> Settings:
    """Read settings from the environment (after loading the .env file)."""
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

    chunk_size = _int("CHUNK_SIZE_WORDS", 500, minimum=50)
    overlap = _int("CHUNK_OVERLAP_WORDS", 50, minimum=0)
    if overlap >= chunk_size:
        raise ConfigError("CHUNK_OVERLAP_WORDS must be smaller than CHUNK_SIZE_WORDS.")

    return Settings(
        llm_provider=_choice("LLM_PROVIDER", "openai", VALID_LLM_PROVIDERS),
        openai_api_key=_api_key("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        anthropic_api_key=_api_key("ANTHROPIC_API_KEY"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
        embedding_provider=_choice("EMBEDDING_PROVIDER", "openai", VALID_EMBEDDING_PROVIDERS),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chroma_persist_dir=_resolve_path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")),
        chroma_collection=os.getenv("CHROMA_COLLECTION", "study_materials"),
        log_file=_resolve_path(os.getenv("LOG_FILE", "./system.log")),
        upload_dir=_resolve_path(os.getenv("UPLOAD_DIR", "./data/uploads")),
        chunk_size_words=chunk_size,
        chunk_overlap_words=overlap,
        retrieval_top_k=_int("RETRIEVAL_TOP_K", 5),
        questions_per_topic=_int("QUESTIONS_PER_TOPIC", 3),
        llm_temperature=_float("LLM_TEMPERATURE", 0.2),
        max_upload_mb=_int("MAX_UPLOAD_MB", 50),
    )
