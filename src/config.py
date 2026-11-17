"""Settings for TuneScout AI, loaded from environment variables / the .env file.

Relative paths are resolved against the project root so the app behaves the
same no matter which directory it is launched from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SONGS_CSV = PROJECT_ROOT / "data" / "songs.csv"
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"

# Values copied unchanged from .env.example are treated as "not set".
_PLACEHOLDER_KEYS = {"sk-your-openai-key-here", "sk-ant-your-anthropic-key-here", "your-gemini-key-here"}


class ConfigError(ValueError):
    """A required setting is missing or invalid. The message is safe to show users."""


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _api_key(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value if value and value not in _PLACEHOLDER_KEYS else None


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a whole number, got {raw!r}.") from exc
    if value < 1:
        raise ConfigError(f"{name} must be at least 1, got {value}.")
    return value


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
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    embedding_provider: str = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    chroma_persist_dir: Path = PROJECT_ROOT / "chroma_db"
    log_file: Path = PROJECT_ROOT / "system.log"
    songs_csv: Path = SONGS_CSV
    knowledge_dir: Path = KNOWLEDGE_DIR
    knowledge_top_k: int = 4
    recommendations_k: int = 5
    llm_temperature: float = 0.2
    max_request_chars: int = 500

    @property
    def collection_name(self) -> str:
        # Vectors from different embedding models are not comparable.
        return f"music_knowledge_{self.embedding_provider}"

    def require_embeddings(self) -> None:
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ConfigError(
                "EMBEDDING_PROVIDER is 'openai' but OPENAI_API_KEY is not set. "
                "Add your key to .env, or set EMBEDDING_PROVIDER=local."
            )

    def require_llm(self) -> None:
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ConfigError("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set in .env.")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ConfigError("LLM_PROVIDER is 'anthropic' but ANTHROPIC_API_KEY is not set in .env.")
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            raise ConfigError("LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set in .env.")

    @property
    def llm_model(self) -> str:
        return {"openai": self.openai_model, "anthropic": self.anthropic_model,
                "gemini": self.gemini_model}[self.llm_provider]


def load_settings(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    try:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    except ValueError as exc:
        raise ConfigError("LLM_TEMPERATURE must be a number.") from exc

    return Settings(
        llm_provider=_choice("LLM_PROVIDER", "openai", ("openai", "anthropic", "gemini")),
        openai_api_key=_api_key("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        anthropic_api_key=_api_key("ANTHROPIC_API_KEY"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
        gemini_api_key=_api_key("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        embedding_provider=_choice("EMBEDDING_PROVIDER", "openai", ("openai", "local")),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chroma_persist_dir=_resolve_path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")),
        log_file=_resolve_path(os.getenv("LOG_FILE", "./system.log")),
        knowledge_top_k=_int("KNOWLEDGE_TOP_K", 4),
        recommendations_k=_int("RECOMMENDATIONS_K", 5),
        llm_temperature=temperature,
        max_request_chars=_int("MAX_REQUEST_CHARS", 500),
    )
