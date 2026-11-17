"""RAG Phase 1: parse PDF/PPTX files, chunk the text, and store it in ChromaDB.

Flow for one file (``ingest_file``):
    validate -> hash -> skip if unchanged -> parse pages/slides -> chunk
    (~500 words, metadata preserved) -> embed -> store in a local Chroma collection

Every chunk keeps the metadata needed for citations later:
    source (filename), page (page or slide number), location_type ("page" or
    "slide"), title (slide title or first line of the page), chunk_index, file_hash.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .llm import friendly_api_error

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".pptx": "pptx"}
_EMBED_BATCH_SIZE = 100
_TITLE_MAX_CHARS = 120


class IngestionError(Exception):
    """A file could not be ingested. The message is safe to show to the user."""


@dataclass
class PageText:
    """Text extracted from one PDF page or one PowerPoint slide."""

    source: str
    location: int  # 1-based page or slide number
    location_type: str  # "page" or "slide"
    title: str
    text: str


@dataclass
class IngestionResult:
    filename: str
    file_type: str
    units_total: int = 0  # pages or slides in the file
    units_with_text: int = 0
    chunks_added: int = 0
    already_ingested: bool = False
    skipped_units: list[int] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Embeddings and vector store
# --------------------------------------------------------------------------- #
class LocalEmbeddings(Embeddings):
    """ChromaDB's built-in all-MiniLM-L6-v2 model, wrapped for LangChain.

    Runs fully on this machine with no API key (EMBEDDING_PROVIDER=local).
    """

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embeddings(settings: Settings) -> Embeddings:
    settings.require_embeddings()
    if settings.embedding_provider == "local":
        return LocalEmbeddings()
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
        max_retries=3,
    )


def get_vector_store(settings: Settings, embeddings: Embeddings | None = None) -> Chroma:
    """Open (or create) the persistent Chroma collection on local disk."""
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings or get_embeddings(settings),
        persist_directory=str(settings.chroma_persist_dir),
    )


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:_TITLE_MAX_CHARS]
    return ""


def parse_pdf(path: Path) -> list[PageText]:
    from PyPDF2 import PdfReader

    name = path.name
    try:
        reader = PdfReader(str(path), strict=False)
    except Exception as exc:  # PyPDF2 raises many different error types for bad files
        raise IngestionError(f"'{name}' could not be opened as a PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")  # many "protected" PDFs use an empty password
        except Exception as exc:
            raise IngestionError(f"'{name}' is encrypted and could not be opened.") from exc
        if not unlocked:
            raise IngestionError(f"'{name}' is password-protected. Remove the password and re-upload.")

    try:
        page_list = list(reader.pages)
    except Exception as exc:
        raise IngestionError(f"'{name}' appears to be corrupted: {exc}") from exc

    pages: list[PageText] = []
    for number, page in enumerate(page_list, start=1):
        try:
            text = _clean(page.extract_text() or "")
        except Exception as exc:
            logger.warning("PDF %s: could not extract text from page %d: %s", name, number, exc)
            text = ""
        pages.append(PageText(name, number, "page", _first_line(text), text))
    return pages


def _shape_text(shape) -> Iterable[str]:
    """Yield the text in a shape, including grouped shapes and table cells."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for child in shape.shapes:
            yield from _shape_text(child)
        return
    if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
        yield shape.text_frame.text
    if getattr(shape, "has_table", False) and shape.has_table:
        for row in shape.table.rows:
            yield " | ".join(cell.text for cell in row.cells)


def parse_pptx(path: Path) -> list[PageText]:
    from pptx import Presentation

    name = path.name
    try:
        presentation = Presentation(str(path))
    except Exception as exc:
        raise IngestionError(f"'{name}' could not be opened as a PowerPoint file: {exc}") from exc

    slides: list[PageText] = []
    for number, slide in enumerate(presentation.slides, start=1):
        title_shape = slide.shapes.title
        title = _clean(title_shape.text) if title_shape is not None and title_shape.has_text_frame else ""

        parts = [t for shape in slide.shapes for t in _shape_text(shape) if t.strip()]
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame
            if notes is not None and notes.text.strip():
                parts.append(f"Speaker notes: {notes.text}")

        text = _clean("\n\n".join(parts))
        slides.append(PageText(name, number, "slide", title[:_TITLE_MAX_CHARS] or _first_line(text), text))
    return slides


def parse_file(path: Path) -> list[PageText]:
    file_type = SUPPORTED_EXTENSIONS.get(path.suffix.lower())
    if file_type == "pdf":
        return parse_pdf(path)
    if file_type == "pptx":
        return parse_pptx(path)
    raise IngestionError(
        f"'{path.name}' is not supported. Upload a PDF or PPTX file "
        "(older .ppt files must be re-saved as .pptx)."
    )


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def _word_count(text: str) -> int:
    return len(text.split())


def chunk_pages(
    pages: list[PageText], file_hash: str, chunk_size_words: int = 500, overlap_words: int = 50
) -> list[Document]:
    """Split each page/slide into chunks of at most ``chunk_size_words`` words.

    Chunks never cross a page boundary, so every chunk maps to exactly one
    page or slide number for citations.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_words,
        chunk_overlap=overlap_words,
        length_function=_word_count,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    documents: list[Document] = []
    for page in pages:
        if not page.text:
            continue
        for index, piece in enumerate(splitter.split_text(page.text)):
            documents.append(
                Document(
                    page_content=piece,
                    id=f"{file_hash[:16]}-{page.location_type}{page.location}-{index}",
                    metadata={
                        "source": page.source,
                        "page": page.location,
                        "location_type": page.location_type,
                        "title": page.title,
                        "chunk_index": index,
                        "word_count": _word_count(piece),
                        "file_hash": file_hash,
                    },
                )
            )
    return documents


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_file(path: Path, settings: Settings) -> str:
    if not path.is_file():
        raise IngestionError(f"File not found: {path}")
    file_type = SUPPORTED_EXTENSIONS.get(path.suffix.lower())
    if file_type is None:
        raise IngestionError(f"'{path.name}' is not supported. Upload a PDF or PPTX file.")
    size = path.stat().st_size
    if size == 0:
        raise IngestionError(f"'{path.name}' is empty.")
    if size > settings.max_upload_mb * 1024 * 1024:
        raise IngestionError(
            f"'{path.name}' is {size / 1_048_576:.1f} MB; the limit is {settings.max_upload_mb} MB."
        )
    return file_type


def _ids_where(store: Chroma, where: dict) -> list[str]:
    return store.get(where=where, include=[])["ids"]


def ingest_file(path: Path | str, settings: Settings, store: Chroma | None = None) -> IngestionResult:
    """Parse, chunk, embed and store one file. Raises IngestionError on failure.

    Re-uploading an unchanged file is a no-op; uploading a changed file with the
    same name replaces its old chunks.
    """
    path = Path(path)
    started = time.perf_counter()
    logger.info("Ingestion started: %s", path.name)

    try:
        file_type = _validate_file(path, settings)
        store = store or get_vector_store(settings)
        file_hash = _file_hash(path)
        result = IngestionResult(filename=path.name, file_type=file_type)

        if _ids_where(store, {"file_hash": file_hash}):
            result.already_ingested = True
            logger.info("Ingestion skipped: %s is unchanged and already stored", path.name)
            return result

        stale_ids = _ids_where(store, {"source": path.name})
        if stale_ids:
            store.delete(ids=stale_ids)
            logger.info("Removed %d chunks from an older version of %s", len(stale_ids), path.name)

        pages = parse_file(path)
        result.units_total = len(pages)
        result.skipped_units = [p.location for p in pages if not p.text]
        result.units_with_text = result.units_total - len(result.skipped_units)
        logger.info(
            "Parsed %s: %d %ss, %d with text", path.name, result.units_total,
            "page" if file_type == "pdf" else "slide", result.units_with_text,
        )
        if result.skipped_units:
            logger.warning("%s: no text found on %s %s", path.name,
                           "pages" if file_type == "pdf" else "slides", result.skipped_units)
        if result.units_with_text == 0:
            hint = " It may be a scanned image; OCR is not supported." if file_type == "pdf" else ""
            raise IngestionError(f"No readable text was found in '{path.name}'.{hint}")

        chunks = chunk_pages(pages, file_hash, settings.chunk_size_words, settings.chunk_overlap_words)
        logger.info("Chunked %s into %d chunks (max %d words each)",
                    path.name, len(chunks), settings.chunk_size_words)

        try:
            for start in range(0, len(chunks), _EMBED_BATCH_SIZE):
                batch = chunks[start:start + _EMBED_BATCH_SIZE]
                store.add_documents(batch, ids=[doc.id for doc in batch])
        except Exception as exc:
            logger.exception("Embedding/storage API error while ingesting %s", path.name)
            message, _ = friendly_api_error(exc, service="The embeddings API")
            raise IngestionError(f"Could not embed '{path.name}'. {message}") from exc

        result.chunks_added = len(chunks)
        logger.info("Ingestion finished: %s -> %d chunks stored in %.1fs",
                    path.name, result.chunks_added, time.perf_counter() - started)
        return result

    except IngestionError as exc:
        logger.error("Ingestion failed: %s: %s", path.name, exc)
        raise


def ingest_files(paths: Iterable[Path | str], settings: Settings, store: Chroma | None = None
                 ) -> tuple[list[IngestionResult], dict[str, str]]:
    """Ingest several files. One bad file does not stop the others.

    Returns (successful results, {filename: error message}).
    """
    store = store or get_vector_store(settings)
    results: list[IngestionResult] = []
    errors: dict[str, str] = {}
    for path in paths:
        try:
            results.append(ingest_file(path, settings, store))
        except IngestionError as exc:
            errors[Path(path).name] = str(exc)
    return results, errors


def save_upload(filename: str, data: bytes, settings: Settings) -> Path:
    """Save uploaded bytes under data/uploads with a safe filename."""
    safe_name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(filename).name).strip() or "upload"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.upload_dir / safe_name
    destination.write_bytes(data)
    return destination


# --------------------------------------------------------------------------- #
# Reading back what is stored (used by the scheduler in Step 3)
# --------------------------------------------------------------------------- #
def get_document_outline(store: Chroma) -> dict[str, list[dict]]:
    """Return a table of contents built from stored chunk metadata.

    ``{filename: [{"location": 3, "location_type": "page", "title": "..."}, ...]}``
    sorted by page/slide number, one entry per page or slide.
    """
    metadatas = store.get(include=["metadatas"])["metadatas"]
    outline: dict[str, dict[int, dict]] = {}
    for meta in metadatas:
        entries = outline.setdefault(meta["source"], {})
        entries.setdefault(meta["page"], {
            "location": meta["page"],
            "location_type": meta["location_type"],
            "title": meta.get("title", ""),
        })
    return {
        source: [entries[loc] for loc in sorted(entries)]
        for source, entries in sorted(outline.items())
    }


def remove_document(store: Chroma, filename: str) -> int:
    """Delete every chunk of one file. Returns how many chunks were removed."""
    ids = _ids_where(store, {"source": filename})
    if ids:
        store.delete(ids=ids)
    logger.info("Removed %s from the vector store (%d chunks)", filename, len(ids))
    return len(ids)


def clear_store(store: Chroma) -> None:
    """Delete every stored chunk (start over with a fresh collection)."""
    store.reset_collection()
    logger.info("Vector store cleared")
