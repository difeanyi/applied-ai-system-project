"""RAG: a small music knowledge base stored in a local ChromaDB collection.

The knowledge base is the Markdown files in data/knowledge/. Each "## " section
becomes one passage (for example "Mood: chill"), so retrieved passages are short,
self-contained and easy to cite. Passage IDs come from a hash of their content,
so the index rebuilds itself automatically whenever a knowledge file changes.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import Settings
from src.llm import friendly_api_error

logger = logging.getLogger("tunescout.rag")


class KnowledgeBaseError(Exception):
    """The knowledge base could not be built or searched. Safe to show to users."""


@dataclass
class Passage:
    source: str  # knowledge file name, e.g. "moods.md"
    section: str  # section heading, e.g. "Mood: chill"
    text: str
    distance: float = 0.0
    match: str = "semantic"  # "semantic" (embedding search) or "keyword: <synonym>"

    @property
    def citation(self) -> str:
        return f"{self.source} > {self.section}"


class LocalEmbeddings(Embeddings):
    """ChromaDB's built-in all-MiniLM-L6-v2 model (EMBEDDING_PROVIDER=local, no API key)."""

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

    return OpenAIEmbeddings(model=settings.openai_embedding_model,
                            api_key=settings.openai_api_key, max_retries=3)


def load_passages(knowledge_dir: Path) -> list[Document]:
    """Split every Markdown file into one passage per '## ' section."""
    files = sorted(knowledge_dir.glob("*.md"))
    if not files:
        raise KnowledgeBaseError(f"No knowledge files found in {knowledge_dir}.")

    documents: list[Document] = []
    for path in files:
        section, lines = None, []

        def flush() -> None:
            body = "\n".join(lines).strip()
            if section and body:
                digest = hashlib.sha256(f"{path.name}|{section}|{body}".encode()).hexdigest()[:16]
                documents.append(Document(
                    page_content=f"{section}\n{body}",
                    id=f"{path.stem}-{digest}",
                    metadata={"source": path.name, "section": section},
                ))

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                flush()
                section, lines = line[3:].strip(), []
            elif section is not None:
                lines.append(line)
        flush()
    return documents


_ALIAS_PREFIXES = ("Also called:", "Similar words:")
MAX_KEYWORD_MATCHES = 3


def extract_aliases(document: Document) -> list[str]:
    """Synonyms listed on a passage's 'Also called:' / 'Similar words:' lines."""
    aliases = []
    for line in document.page_content.splitlines():
        for prefix in _ALIAS_PREFIXES:
            if line.startswith(prefix):
                for alias in line[len(prefix):].split(","):
                    alias = re.sub(r"\(.*?\)", "", alias).strip(" .").lower()
                    if alias:
                        aliases.append(alias)
    return aliases


class KnowledgeBase:
    def __init__(self, settings: Settings, embeddings: Embeddings | None = None) -> None:
        self.settings = settings
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings or get_embeddings(settings),
            persist_directory=str(settings.chroma_persist_dir),
        )

    def ensure_index(self) -> int:
        """Index the knowledge files if they changed since the last run. Returns passage count."""
        documents = load_passages(self.settings.knowledge_dir)
        self._aliases = [(doc, extract_aliases(doc)) for doc in documents]
        wanted = {doc.id for doc in documents}
        existing = set(self.store.get(include=[])["ids"])
        if existing == wanted:
            logger.info("Knowledge base up to date: %d passages", len(wanted))
            return len(wanted)
        try:
            if existing:
                self.store.reset_collection()
            self.store.add_documents(documents, ids=[doc.id for doc in documents])
        except Exception as exc:
            logger.exception("Embedding API error while indexing the knowledge base")
            message, _ = friendly_api_error(exc, "OpenAI (embeddings)")
            raise KnowledgeBaseError(f"Could not index the knowledge base. {message}") from exc
        logger.info("Knowledge base indexed: %d passages from %s", len(documents),
                    sorted({d.metadata["source"] for d in documents}))
        return len(documents)

    def keyword_matches(self, query: str) -> list[Passage]:
        """Passages whose listed synonyms appear word for word in the query.

        Small embedding models can split unusual words (e.g. "chillhop" looks like
        "chill" + "hop"), so exact synonym matches are always included.
        """
        if not hasattr(self, "_aliases"):
            self._aliases = [(doc, extract_aliases(doc)) for doc in load_passages(self.settings.knowledge_dir)]
        text = query.lower()
        matches = []
        for doc, aliases in self._aliases:
            hit = next((a for a in aliases if re.search(rf"(?<![\w-]){re.escape(a)}(?![\w-])", text)), None)
            if hit:
                matches.append(Passage(doc.metadata["source"], doc.metadata["section"], doc.page_content,
                                       0.0, f"keyword: {hit}"))
        return matches

    def retrieve(self, query: str, k: int | None = None) -> list[Passage]:
        """Hybrid retrieval: exact synonym matches plus the k closest passages by
        embedding similarity. Logs the exact text of every passage."""
        k = k or self.settings.knowledge_top_k
        try:
            hits = self.store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            logger.exception("Embedding API error while searching the knowledge base")
            message, _ = friendly_api_error(exc, "OpenAI (embeddings)")
            raise KnowledgeBaseError(f"Could not search the knowledge base. {message}") from exc

        # Keyword matches are added on top of the k semantic results (at most
        # MAX_KEYWORD_MATCHES), so common words like "sad" can't crowd them out.
        passages = self.keyword_matches(query)[:MAX_KEYWORD_MATCHES]
        seen = {p.section for p in passages}
        for doc, score in hits:
            if doc.metadata["section"] not in seen:
                passages.append(Passage(doc.metadata["source"], doc.metadata["section"], doc.page_content, float(score)))
                seen.add(doc.metadata["section"])

        logger.info("Retrieval for %r (k=%d): %d passages", query, k, len(passages))
        for rank, p in enumerate(passages, start=1):
            logger.info("  Retrieved %d/%d | %s | %s | distance=%.4f\n----- BEGIN PASSAGE -----\n%s\n----- END PASSAGE -----",
                        rank, len(passages), p.citation, p.match, p.distance, p.text)
        return passages
