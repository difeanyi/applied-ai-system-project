"""Shared test fixtures: sample PDF/PPTX files and an offline vector store."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from study_generator.config import Settings
from study_generator.ingestion import get_vector_store


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(path: Path, pages: list[list[str]]) -> Path:
    """Write a minimal, valid text PDF. ``pages`` is a list of lines per page."""
    objects: list[bytes] = []
    page_ids = [3 + 2 * i for i in range(len(pages))]
    font_id = 3 + 2 * len(pages)

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    for pid, lines in zip(page_ids, pages):
        ops = ["BT", "/F1 11 Tf", "14 TL", "50 750 Td"]
        for line in lines:
            ops.append(f"({_escape(line)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {pid + 1} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


def make_pptx(path: Path, slides: list[tuple[str, str, str | None]]) -> Path:
    """Write a PPTX. Each slide is (title, body, speaker_notes_or_None)."""
    from pptx import Presentation

    presentation = Presentation()
    layout = presentation.slide_layouts[1]  # "Title and Content"
    for title, body, notes in slides:
        slide = presentation.slides.add_slide(layout)
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
        if notes:
            slide.notes_slide.notes_text_frame.text = notes
    presentation.save(str(path))
    return path


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        embedding_provider="local",  # never needs an API key
        chroma_persist_dir=tmp_path / "chroma",
        log_file=tmp_path / "test.log",
        upload_dir=tmp_path / "uploads",
    )


@pytest.fixture
def store(settings: Settings):
    """A real on-disk Chroma collection using a fake (offline) embedder."""
    return get_vector_store(settings, embeddings=DeterministicFakeEmbedding(size=32))
