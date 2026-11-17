from __future__ import annotations

import logging

import pytest

from study_generator.ingestion import (
    IngestionError,
    PageText,
    chunk_pages,
    get_document_outline,
    ingest_file,
    ingest_files,
    parse_pdf,
    parse_pptx,
    remove_document,
    save_upload,
)
from study_generator.logging_config import setup_logging

from .conftest import make_pdf, make_pptx


# ---- parsing ---------------------------------------------------------------
def test_parse_pdf_keeps_page_numbers(tmp_path):
    pdf = make_pdf(tmp_path / "bio.pdf", [["Photosynthesis", "Light reactions"], ["Cell Respiration", "ATP"]])
    pages = parse_pdf(pdf)
    assert [p.location for p in pages] == [1, 2]
    assert pages[0].location_type == "page"
    assert pages[0].title == "Photosynthesis"
    assert "ATP" in pages[1].text


def test_parse_pptx_reads_titles_body_and_notes(tmp_path):
    deck = make_pptx(tmp_path / "chem.pptx", [
        ("Atoms", "Protons and neutrons", "Remember the nucleus"),
        ("Bonds", "Covalent and ionic", None),
    ])
    slides = parse_pptx(deck)
    assert [s.location for s in slides] == [1, 2]
    assert slides[0].title == "Atoms"
    assert "Protons and neutrons" in slides[0].text
    assert "Speaker notes: Remember the nucleus" in slides[0].text
    assert slides[1].location_type == "slide"


# ---- chunking ----------------------------------------------------------------
def test_chunks_respect_word_limit_and_keep_metadata():
    long_text = " ".join(f"word{i}" for i in range(1200))
    pages = [PageText("notes.pdf", 7, "page", "Big page", long_text)]
    chunks = chunk_pages(pages, file_hash="a" * 64, chunk_size_words=500, overlap_words=50)

    assert len(chunks) == 3
    assert all(len(c.page_content.split()) <= 500 for c in chunks)
    assert {c.metadata["page"] for c in chunks} == {7}
    assert {c.metadata["source"] for c in chunks} == {"notes.pdf"}
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1, 2]
    assert len({c.id for c in chunks}) == 3  # unique, deterministic ids


def test_empty_pages_produce_no_chunks():
    pages = [PageText("x.pdf", 1, "page", "", ""), PageText("x.pdf", 2, "page", "T", "some text")]
    chunks = chunk_pages(pages, file_hash="b" * 64)
    assert [c.metadata["page"] for c in chunks] == [2]


# ---- guardrails --------------------------------------------------------------
@pytest.mark.parametrize("name, content, message", [
    ("notes.txt", b"hello", "not supported"),
    ("empty.pdf", b"", "is empty"),
    ("broken.pdf", b"this is not a pdf", "could not be opened"),
    ("broken.pptx", b"this is not a pptx", "could not be opened"),
])
def test_bad_files_raise_clear_errors(tmp_path, settings, store, name, content, message):
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(IngestionError, match=message):
        ingest_file(path, settings, store)


def test_pdf_without_text_is_rejected(tmp_path, settings, store):
    pdf = make_pdf(tmp_path / "scan.pdf", [[]])
    with pytest.raises(IngestionError, match="No readable text"):
        ingest_file(pdf, settings, store)


def test_one_bad_file_does_not_stop_the_batch(tmp_path, settings, store):
    good = make_pdf(tmp_path / "good.pdf", [["Topic A", "details"]])
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"garbage")
    results, errors = ingest_files([good, bad], settings, store)
    assert [r.filename for r in results] == ["good.pdf"]
    assert list(errors) == ["bad.pdf"]


# ---- vector store ------------------------------------------------------------
def test_ingest_stores_chunks_and_builds_outline(tmp_path, settings, store):
    pdf = make_pdf(tmp_path / "bio.pdf", [["Photosynthesis", "Light"], [], ["Respiration", "ATP"]])
    deck = make_pptx(tmp_path / "chem.pptx", [("Atoms", "Protons", None)])

    pdf_result = ingest_file(pdf, settings, store)
    assert pdf_result.units_total == 3
    assert pdf_result.skipped_units == [2]
    assert pdf_result.chunks_added == 2
    ingest_file(deck, settings, store)

    outline = get_document_outline(store)
    assert list(outline) == ["bio.pdf", "chem.pptx"]
    assert [(e["location"], e["title"]) for e in outline["bio.pdf"]] == [(1, "Photosynthesis"), (3, "Respiration")]
    assert outline["chem.pptx"][0] == {"location": 1, "location_type": "slide", "title": "Atoms"}

    hit = store.similarity_search("Respiration ATP", k=1, filter={"source": "bio.pdf"})[0]
    assert hit.metadata["source"] == "bio.pdf"


def test_reingest_is_idempotent_and_updates_replace_old_chunks(tmp_path, settings, store):
    pdf = make_pdf(tmp_path / "bio.pdf", [["Version one"]])
    ingest_file(pdf, settings, store)
    again = ingest_file(pdf, settings, store)
    assert again.already_ingested and again.chunks_added == 0
    assert len(store.get()["ids"]) == 1

    make_pdf(pdf, [["Version two"], ["Extra page"]])
    ingest_file(pdf, settings, store)
    texts = store.get()["documents"]
    assert len(texts) == 2 and not any("Version one" in t for t in texts)

    assert remove_document(store, "bio.pdf") == 2
    assert store.get()["ids"] == []


def test_save_upload_sanitizes_filename(settings):
    saved = save_upload("../../etc/evil?.pdf", b"%PDF", settings)
    assert saved.parent == settings.upload_dir
    assert saved.name == "evil_.pdf"


def test_ingestion_is_logged(tmp_path, settings, store):
    setup_logging(settings.log_file)
    try:
        ingest_file(make_pdf(tmp_path / "bio.pdf", [["Photosynthesis"]]), settings, store)
        log = settings.log_file.read_text()
        assert "Ingestion started: bio.pdf" in log
        assert "Ingestion finished: bio.pdf -> 1 chunks stored" in log
    finally:
        logger = logging.getLogger("study_generator")
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
