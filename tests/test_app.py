"""End-to-end test of app.py using Streamlit's headless AppTest runner.

The LLM and embeddings are replaced with offline fakes so no API key is needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from streamlit.testing.v1 import AppTest

from study_generator import ingestion, llm as llm_module
from study_generator.config import load_settings
from study_generator.ingestion import get_vector_store, ingest_file

from .conftest import FakeStructuredLLM, make_pdf

APP = str(Path(__file__).resolve().parents[1] / "app.py")

PLAN = {"days": [
    {"day": 1, "focus": "Energy", "is_review_day": False, "topics": [
        {"title": "Photosynthesis", "summary": "How plants capture light", "search_query": "chlorophyll",
         "sources": [{"filename": "bio.pdf", "pages": [1]}]}]},
    {"day": 2, "focus": "Review", "is_review_day": True, "topics": [
        {"title": "Respiration", "summary": "How cells make ATP", "search_query": "mitochondria",
         "sources": [{"filename": "bio.pdf", "pages": [2]}]}]},
]}


def q(n, page):
    return {"question": f"Q{n} about page {page}?", "options": ["w", "x", "y", "z"], "correct_option": 2,
            "explanation": "Because the notes say so.", "source_filename": "bio.pdf", "source_location": page}


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "system.log"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(ingestion, "get_embeddings", lambda s: DeterministicFakeEmbedding(size=32))
    yield load_settings()
    logger = logging.getLogger("study_generator")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_empty_app_renders(app_env):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value.endswith("Study Syllabus & Quiz Generator")
    assert any("Upload and process at least one file" in i.value for i in at.info)


def test_missing_api_key_shows_config_error(app_env, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "OPENAI_API_KEY is not set" in at.error[0].value


def test_full_pipeline_schedule_then_quizzes(app_env, tmp_path, monkeypatch):
    pdf = make_pdf(tmp_path / "bio.pdf", [["Photosynthesis", "Chlorophyll absorbs light."],
                                          ["Respiration", "Mitochondria make ATP."]])
    ingest_file(pdf, app_env, get_vector_store(app_env))

    fake = FakeStructuredLLM([
        PLAN,
        {"insufficient_context": False, "questions": [q(1, 1), q(2, 1), q(3, 1)]},
        {"insufficient_context": True, "questions": []},
    ])
    monkeypatch.setattr(llm_module, "get_chat_model", lambda settings, temperature=None: fake)

    at = AppTest.from_file(APP, default_timeout=30).run()
    assert "bio.pdf" in at.sidebar.markdown[1].value

    at.radio[0].set_value("Number of days").run()
    at.number_input[0].set_value(2).run()
    next(b for b in at.button if b.label == "Generate schedule").click().run()
    assert not at.exception, at.exception
    assert at.session_state.schedule["total_days"] == 2
    assert any("Day 2" in e.label and "Review" in e.label for e in at.expander)

    next(b for b in at.button if b.label == "Generate quizzes").click().run()
    assert not at.exception, at.exception
    quizzes = at.session_state.quizzes
    assert [quiz["status"] for quiz in quizzes] == ["ok", "not_found"]
    assert any("Information not found in documents." in w.value for w in at.warning)
    assert any("Source: bio.pdf, page 1" in c.value for c in at.caption)

    at.radio(key="answer-0-0").set_value("C. y").run()
    assert any(s.value == "Correct!" for s in at.success)

    log = Path(app_env.log_file).read_text()
    assert "App: schedule requested for 2 days" in log
    assert "Retrieved chunk 1/" in log and "Chlorophyll absorbs light." in log
