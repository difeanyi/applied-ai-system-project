"""End-to-end test of app.py with Streamlit's headless AppTest runner (no API key needed)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from streamlit.testing.v1 import AppTest

from src import llm as llm_module, rag

from .conftest import FakeStructuredLLM, good_explanation, interpretation

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "system.log"))
    monkeypatch.setattr(rag, "get_embeddings", lambda s: DeterministicFakeEmbedding(size=32))
    yield tmp_path / "system.log"
    logger = logging.getLogger("tunescout")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def use_fake_llm(monkeypatch, replies):
    fake = FakeStructuredLLM(replies)
    monkeypatch.setattr(llm_module, "get_chat_model", lambda settings, temperature=None: fake)
    return fake


def test_missing_key_shows_setup_error(app_env, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "OPENAI_API_KEY is not set" in at.error[0].value


def test_request_to_recommendations_and_human_override(app_env, monkeypatch):
    use_fake_llm(monkeypatch, [
        interpretation(genre="lofi", mood="focused", energy=0.4, acoustic=True, mode="energy-focused",
                       text="Calm, focused lofi for studying."),
        good_explanation,
    ])
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception

    at.text_area(key="request").set_value("something mellow for studying").run()
    next(b for b in at.button if b.label == "Find music").click().run()
    assert not at.exception, at.exception
    assert at.info[0].value == "Calm, focused lofi for studying."
    assert at.session_state.result.recommendations[0].song["title"] == "Focus Flow"
    assert any("#1 Focus Flow" in m.value for m in at.markdown)

    # Human in the loop: switch to jazz-first and re-rank without the AI.
    at.selectbox[0].set_value("jazz")
    at.selectbox[2].set_value("genre-first")
    next(b for b in at.button if b.label == "Re-rank").click().run()
    assert not at.exception, at.exception
    manual_profile, mode, recs = at.session_state.manual
    assert manual_profile["favorite_genre"] == "jazz"
    assert recs[0].song["title"] == "Coffee Shop Stories"
    assert "Human override" in app_env.read_text()


def test_off_topic_request_shows_warning(app_env, monkeypatch):
    use_fake_llm(monkeypatch, [interpretation(music=False)])
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_area(key="request").set_value("do my taxes").run()
    next(b for b in at.button if b.label == "Find music").click().run()
    assert "only help with music" in at.warning[0].value
