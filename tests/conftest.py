"""Shared fixtures: offline settings, a fake embedder and a scripted fake LLM."""

from __future__ import annotations

import json
import logging
import re

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.config import KNOWLEDGE_DIR, SONGS_CSV, Settings
from src.rag import KnowledgeBase


class FakeStructuredLLM:
    """Stands in for a chat model: returns scripted structured replies in order.

    Each reply is a dict (validated against the requested schema), a callable
    taking the prompt messages and returning a dict, or an Exception to raise.
    Every prompt is recorded in ``self.calls``.
    """

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.calls: list[list] = []

    def with_structured_output(self, schema, include_raw: bool = False):
        def run(prompt_value):
            messages = prompt_value.to_messages()
            self.calls.append(messages)
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            if callable(reply):
                reply = reply(messages)
            return {"raw": AIMessage(content=json.dumps(reply)),
                    "parsed": schema.model_validate(reply), "parsing_error": None}

        return RunnableLambda(run)


def interpretation(genre=None, mood=None, energy=None, valence=None, acoustic=None,
                   mode="balanced", music=True, text="test interpretation"):
    return {"is_music_request": music, "favorite_genre": genre, "favorite_mood": mood,
            "target_energy": energy, "target_valence": valence, "likes_acoustic": acoustic,
            "ranking_mode": mode, "interpretation": text, "knowledge_used": ["Mood: chill"]}


def good_explanation(messages):
    """Builds a valid explanation for whatever songs appear in the prompt."""
    ids = [int(i) for i in re.findall(r'"song_id": (\d+)', messages[1].content)]
    return {"summary": "Here are your picks.", "notes": [{"song_id": i, "note": "Fits the request."} for i in ids]}


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(embedding_provider="local", openai_api_key="sk-test",
                    chroma_persist_dir=tmp_path / "chroma", log_file=tmp_path / "test.log",
                    songs_csv=SONGS_CSV, knowledge_dir=KNOWLEDGE_DIR)


@pytest.fixture
def kb(settings) -> KnowledgeBase:
    base = KnowledgeBase(settings, embeddings=DeterministicFakeEmbedding(size=32))
    base.ensure_index()
    return base


@pytest.fixture
def log_file(settings):
    from src.logging_config import setup_logging

    setup_logging(settings.log_file)
    yield settings.log_file
    logger = logging.getLogger("tunescout")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
