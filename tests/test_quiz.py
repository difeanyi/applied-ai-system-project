from __future__ import annotations

import logging

import pytest

from study_generator.ingestion import ingest_file
from study_generator.llm import LLMError
from study_generator.logging_config import setup_logging
from study_generator.quiz import (
    NOT_FOUND_MESSAGE,
    build_source_filter,
    generate_quizzes,
    generate_topic_quiz,
)

from .conftest import FakeStructuredLLM, make_pdf

TOPIC = {"title": "Photosynthesis", "summary": "How plants make sugar",
         "search_query": "chlorophyll light", "sources": [{"filename": "bio.pdf", "pages": [1]}]}


def question(n, filename="bio.pdf", page=1, options=None):
    return {"question": f"Question {n}?", "options": options or ["A", "B", "C", "D"],
            "correct_option": 1, "explanation": "Stated on page 1.",
            "source_filename": filename, "source_location": page}


@pytest.fixture
def loaded_store(tmp_path, settings, store):
    pdf = make_pdf(tmp_path / "bio.pdf", [
        ["Photosynthesis", "Chlorophyll absorbs light energy."],
        ["Respiration", "Mitochondria produce ATP."],
    ])
    ingest_file(pdf, settings, store)
    return store


@pytest.fixture
def log_file(settings):
    setup_logging(settings.log_file)
    yield settings.log_file
    logger = logging.getLogger("study_generator")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


# ---- retrieval filter ----------------------------------------------------------
def test_filter_for_one_source_with_pages():
    assert build_source_filter([{"filename": "a.pdf", "pages": [1, 2]}]) == {
        "$and": [{"source": "a.pdf"}, {"page": {"$in": [1, 2]}}]}


def test_filter_for_several_sources():
    f = build_source_filter([{"filename": "a.pdf", "pages": []}, {"filename": "b.pptx", "pages": [3]}])
    assert f == {"$or": [{"source": "a.pdf"}, {"$and": [{"source": "b.pptx"}, {"page": {"$in": [3]}}]}]}


# ---- generation and guardrails ---------------------------------------------------
def test_generates_cited_questions_from_retrieved_chunks(settings, loaded_store, log_file):
    llm = FakeStructuredLLM([{"insufficient_context": False,
                              "questions": [question(1), question(2), question(3)]}])
    quiz = generate_topic_quiz(1, TOPIC, loaded_store, settings, llm)

    assert quiz.status == "ok"
    assert len(quiz.questions) == 3
    assert quiz.questions[0]["citation"] == "bio.pdf, page 1"
    assert quiz.questions[0]["correct_answer"] == "B"
    # Only the scheduled page was retrieved and sent to the model.
    assert {(c.source, c.page) for c in quiz.retrieved} == {("bio.pdf", 1)}
    prompt = llm.calls[0][1].content
    assert "Chlorophyll absorbs light energy." in prompt
    assert "Mitochondria" not in prompt
    # The exact chunk text is in the log.
    log = log_file.read_text()
    assert "Retrieval for query 'Photosynthesis: chlorophyll light'" in log
    assert "Chlorophyll absorbs light energy." in log


def test_hallucinated_citations_are_rejected_and_replaced(settings, loaded_store):
    first = {"insufficient_context": False, "questions": [
        question(1), question(2, filename="fake.pdf"), question(3, page=2),  # page 2 was not retrieved
    ]}
    second = {"insufficient_context": False, "questions": [question(4), question(5)]}
    llm = FakeStructuredLLM([first, second])
    quiz = generate_topic_quiz(1, TOPIC, loaded_store, settings, llm)

    assert [q["question"] for q in quiz.questions] == ["Question 1?", "Question 4?", "Question 5?"]
    feedback = llm.calls[1][-1].content
    assert "cites fake.pdf page 1" in feedback and "cites bio.pdf page 2" in feedback
    assert "Return 2 new valid question(s)" in feedback


def test_malformed_options_are_rejected(settings, loaded_store):
    bad = {"insufficient_context": False, "questions": [
        question(1, options=["A", "B", "C"]), question(2, options=["A", "A", "B", "C"])]}
    llm = FakeStructuredLLM([bad, {"insufficient_context": True, "questions": []}])
    quiz = generate_topic_quiz(1, TOPIC, loaded_store, settings, llm)
    assert quiz.status == "not_found"
    assert any("3 options" in w for w in quiz.warnings)
    assert any("distinct" in w for w in quiz.warnings)


def test_insufficient_context_gives_not_found_message(settings, loaded_store):
    llm = FakeStructuredLLM([{"insufficient_context": True, "questions": []}])
    quiz = generate_topic_quiz(1, TOPIC, loaded_store, settings, llm)
    assert quiz.status == "not_found"
    assert quiz.message == NOT_FOUND_MESSAGE == "Information not found in documents."
    assert quiz.questions == []


def test_empty_store_gives_not_found_without_calling_llm(settings, store):
    llm = FakeStructuredLLM([])
    quiz = generate_topic_quiz(1, TOPIC, store, settings, llm)
    assert quiz.message == NOT_FOUND_MESSAGE
    assert llm.calls == []


# ---- looping over the schedule ---------------------------------------------------
SCHEDULE = {"days": [
    {"day": 1, "topics": [TOPIC]},
    {"day": 2, "topics": [{**TOPIC, "title": "Respiration",
                           "sources": [{"filename": "bio.pdf", "pages": [2]}]}]},
]}


def test_loop_continues_after_a_non_fatal_error(settings, loaded_store):
    class RateLimitError(Exception):
        pass

    good = {"insufficient_context": False, "questions": [
        question(1, page=2), question(2, page=2), question(3, page=2)]}
    llm = FakeStructuredLLM([RateLimitError("429"), good])
    progress = []
    quizzes = generate_quizzes(SCHEDULE, loaded_store, settings, llm,
                               on_progress=lambda i, n, t: progress.append((i, n, t)))

    assert [q.status for q in quizzes] == ["error", "ok"]
    assert "rate limit" in quizzes[0].message
    assert progress == [(1, 2, "Photosynthesis"), (2, 2, "Respiration")]


def test_loop_stops_on_a_fatal_error(settings, loaded_store):
    class AuthenticationError(Exception):
        pass

    llm = FakeStructuredLLM([AuthenticationError("401")])
    with pytest.raises(LLMError):
        generate_quizzes(SCHEDULE, loaded_store, settings, llm)
