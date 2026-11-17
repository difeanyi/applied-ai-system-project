from __future__ import annotations

from datetime import date

import pytest

from study_generator.llm import LLMError
from study_generator.scheduler import (
    SchedulerError,
    build_schedule,
    format_outline,
    resolve_study_days,
)

from .conftest import FakeStructuredLLM

OUTLINE = {
    "bio.pdf": [
        {"location": 1, "location_type": "page", "title": "Photosynthesis"},
        {"location": 2, "location_type": "page", "title": "Respiration"},
    ],
    "chem.pptx": [{"location": 1, "location_type": "slide", "title": "Atoms"}],
}


def topic(title, filename, pages):
    return {"title": title, "summary": f"Learn {title}", "search_query": title,
            "sources": [{"filename": filename, "pages": pages}]}


def day(n, topics, review=False):
    return {"day": n, "focus": f"Day {n}", "is_review_day": review, "topics": topics}


GOOD_PLAN = {"days": [
    day(1, [topic("Photosynthesis", "bio.pdf", [1])]),
    day(2, [topic("Respiration", "bio.pdf", [2]), topic("Atoms", "chem.pptx", [1])]),
    day(3, [topic("Review", "bio.pdf", [1, 2])], review=True),
]}


# ---- day calculation ---------------------------------------------------------
def test_exam_date_gives_days_until_exam():
    assert resolve_study_days(exam_date=date(2026, 12, 1), today=date(2026, 11, 24)) == 7


@pytest.mark.parametrize("kwargs, message", [
    ({"exam_date": date(2026, 11, 24)}, "after today"),
    ({"total_days": 0}, "at least 1"),
    ({"total_days": 500}, "60 days or fewer"),
    ({}, "Provide either"),
])
def test_bad_day_inputs_are_rejected(kwargs, message):
    with pytest.raises(SchedulerError, match=message):
        resolve_study_days(today=date(2026, 11, 24), **kwargs)


def test_outline_lists_every_page():
    text = format_outline(OUTLINE)
    assert "File: bio.pdf (2 pages with text)" in text
    assert "slide 1: Atoms" in text


# ---- agent loop ----------------------------------------------------------------
def test_valid_plan_is_returned_as_json_with_dates(settings):
    llm = FakeStructuredLLM([GOOD_PLAN])
    schedule = build_schedule(OUTLINE, 3, settings, start_date=date(2026, 11, 24), llm=llm)

    assert len(llm.calls) == 1
    assert schedule["total_days"] == 3
    assert [d["date"] for d in schedule["days"]] == ["2026-11-24", "2026-11-25", "2026-11-26"]
    assert schedule["days"][1]["topics"][1]["sources"] == [{"filename": "chem.pptx", "pages": [1]}]
    assert schedule["uncovered_pages"] == {}


def test_agent_revises_an_invalid_plan(settings):
    bad_plan = {"days": [
        day(1, [topic("Photosynthesis", "bio.pdf", [1, 99])]),  # page 99 does not exist
        day(2, [topic("Made up", "physics.pdf", [1])]),  # unknown file; chem.pptx uncovered
    ]}  # and only 2 of 3 days
    llm = FakeStructuredLLM([bad_plan, GOOD_PLAN])
    schedule = build_schedule(OUTLINE, 3, settings, llm=llm)

    assert len(llm.calls) == 2
    feedback = llm.calls[1][-1].content
    assert "exactly 3" in feedback
    assert "pages [99]" in feedback
    assert "unknown file 'physics.pdf'" in feedback
    assert "'chem.pptx' is not covered" in feedback
    assert len(schedule["days"]) == 3


def test_gives_up_after_max_attempts(settings):
    bad_plan = {"days": [day(1, [])]}
    llm = FakeStructuredLLM([bad_plan] * 3)
    with pytest.raises(SchedulerError, match="after 3 attempts"):
        build_schedule(OUTLINE, 1, settings, llm=llm)
    assert len(llm.calls) == 3


def test_reports_uncovered_pages(settings):
    plan = {"days": [day(1, [topic("Photo", "bio.pdf", [1]), topic("Atoms", "chem.pptx", [1])])]}
    schedule = build_schedule(OUTLINE, 1, settings, llm=FakeStructuredLLM([plan]))
    assert schedule["uncovered_pages"] == {"bio.pdf": [2]}


def test_empty_outline_is_rejected(settings):
    with pytest.raises(SchedulerError, match="No documents"):
        build_schedule({}, 3, settings, llm=FakeStructuredLLM([]))


def test_invalid_api_key_is_a_fatal_error(settings):
    class AuthenticationError(Exception):
        pass

    llm = FakeStructuredLLM([AuthenticationError("Error code: 401")])
    with pytest.raises(LLMError, match="rejected the API key") as info:
        build_schedule(OUTLINE, 3, settings, llm=llm)
    assert info.value.fatal
