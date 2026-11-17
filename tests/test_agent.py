from __future__ import annotations

import pytest

from src.agent import (
    OFF_TOPIC_MESSAGE,
    AgentError,
    Catalog,
    InterpretedRequest,
    RequestError,
    TuneScoutAgent,
    critique,
    rank,
    validate_interpretation,
)
from src.config import SONGS_CSV
from src.llm import LLMError

from .conftest import FakeStructuredLLM, good_explanation, interpretation

CATALOG = Catalog.from_csv(SONGS_CSV)


def make_agent(settings, kb, replies):
    llm = FakeStructuredLLM(replies)
    return TuneScoutAgent(settings, kb, llm, CATALOG), llm


# ---- deterministic checks --------------------------------------------------------
def test_validation_catches_invented_labels_and_bad_numbers():
    parsed = InterpretedRequest(**interpretation(genre="lo-fi", mood="sleepy", energy=1.5, mode="vibes"))
    problems = " ".join(validate_interpretation(parsed, CATALOG))
    assert "'lo-fi' is not a catalog genre" in problems
    assert "'sleepy' is not a catalog mood" in problems
    assert "target_energy is 1.5" in problems
    assert "ranking_mode 'vibes' is not valid" in problems


def test_validation_requires_at_least_one_preference():
    problems = validate_interpretation(InterpretedRequest(**interpretation()), CATALOG)
    assert any("No preferences" in p for p in problems)


def test_critic_flags_the_metal_fan_with_no_metal():
    # The "Contradictory metal" profile from the original model card.
    profile = {"favorite_genre": "metal", "favorite_mood": "peaceful", "target_energy": 0.1,
               "target_valence": 0.3, "likes_acoustic": True}
    recs = rank(profile, "balanced", CATALOG, 5)
    assert "metal" not in [r.song["genre"] for r in recs]
    assert any("genre 'metal'" in issue for issue in critique(profile, recs, "balanced", CATALOG))


def test_scorer_is_unchanged_from_the_original_project():
    profile = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8,
               "target_valence": 0.8, "likes_acoustic": False}
    recs = rank(profile, "balanced", CATALOG, 5)
    assert [(r.song["title"], round(r.score, 2)) for r in recs] == [
        ("Sunrise City", 4.85), ("Rooftop Lights", 3.74), ("Gym Hero", 3.70),
        ("Concrete Crown", 2.81), ("Neon Pulse", 2.69)]  # the README's worked example


# ---- full agent loop ---------------------------------------------------------------
def test_happy_path(settings, kb, log_file):
    agent, llm = make_agent(settings, kb, [
        interpretation(genre="lofi", mood="focused", energy=0.4, acoustic=True, mode="energy-focused"),
        good_explanation,
    ])
    result = agent.run("something mellow for studying")

    assert result.status == "ok"
    assert result.recommendations[0].song["title"] == "Focus Flow"
    assert result.ranking_mode == "energy-focused" and not result.revised
    assert result.explanation_source == "ai"
    assert all(r.note == "Fits the request." for r in result.recommendations)
    assert len(llm.calls) == 2
    # The retrieved passages were sent to the model.
    assert result.passages[0].text in llm.calls[0][1].content
    assert "Agent run finished" in log_file.read_text()


def test_invalid_label_is_sent_back_and_corrected(settings, kb):
    agent, llm = make_agent(settings, kb, [
        interpretation(genre="lo-fi hip hop", energy=0.4),
        interpretation(genre="lofi", energy=0.4),
        good_explanation,
    ])
    result = agent.run("lo-fi hip hop please")
    assert result.profile["favorite_genre"] == "lofi"
    assert "'lo-fi hip hop' is not a catalog genre" in llm.calls[1][-1].content
    assert any("attempt 1 rejected" in s for s in result.steps)


def test_gives_up_after_three_invalid_answers(settings, kb):
    agent, _ = make_agent(settings, kb, [interpretation(genre="k-pop")] * 3)
    with pytest.raises(AgentError, match="could not turn this request"):
        agent.run("k-pop")


def test_off_topic_requests_are_refused(settings, kb):
    agent, llm = make_agent(settings, kb, [interpretation(music=False)])
    result = agent.run("write my history essay")
    assert result.status == "off_topic" and result.message == OFF_TOPIC_MESSAGE
    assert result.recommendations == [] and len(llm.calls) == 1


def test_critic_triggers_a_replan_that_fixes_the_metal_case(settings, kb):
    agent, llm = make_agent(settings, kb, [
        interpretation(genre="metal", mood="peaceful", energy=0.1, valence=0.3, acoustic=True),
        interpretation(genre="metal", mood="peaceful", energy=0.1, valence=0.3, acoustic=True,
                       mode="genre-first"),
        good_explanation,
    ])
    result = agent.run("peaceful metal")
    assert result.revised and result.ranking_mode == "genre-first"
    assert result.recommendations[0].song["title"] == "Iron Tempest"
    assert "none made the top 5" in llm.calls[1][-1].content


def test_replan_is_rejected_if_it_is_not_better(settings, kb):
    same = interpretation(genre="metal", mood="peaceful", energy=0.1, valence=0.3, acoustic=True)
    agent, _ = make_agent(settings, kb, [same, same, good_explanation])
    result = agent.run("peaceful metal")
    assert not result.revised and result.issues
    assert any("Re-plan rejected" in s for s in result.steps)


def test_explanations_mentioning_other_songs_fall_back_to_scorer_reasons(settings, kb):
    def hallucinating(messages):
        reply = good_explanation(messages)
        reply["summary"] = "Also try Iron Tempest!"  # not in the lofi results
        return reply

    agent, llm = make_agent(settings, kb, [
        interpretation(genre="lofi", mood="chill", energy=0.4), hallucinating, hallucinating])
    result = agent.run("lofi")
    assert result.explanation_source == "fallback"
    assert "Iron Tempest" in llm.calls[2][-1].content  # feedback named the problem
    assert result.recommendations[0].note.startswith("genre match: lofi")


@pytest.mark.parametrize("text, message", [("   ", "describe"), ("x" * 501, "under 500")])
def test_bad_input_is_rejected_before_any_ai_call(settings, kb, text, message):
    agent, llm = make_agent(settings, kb, [])
    with pytest.raises(RequestError, match=message):
        agent.run(text)
    assert llm.calls == []


def test_invalid_api_key_is_fatal(settings, kb):
    class AuthenticationError(Exception):
        pass

    agent, _ = make_agent(settings, kb, [AuthenticationError("401")])
    with pytest.raises(LLMError, match="rejected the API key") as info:
        agent.run("happy pop")
    assert info.value.fatal
