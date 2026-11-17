"""TuneScout AI: Streamlit UI.

Run with:  streamlit run app.py

Describe what you want to hear in plain words. The agent retrieves music
knowledge (RAG), turns your request into a taste profile, ranks the catalog with
the original rule-based scorer, checks the results, and explains them. You can
then adjust the profile yourself and re-rank without any AI.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import llm as llm_module  # noqa: E402  (module import keeps it patchable in tests)
from src.agent import AgentError, Catalog, RequestError, TuneScoutAgent, rank  # noqa: E402
from src.config import ConfigError, Settings, load_settings  # noqa: E402
from src.llm import LLMError  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402
from src.rag import KnowledgeBase, KnowledgeBaseError  # noqa: E402
from src.recommender import STRATEGIES  # noqa: E402

logger = logging.getLogger("tunescout.app")

EXAMPLES = [
    "Something mellow for studying, nothing too sad",
    "High-energy songs for the gym",
    "Peaceful metal",
    "Romantic music for a dinner date",
]
ANY = "(any)"

st.set_page_config(page_title="TuneScout AI", page_icon="🎧", layout="wide")


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the music knowledge base...")
def open_knowledge_base(settings: Settings) -> KnowledgeBase:
    kb = KnowledgeBase(settings)
    kb.ensure_index()
    return kb


@st.cache_resource
def open_catalog(settings: Settings) -> Catalog:
    return Catalog.from_csv(settings.songs_csv)


def startup() -> tuple[Settings, KnowledgeBase, Catalog]:
    try:
        settings = load_settings()
        setup_logging(settings.log_file)
        settings.require_embeddings()
        return settings, open_knowledge_base(settings), open_catalog(settings)
    except (ConfigError, KnowledgeBaseError) as exc:
        st.error(f"**Setup problem:** {exc}")
        st.info("Copy `.env.example` to `.env`, add your API key, then refresh this page.")
        st.stop()


settings, kb, catalog = startup()
st.session_state.setdefault("result", None)
st.session_state.setdefault("manual", None)
st.session_state.setdefault("request", "")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Retrieve** genre, mood and activity notes that match your words\n"
        "2. **Interpret** your request as a taste profile (AI)\n"
        "3. **Validate** labels and numbers against the catalog\n"
        "4. **Rank** all songs with the original scoring rules\n"
        "5. **Critique** the results and re-plan once if needed\n"
        "6. **Explain** the picks, checked against the catalog"
    )
    model = settings.llm_model
    st.caption(f"LLM: {settings.llm_provider} · {model}  \nEmbeddings: {settings.embedding_provider}  \n"
               f"Catalog: {len(catalog.songs)} songs, {len(catalog.genres)} genres, {len(catalog.moods)} moods")
    with st.expander("system.log (last 60 lines)"):
        path = settings.log_file
        lines = path.read_text(encoding="utf-8").splitlines()[-60:] if path.exists() else []
        st.code("\n".join(lines) or "(empty)", language="log")


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #
st.title("🎧 TuneScout AI")
st.caption("Describe what you want to hear. TuneScout finds the best matches in its catalog and explains why.")

st.write("Try an example:")
columns = st.columns(len(EXAMPLES))
for column, example in zip(columns, EXAMPLES):
    if column.button(example, use_container_width=True):
        st.session_state.request = example

request = st.text_area("What do you feel like listening to?", key="request", max_chars=settings.max_request_chars,
                       placeholder="e.g. upbeat songs for a road trip, but not too intense")

if st.button("Find music", type="primary", disabled=not request.strip()):
    try:
        with st.spinner("Thinking about your request..."):
            agent = TuneScoutAgent(settings, kb, llm_module.get_chat_model(settings), catalog)
            st.session_state.result = agent.run(request)
            st.session_state.manual = None
    except (RequestError, AgentError, LLMError, ConfigError, KnowledgeBaseError) as exc:
        st.error(str(exc))
    except Exception as exc:
        logger.exception("Unexpected error while handling %r", request)
        st.error(f"Unexpected error: {exc}. Full details are in system.log.")


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def score_bar(score: float, max_score: float) -> str:
    filled = round(score / max_score * 20)
    return "█" * filled + "░" * (20 - filled)


def show_recommendations(recs, max_score: float, use_notes: bool) -> None:
    for i, rec in enumerate(recs, start=1):
        song = rec.song
        with st.container(border=True):
            left, right = st.columns([3, 2])
            left.markdown(f"**#{i} {song['title']}** · {song['artist']}  \n{song['genre']} / {song['mood']}")
            right.markdown(f"`{score_bar(rec.score, max_score)}`  \n**{rec.score:.2f}** / {max_score:.1f}")
            if use_notes and rec.note:
                st.write(rec.note)
            with st.expander("Scoring reasons"):
                for reason in rec.reasons or ["no preferences matched"]:
                    st.markdown(f"- {reason}")


result = st.session_state.result
if result is not None:
    if result.status == "off_topic":
        st.warning(result.message)
    else:
        st.subheader("What TuneScout understood")
        st.info(result.interpretation)
        profile_text = " · ".join(f"**{k.replace('_', ' ')}:** {v}" for k, v in result.profile.items())
        st.markdown(f"{profile_text} · **mode:** {result.ranking_mode}")
        if result.revised:
            st.success("The agent's first plan failed a quality check, so it re-planned. See the agent trace below.")
        for issue in result.issues:
            st.warning(f"Heads up: {issue}")

        st.subheader("Recommendations")
        if result.summary:
            st.write(result.summary)
        if result.explanation_source == "fallback":
            st.caption("The AI's explanation failed a fact check, so these notes come straight from the scoring rules.")
        show_recommendations(result.recommendations, result.max_score, use_notes=True)

        col_trace, col_knowledge = st.columns(2)
        with col_trace.expander("Agent trace"):
            for step in result.steps:
                st.markdown(f"- {step}")
        with col_knowledge.expander(f"Knowledge used ({len(result.passages)} passages retrieved)"):
            for passage in result.passages:
                used = "✅ " if passage.section in result.knowledge_used else ""
                st.markdown(f"{used}**{passage.citation}** · distance {passage.distance:.3f}")
                st.text(passage.text)

        # ---- Human in the loop: adjust the AI's profile and re-rank without AI ---- #
        st.subheader("Not quite right? Adjust it yourself")
        st.caption("Change anything the AI got wrong. Re-ranking uses only the scoring rules, with no AI.")
        p = result.profile
        with st.form("adjust"):
            c1, c2, c3 = st.columns(3)
            genres, moods = [ANY, *catalog.genres], [ANY, *catalog.moods]
            genre = c1.selectbox("Genre", genres, index=genres.index(p.get("favorite_genre", ANY)))
            mood = c2.selectbox("Mood", moods, index=moods.index(p.get("favorite_mood", ANY)))
            modes = list(STRATEGIES)
            mode = c3.selectbox("Ranking mode", modes, index=modes.index(result.ranking_mode))
            c4, c5, c6 = st.columns(3)
            energy = c4.slider("Energy", 0.0, 1.0, float(p.get("target_energy", 0.5)), 0.05)
            use_energy = c4.checkbox("Use energy", value="target_energy" in p)
            valence = c5.slider("Brightness (valence)", 0.0, 1.0, float(p.get("target_valence", 0.5)), 0.05)
            use_valence = c5.checkbox("Use brightness", value="target_valence" in p)
            acoustic_options = {"No preference": None, "Acoustic": True, "Produced / electronic": False}
            current = {None: 0, True: 1, False: 2}[p.get("likes_acoustic")]
            acoustic = c6.radio("Sound", list(acoustic_options), index=current)
            if st.form_submit_button("Re-rank"):
                manual = {"favorite_genre": None if genre == ANY else genre,
                          "favorite_mood": None if mood == ANY else mood,
                          "target_energy": energy if use_energy else None,
                          "target_valence": valence if use_valence else None,
                          "likes_acoustic": acoustic_options[acoustic]}
                manual = {k: v for k, v in manual.items() if v is not None}
                logger.info("Human override for %r: AI profile=%s mode=%s -> user profile=%s mode=%s",
                            result.request, result.profile, result.ranking_mode, manual, mode)
                st.session_state.manual = (manual, mode, rank(manual, mode, catalog, settings.recommendations_k))

        if st.session_state.manual:
            manual, mode, recs = st.session_state.manual
            st.markdown(f"**Your adjusted ranking** (mode: {mode})")
            show_recommendations(recs, STRATEGIES[mode].max_score, use_notes=False)
