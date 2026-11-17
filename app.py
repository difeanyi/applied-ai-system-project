"""Streamlit UI for the Data-Driven Study Syllabus and Quiz Generator.

Run with:  streamlit run app.py

Pipeline: upload PDF/PPTX -> ingest into ChromaDB -> agentic scheduler builds a
JSON study plan -> RAG quiz generator writes cited questions for every topic.
Everything is logged to system.log.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from study_generator import llm as llm_module  # noqa: E402  (module import keeps it patchable in tests)
from study_generator.config import ConfigError, Settings, load_settings  # noqa: E402
from study_generator.ingestion import (  # noqa: E402
    clear_store,
    get_document_outline,
    get_vector_store,
    ingest_files,
    remove_document,
    save_upload,
)
from study_generator.llm import LLMError  # noqa: E402
from study_generator.logging_config import setup_logging  # noqa: E402
from study_generator.quiz import NOT_FOUND_MESSAGE, generate_quizzes  # noqa: E402
from study_generator.scheduler import (  # noqa: E402
    MAX_STUDY_DAYS,
    SchedulerError,
    build_schedule,
    resolve_study_days,
)

logger = logging.getLogger("study_generator.app")
LETTERS = "ABCD"

st.set_page_config(page_title="Study Syllabus & Quiz Generator", page_icon="📚", layout="wide")


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Opening the document store...")
def open_store(settings: Settings):
    """One Chroma connection per settings, reused across Streamlit reruns."""
    return get_vector_store(settings)


def load_app_settings() -> Settings:
    try:
        settings = load_settings()
        setup_logging(settings.log_file)
        settings.require_embeddings()
        return settings
    except ConfigError as exc:
        st.error(f"**Configuration problem:** {exc}")
        st.info("Copy `.env.example` to `.env`, fill in your values, then refresh this page.")
        st.stop()


def report_unexpected(action: str, exc: Exception) -> None:
    logger.exception("Unexpected error during %s", action)
    st.error(f"Unexpected error during {action}: {exc}. Full details are in system.log.")


settings = load_app_settings()
store = open_store(settings)
for key, default in (("schedule", None), ("quizzes", None)):
    st.session_state.setdefault(key, default)


# --------------------------------------------------------------------------- #
# Sidebar: configuration, stored documents, log viewer
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    model = settings.anthropic_model if settings.llm_provider == "anthropic" else settings.openai_model
    st.markdown(
        f"- **LLM:** {settings.llm_provider} · `{model}`\n"
        f"- **Embeddings:** {settings.embedding_provider}\n"
        f"- **Chunks:** {settings.chunk_size_words} words, top {settings.retrieval_top_k} retrieved\n"
        f"- **Questions per topic:** {settings.questions_per_topic}"
    )

    st.header("Stored documents")
    outline = get_document_outline(store)
    if not outline:
        st.caption("Nothing stored yet.")
    for filename, entries in outline.items():
        kind = entries[0]["location_type"] if entries else "page"
        col_name, col_button = st.columns([4, 1])
        col_name.markdown(f"**{filename}**  \n{len(entries)} {kind}s")
        if col_button.button("✕", key=f"remove-{filename}", help=f"Remove {filename}"):
            remove_document(store, filename)
            st.session_state.schedule = st.session_state.quizzes = None
            st.rerun()
    if outline:
        confirm = st.checkbox("I want to remove all documents")
        if st.button("Clear all documents", disabled=not confirm):
            clear_store(store)
            st.session_state.schedule = st.session_state.quizzes = None
            st.rerun()

    with st.expander("system.log (last 60 lines)"):
        log_path = settings.log_file
        lines = log_path.read_text(encoding="utf-8").splitlines()[-60:] if log_path.exists() else []
        st.code("\n".join(lines) or "(empty)", language="log")


# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("📚 Study Syllabus & Quiz Generator")
st.caption("Upload your course materials, get a day-by-day study plan, and practise with quizzes "
           "written only from your documents, with a source for every question.")

# ---- 1. Upload ------------------------------------------------------------- #
st.subheader("1. Upload your materials")
uploads = st.file_uploader("PDF or PowerPoint files", type=["pdf", "pptx"], accept_multiple_files=True)
if st.button("Process files", type="primary", disabled=not uploads):
    try:
        with st.status("Reading, chunking and embedding your files...", expanded=True) as status:
            paths = [save_upload(f.name, f.getvalue(), settings) for f in uploads]
            logger.info("App: user uploaded %s", [p.name for p in paths])
            results, errors = ingest_files(paths, settings, store)
            for r in results:
                if r.already_ingested:
                    st.write(f"↺ **{r.filename}**: already stored, skipped")
                else:
                    unit = "page" if r.file_type == "pdf" else "slide"
                    note = f" (no text on {unit}s {r.skipped_units})" if r.skipped_units else ""
                    st.write(f"✅ **{r.filename}**: {r.units_with_text} {unit}s → {r.chunks_added} chunks{note}")
            for name, message in errors.items():
                st.write(f"❌ **{name}**: {message}")
            status.update(
                label=f"Processed {len(results)} file(s)" + (f", {len(errors)} failed" if errors else ""),
                state="error" if errors and not results else "complete",
            )
        if any(not r.already_ingested for r in results):
            st.session_state.schedule = st.session_state.quizzes = None  # materials changed
            st.rerun()
    except Exception as exc:
        report_unexpected("file processing", exc)

# ---- 2. Schedule ----------------------------------------------------------- #
st.subheader("2. Build your study schedule")
if not outline:
    st.info("Upload and process at least one file first.")
else:
    mode = st.radio("Plan by", ["Exam date", "Number of days"], horizontal=True)
    today = date.today()
    exam_date, day_count = None, None
    if mode == "Exam date":
        exam_date = st.date_input("Exam date", value=today + timedelta(days=7),
                                  min_value=today + timedelta(days=1),
                                  max_value=today + timedelta(days=MAX_STUDY_DAYS))
    else:
        day_count = st.number_input("Study days", min_value=1, max_value=MAX_STUDY_DAYS, value=5)

    if st.button("Generate schedule", type="primary"):
        try:
            total = resolve_study_days(exam_date=exam_date, total_days=day_count, today=today)
            logger.info("App: schedule requested for %d days (exam date: %s)", total, exam_date)
            with st.spinner(f"Planning {total} days of study..."):
                st.session_state.schedule = build_schedule(
                    outline, total, settings, start_date=today, exam_date=exam_date,
                    llm=llm_module.get_chat_model(settings),
                )
            st.session_state.quizzes = None
        except (SchedulerError, LLMError, ConfigError) as exc:
            st.error(str(exc))
        except Exception as exc:
            report_unexpected("schedule generation", exc)

schedule = st.session_state.schedule
if schedule:
    header = f"**{schedule['total_days']} study days** starting {schedule['start_date']}"
    if schedule["exam_date"]:
        header += f" · exam on {schedule['exam_date']}"
    st.markdown(header)
    for day in schedule["days"]:
        badge = " 🔁 Review" if day["is_review_day"] else ""
        with st.expander(f"Day {day['day']} · {day['date']} — {day['focus']}{badge}"):
            for topic in day["topics"]:
                refs = "; ".join(
                    f"{s['filename']} ({', '.join(map(str, s['pages'])) or 'all'})" for s in topic["sources"]
                )
                st.markdown(f"**{topic['title']}**: {topic['summary']}  \n<small>📄 {refs}</small>",
                            unsafe_allow_html=True)
    if schedule["uncovered_pages"]:
        gaps = "; ".join(f"{f}: {pages}" for f, pages in schedule["uncovered_pages"].items())
        st.caption(f"Pages not assigned to any topic (often title or reference pages): {gaps}")
    st.download_button("Download schedule (JSON)", json.dumps(schedule, indent=2),
                       file_name="study_schedule.json", mime="application/json")

# ---- 3. Quizzes ------------------------------------------------------------ #
if schedule:
    st.subheader("3. Generate quizzes")
    topic_count = sum(len(d["topics"]) for d in schedule["days"])
    st.caption(f"{topic_count} topics × {settings.questions_per_topic} questions. "
               "Each topic uses one or two LLM calls.")
    if st.button("Generate quizzes", type="primary"):
        progress = st.progress(0.0, text="Starting...")
        try:
            quizzes = generate_quizzes(
                schedule, store, settings, llm=llm_module.get_chat_model(settings),
                on_progress=lambda i, n, title: progress.progress((i - 1) / n, text=f"Topic {i}/{n}: {title}"),
            )
            st.session_state.quizzes = [q.to_dict() for q in quizzes]
            progress.progress(1.0, text="Done")
        except (LLMError, ConfigError) as exc:
            progress.empty()
            st.error(str(exc))
        except Exception as exc:
            progress.empty()
            report_unexpected("quiz generation", exc)


def render_question(quiz_index: int, q_index: int, q: dict) -> None:
    st.markdown(f"**Q{q_index + 1}. {q['question']}**")
    labels = [f"{LETTERS[i]}. {option}" for i, option in enumerate(q["options"])]
    choice = st.radio("Your answer", labels, index=None, key=f"answer-{quiz_index}-{q_index}",
                      label_visibility="collapsed")
    if choice is not None:
        if labels.index(choice) == q["correct_option"]:
            st.success("Correct!")
        else:
            st.error(f"Not quite. The answer is {LETTERS[q['correct_option']]}. {q['correct_answer']}")
        st.markdown(f"_{q['explanation']}_")
    st.caption(f"📄 Source: {q['citation']}")


quizzes = st.session_state.quizzes
if quizzes:
    by_day: dict[int, list[tuple[int, dict]]] = {}
    for index, quiz in enumerate(quizzes):
        by_day.setdefault(quiz["day"], []).append((index, quiz))

    for day_number, day_quizzes in by_day.items():
        with st.expander(f"Day {day_number} quizzes", expanded=day_number == 1):
            for index, quiz in day_quizzes:
                with st.container(border=True):
                    st.markdown(f"#### {quiz['topic']}")
                    if quiz["status"] == "not_found":
                        st.warning(NOT_FOUND_MESSAGE)
                    elif quiz["status"] == "error":
                        st.error(f"This topic failed: {quiz['message']}")
                    for q_index, q in enumerate(quiz["questions"]):
                        render_question(index, q_index, q)
                    for warning in quiz["warnings"]:
                        if warning.startswith("Only "):
                            st.caption(f"⚠️ {warning}")
                    if quiz["retrieved"]:
                        with st.popover(f"Retrieved context ({len(quiz['retrieved'])} chunks)"):
                            for chunk in quiz["retrieved"]:
                                st.markdown(f"**{chunk['source']}**, {chunk['location_type']} "
                                            f"{chunk['page']} · distance {chunk['distance']:.3f}")
                                st.text(chunk["text"][:1500])

    st.download_button("Download quizzes (JSON)", json.dumps(quizzes, indent=2),
                       file_name="quizzes.json", mime="application/json")
