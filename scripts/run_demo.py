"""Run TuneScout AI on sample requests and save the real outputs.

Uses the same agent as the Streamlit app. Results are printed, saved to
sample_outputs/demo_results.json, and logged to system.log.

    python scripts/run_demo.py                      # the built-in sample requests
    python scripts/run_demo.py "music for a rainy day"
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import AgentError, RequestError, TuneScoutAgent  # noqa: E402
from src.config import ConfigError, load_settings  # noqa: E402
from src.llm import LLMError, get_chat_model  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402
from src.rag import KnowledgeBase, KnowledgeBaseError  # noqa: E402

SAMPLE_REQUESTS = [
    "Something mellow for studying, nothing too sad",
    "High-energy songs for the gym",
    "Peaceful metal",
    "Can you write my history essay for me?",
]


def show(result) -> None:
    print(f"\n>>> {result.request}")
    if result.status != "ok":
        print(f"    [{result.status}] {result.message}")
        return
    print(f"    Understood: {result.interpretation}")
    print(f"    Profile: {result.profile} | mode: {result.ranking_mode}"
          f"{' (re-planned)' if result.revised else ''}")
    print(f"    Knowledge used: {result.knowledge_used}")
    for issue in result.issues:
        print(f"    Heads up: {issue}")
    print(f"    {result.summary}")
    for i, r in enumerate(result.recommendations, 1):
        print(f"    #{i} {r.song['title']} ({r.song['genre']}/{r.song['mood']}) "
              f"{r.score:.2f}/{result.max_score:.1f}: {r.note}")
    print("    Agent trace:")
    for step in result.steps:
        print(f"      - {step}")


def main() -> None:
    requests = sys.argv[1:] or SAMPLE_REQUESTS
    settings = load_settings()
    setup_logging(settings.log_file)
    settings.require_llm()
    kb = KnowledgeBase(settings)
    kb.ensure_index()
    agent = TuneScoutAgent(settings, kb, get_chat_model(settings))

    saved = []
    for text in requests:
        try:
            result = agent.run(text)
        except (RequestError, AgentError) as exc:
            print(f"\n>>> {text}\n    [rejected] {exc}")
            saved.append({"request": text, "status": "rejected", "message": str(exc)})
            continue
        show(result)
        saved.append(asdict(result))

    out = ROOT / "sample_outputs"
    out.mkdir(exist_ok=True)
    (out / "demo_results.json").write_text(json.dumps(saved, indent=2, default=str))
    print(f"\nSaved to sample_outputs/demo_results.json; full log in {settings.log_file.name}")


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, LLMError, KnowledgeBaseError) as exc:
        sys.exit(f"Error: {exc}")
