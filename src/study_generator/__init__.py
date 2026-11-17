"""Data-Driven Study Syllabus and Quiz Generator.

Modules:
    config.py          - settings loaded from .env, with validation
    logging_config.py  - writes all logs to system.log
    ingestion.py       - parse PDF/PPTX, chunk text, store embeddings in ChromaDB (RAG phase 1)
    llm.py             - chat-model factory and API error handling
    scheduler.py       - agentic plan -> check -> revise loop producing a JSON schedule
    quiz.py            - RAG quiz generation with verified source citations (RAG phase 2)
"""
