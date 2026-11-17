"""Data-Driven Study Syllabus and Quiz Generator.

Modules (added step by step):
    ingestion.py  - parse PDF/PPTX, chunk text, store embeddings in ChromaDB (Step 2)
    scheduler.py  - agentic LLM chain that builds a JSON study schedule (Step 3)
    quiz.py       - RAG quiz generation with source citations (Step 3)
"""
