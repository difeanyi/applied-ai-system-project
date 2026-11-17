from __future__ import annotations

from dataclasses import replace

from langchain_core.embeddings import DeterministicFakeEmbedding

from src.config import KNOWLEDGE_DIR
from src.rag import KnowledgeBase, load_passages


def test_every_section_becomes_one_passage():
    passages = load_passages(KNOWLEDGE_DIR)
    sections = [p.metadata["section"] for p in passages]
    assert "Mood: chill" in sections and "Genre: lofi" in sections
    assert "Activity: workout, gym, running, training" in sections
    assert len(sections) == len(set(sections))
    chill = next(p for p in passages if p.metadata["section"] == "Mood: chill")
    assert "Catalog mood label: chill" in chill.page_content and "mellow" in chill.page_content


def test_knowledge_covers_every_catalog_label():
    from src.agent import Catalog
    from src.config import SONGS_CSV

    text = " ".join(p.page_content for p in load_passages(KNOWLEDGE_DIR))
    catalog = Catalog.from_csv(SONGS_CSV)
    for genre in catalog.genres:
        assert f"Catalog genre label: {genre}" in text
    for mood in catalog.moods:
        assert f"Catalog mood label: {mood}" in text


def test_index_is_reused_and_rebuilt_when_files_change(settings, tmp_path):
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    (kdir / "a.md").write_text("# A\n\n## One\nfirst\n\n## Two\nsecond\n")
    s = replace(settings, knowledge_dir=kdir)
    kb = KnowledgeBase(s, embeddings=DeterministicFakeEmbedding(size=16))
    assert kb.ensure_index() == 2
    assert kb.ensure_index() == 2  # unchanged: reused

    (kdir / "a.md").write_text("# A\n\n## One\nchanged\n")
    assert kb.ensure_index() == 1
    assert kb.store.get()["documents"] == ["One\nchanged"]


def test_retrieval_is_logged_word_for_word(kb, log_file):
    passages = kb.retrieve("something mellow", k=2)
    assert len(passages) == 3  # 2 semantic results + 1 keyword match ("mellow" -> Mood: chill)
    log = log_file.read_text()
    assert "Retrieval for 'something mellow' (k=2): 3 passages" in log
    assert "keyword: mellow" in log
    assert all(p.text in log for p in passages)


def test_exact_synonyms_are_always_retrieved(kb):
    # The live evaluation's one failure: the embedding search missed "chillhop".
    passages = kb.retrieve("some chillhop beats", k=4)
    lofi = [p for p in passages if p.section == "Genre: lofi"]
    assert lofi and lofi[0].match == "keyword: chillhop"


def test_synonyms_match_whole_words_only(kb):
    assert kb.keyword_matches("a rapper's delight") == []  # "rap" inside another word
    assert [p.section for p in kb.keyword_matches("some rap please")] == ["Genre: hip-hop"]


def test_keyword_matches_do_not_crowd_out_semantic_results(kb):
    passages = kb.retrieve("mellow studying, nothing too sad", k=4)
    keyword = [p for p in passages if p.match.startswith("keyword")]
    assert 1 <= len(keyword) <= 3
    assert len(passages) - len(keyword) >= 1  # semantic results still present
