import os
import tempfile
from sub_agents.retriever.retriever import Retriever

def test_retrieve_keyword_matches():
    with tempfile.TemporaryDirectory() as tdir:
        f1 = os.path.join(tdir, "doc1.txt")
        f2 = os.path.join(tdir, "doc2.txt")
        with open(f1, "w", encoding="utf-8") as fh:
            fh.write("AI improves healthcare and future diagnostics.")
        with open(f2, "w", encoding="utf-8") as fh:
            fh.write("Cooking recipes and food tips.")
        r = Retriever(docs_dir=tdir)
        results = r.retrieve("AI future", top_k=2)
        assert isinstance(results, list)
        assert len(results) >= 1
        got = " ".join(results).lower()
        assert "ai" in got or "future" in got

def test_retrieve_empty_query_returns_empty():
    r = Retriever(docs_dir=tempfile.gettempdir())
    assert r.retrieve("") == []
    assert r.retrieve("   ") == []
