import os
from typing import List, Dict

class Retriever:
    """
    Simple file-based retriever:
      - reads text files from assets/docs (or injected docs_dir)
      - scores docs by keyword counts and returns top_k texts
    """
    def __init__(self, docs_dir: str = None):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        default_docs = os.path.join(project_root, "assets", "docs")
        self.docs_dir = docs_dir or default_docs
        os.makedirs(self.docs_dir, exist_ok=True)

    def _load_documents(self) -> List[Dict[str, str]]:
        docs = []
        for fn in sorted(os.listdir(self.docs_dir)):
            path = os.path.join(self.docs_dir, fn)
            if os.path.isfile(path) and fn.lower().endswith((".txt", ".md")):
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        text = fh.read()
                    docs.append({"id": fn, "text": text})
                except Exception:
                    continue
        return docs

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if not query or not query.strip():
            return []
        tokens = [t.lower() for t in query.split() if t.strip()]
        docs = self._load_documents()
        scored = []
        for d in docs:
            text_low = d["text"].lower()
            score = sum(text_low.count(tok) for tok in tokens)
            if score > 0:
                scored.append((score, d["text"], d["id"]))
        scored.sort(key=lambda x: (-x[0], x[2]))
        return [text for score, text, _ in scored[:top_k]]
