# src/sub_agents/text_generator/modules/content_retrieval/llamaindex_retriever.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


SUPPORTED_EXTS = {".txt", ".md", ".html", ".htm"}


@dataclass
class RetrievedChunk:
    source: str
    score: float
    snippet: str


class Retriever:
    """
    Local-document retriever.

    Compatible with app.py:
      Retriever().retrieve(input_text) -> STRING context

    Optional:
      retrieve_with_metadata(...) -> dict with chunks
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def retrieve(
        self,
        input_text: str,
        *,
        query: Optional[str] = None,
        top_k: int = 4,
        max_chars_per_snippet: int = 450,
    ) -> str:
        result = self.retrieve_with_metadata(
            input_text,
            query=query,
            top_k=top_k,
            max_chars_per_snippet=max_chars_per_snippet,
        )
        return result["context"]

    def retrieve_with_metadata(
        self,
        input_text: str,
        *,
        query: Optional[str] = None,
        top_k: int = 4,
        max_chars_per_snippet: int = 450,
    ) -> Dict[str, Any]:
        q = (query or input_text or "").strip()
        if not q:
            return {"query": "", "context": "", "chunks": []}

        docs = self._load_documents(self.data_dir)
        if not docs:
            return {
                "query": q,
                "context": (
                    "No local documents were found for retrieval.\n"
                    f"Tip: create a '{self.data_dir}/' folder and add .txt/.md files.\n"
                ),
                "chunks": [],
            }

        ranked = self._rank_documents(q, docs)
        ranked = ranked[: max(1, int(top_k))]

        chunks: List[RetrievedChunk] = []
        parts: List[str] = []

        for source, score, text in ranked:
            snippet = self._make_snippet(q, text, max_chars=max_chars_per_snippet)
            chunks.append(RetrievedChunk(source=source, score=float(score), snippet=snippet))
            parts.append(f"[Source: {source} | relevance={float(score):.3f}]\n{snippet}")

        context = "\n\n---\n\n".join(parts).strip()
        return {"query": q, "context": context, "chunks": [asdict(c) for c in chunks]}

    # ----------------- loaders -----------------

    def _load_documents(self, data_dir: str) -> List[Tuple[str, str]]:
        if not os.path.isdir(data_dir):
            return []

        out: List[Tuple[str, str]] = []
        for root, _, files in os.walk(data_dir):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in SUPPORTED_EXTS:
                    continue
                path = os.path.join(root, fn)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        txt = f.read()
                    txt = self._clean_text(txt)
                    if txt:
                        rel = os.path.relpath(path, data_dir)
                        out.append((rel, txt))
                except Exception:
                    continue
        return out

    # ----------------- ranking -----------------

    def _rank_documents(self, query: str, docs: List[Tuple[str, str]]) -> List[Tuple[str, float, str]]:
        """
        Tries TF-IDF cosine similarity (sklearn). If sklearn isn't installed, falls back
        to keyword overlap scoring.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            sources = [s for s, _ in docs]
            texts = [t for _, t in docs]

            vec = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=5000,
                ngram_range=(1, 2),
            )
            doc_m = vec.fit_transform(texts)
            q_v = vec.transform([query])
            sims = cosine_similarity(q_v, doc_m).flatten()

            ranked = sorted(
                [(sources[i], float(sims[i]), texts[i]) for i in range(len(sources))],
                key=lambda x: x[1],
                reverse=True,
            )

            filtered = [r for r in ranked if r[1] > 0.01]
            return filtered if filtered else ranked[:1]

        except Exception:
            # Fallback: keyword overlap (no external deps)
            q_terms = set(self._terms(query))
            scored: List[Tuple[str, float, str]] = []

            for source, text in docs:
                t_terms = set(self._terms(text))
                overlap = len(q_terms & t_terms)
                denom = max(1, len(q_terms))
                score = overlap / denom
                scored.append((source, float(score), text))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[: max(1, len(scored))]

    # ----------------- text utils -----------------

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
    
    def _terms(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-ZÀ-ÿ0-9][a-zA-ZÀ-ÿ0-9\-']+", (text or "").lower())

    def _make_snippet(self, query: str, text: str, *, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text

        q_terms = set(self._terms(query))
        if not q_terms:
            return text[:max_chars].strip()

        sentences = re.split(r"(?<=[.!?])\s+", text)
        best_i, best_score = 0, -1

        for i, sent in enumerate(sentences):
            s_terms = set(self._terms(sent))
            score = len(q_terms & s_terms)
            if score > best_score:
                best_score = score
                best_i = i

        snippet = sentences[best_i]
        j = best_i - 1
        while j >= 0 and len(snippet) < max_chars * 0.8:
            snippet = sentences[j] + " " + snippet
            j -= 1
        k = best_i + 1
        while k < len(sentences) and len(snippet) < max_chars:
            snippet = snippet + " " + sentences[k]
            k += 1

        return snippet[:max_chars].strip()
