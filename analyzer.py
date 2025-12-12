from __future__ import annotations

from typing import Dict, Any, List
import re

try:
    from textstat import textstat
except Exception:
    textstat = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None

try:
    from spellchecker import SpellChecker
except Exception:
    SpellChecker = None


class Analyzer:
    """Simple Analyzer module for text quality and metadata extraction.

    This starter implementation provides:
    - Readability (Flesch reading ease / grade level) using `textstat` if available
    - Sentiment scores using VADER if available
    - Basic keyword extraction (frequency-based)
    - Simple spelling checks using `pyspellchecker` if available

    The Analyzer is intentionally lightweight and defensive: it works even if
    optional dependencies are not installed.
    """

    def __init__(self):
        self._sentiment = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
        self._spell = SpellChecker() if SpellChecker else None

    def analyze(self, text: str) -> Dict[str, Any]:
        """Run analysis on `text` and return a dictionary of results.

        Returns keys:
        - `readability`: dict with `flesch_reading_ease` and `flesch_kincaid_grade`
        - `sentiment`: dict of VADER scores (pos/neu/neg/compound) or None
        - `keywords`: list of top keywords (strings)
        - `spelling`: dict with `misspell_count` and `misspell_examples`
        - `stats`: word_count, sentence_count, avg_word_length
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        clean = text.strip()
        words = re.findall(r"\w+", clean)
        word_count = len(words)
        sentences = re.split(r'[.!?]+', clean)
        sentence_count = len([s for s in sentences if s.strip()]) or 1

        # Readability
        readability: Dict[str, Any] = {}
        if textstat:
            try:
                readability = {
                    "flesch_reading_ease": textstat.flesch_reading_ease(clean),
                    "flesch_kincaid_grade": textstat.flesch_kincaid_grade(clean),
                }
            except Exception:
                readability = {}
        else:
            readability = {"note": "textstat not installed"}

        # Sentiment
        sentiment = None
        if self._sentiment:
            try:
                sentiment = self._sentiment.polarity_scores(clean)
            except Exception:
                sentiment = None

        # Simple keyword extraction: frequency of non-stop words
        tokens = [w.lower() for w in words if len(w) > 2]
        freq: Dict[str, int] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0) + 1
        # sort by frequency
        keywords = [k for k, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))][:10]

        # Spelling
        spelling = {"note": "pyspellchecker not installed"}
        if self._spell:
            try:
                miss = list(self._spell.unknown(tokens))
                spelling = {"misspell_count": len(miss), "misspell_examples": miss[:10]}
            except Exception:
                spelling = {"note": "spell check failed"}

        stats = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": (sum(len(w) for w in words) / word_count) if word_count else 0,
        }

        return {
            "readability": readability,
            "sentiment": sentiment,
            "keywords": keywords,
            "spelling": spelling,
            "stats": stats,
        }

    # Backwards-compatible small helper for tests
    def analyze_input(self, text: str) -> str:
        res = self.analyze(text)
        return f"Analyzed: words={res['stats']['word_count']}"
