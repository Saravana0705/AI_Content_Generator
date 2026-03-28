import logging
import re
from collections import Counter
from typing import Any, Dict, Optional, Tuple

from spellchecker import SpellChecker
from textstat import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..generator.content_generator import Generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Optimizer:
    """
    Optimizes content and computes an optimization score.

    Auto-revision behavior:
    - If final_score < benchmark_score and auto_revise=True and revision_rounds < max_revision_rounds:
        - rewrite content (LLM) and re-score
        - return revision_metrics (initial_score, revised_score, score_improvement, revision_rounds)
    - Else:
        - no revision_metrics (None)
    """

    def __init__(self, benchmark_score: float = 60.0):
        self.spell = SpellChecker()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.benchmark_score = benchmark_score
        self.generator = Generator()

    # -----------------------------
    # Utility helpers
    # -----------------------------
    def preserve_paragraphs(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def llm_adjust_tone(self, text: str, tone: str) -> str:
        """
        Light placeholder. Keeps current text unchanged unless you later
        add an explicit tone-adjustment LLM step.
        """
        if not text:
            return ""
        return text

    # -----------------------------
    # English scoring path
    # -----------------------------
    def _score_text(self, optimized_text: str, original_topic: str, content_type: str) -> Tuple[float, str]:
        """
        English-oriented scoring path.
        Uses readability/sentiment/repetition with content-type aware weighting.
        """
        optimized_text = self.preserve_paragraphs(optimized_text)

        long_form_types = {"blog_article", "script", "news_article"}
        social_media_types = {"social_post", "tweet", "short_form_social"}
        direct_response_types = {"email_copy", "video_description"}
        utility_types = {"faq_section"}

        readability_notes = "**Not Applicable**"
        conciseness_notes = "**Not Applicable**"

        sentiment = self.sentiment_analyzer.polarity_scores(optimized_text).get("compound", 0.0)
        sentiment_score = ((sentiment + 1) / 2) * 100  # map [-1,1] -> [0,100]

        # repetition penalty (trigrams)
        topic_keywords = set((original_topic or "").lower().split())
        words = (optimized_text or "").lower().split()
        trigrams = [" ".join(words[i:i + 3]) for i in range(max(0, len(words) - 2))]

        repetition_penalty = 0.0
        repeated_phrases = Counter(trigrams)
        for phrase, count in repeated_phrases.items():
            is_topic_phrase = any(k in phrase for k in topic_keywords if k)
            if (not is_topic_phrase) and count > 2:
                repetition_penalty += (count - 2) * 3

        final_score = 0.0

        if content_type in long_form_types:
            readability = textstat.flesch_reading_ease(optimized_text)
            readability_score = max(0.0, min(100.0, float(readability)))
            final_score = (readability_score * 0.6) + (sentiment_score * 0.4) - repetition_penalty
            readability_notes = f"**{readability:.2f}**"
            conciseness_notes = "**Not Applicable**"

        elif content_type in social_media_types:
            target_length = 280 if content_type == "tweet" else 600
            conciseness_score = max(0.0, 100.0 - (len(optimized_text) / (target_length / 10.0)))
            final_score = (sentiment_score * 0.7) + (conciseness_score * 0.3) - repetition_penalty
            conciseness_notes = f"**{conciseness_score:.0f}/100**"

        elif content_type in direct_response_types:
            readability = textstat.flesch_reading_ease(optimized_text)
            readability_score = max(0.0, min(100.0, float(readability)))
            final_score = (readability_score * 0.5) + (sentiment_score * 0.5) - repetition_penalty
            readability_notes = f"**{readability:.2f}**"
            conciseness_notes = "**Not Applicable**"

        elif content_type in utility_types:
            readability = textstat.flesch_reading_ease(optimized_text)
            readability_score = max(0.0, min(100.0, float(readability)))
            final_score = (readability_score * 0.8) + (sentiment_score * 0.2) - (repetition_penalty / 2.0)
            readability_notes = f"**{readability:.2f}**"
            conciseness_notes = "**Not Applicable**"

        else:
            readability = textstat.flesch_reading_ease(optimized_text)
            readability_score = max(0.0, min(100.0, float(readability)))
            final_score = (readability_score * 0.6) + (sentiment_score * 0.4) - repetition_penalty
            readability_notes = f"**{readability:.2f}**"

        final_score = max(0.0, min(100.0, float(final_score)))

        score_report = (
            f"---\n"
            f"**Optimization Analysis Report:**\n"
            f"- **Readability:** {readability_notes}\n"
            f"- **Conciseness:** {conciseness_notes}\n"
            f"- **Sentiment Score:** {sentiment:.2f}\n"
            f"- **Repetition Penalty:** {repetition_penalty:.0f}\n\n"
            f"**Final Optimization Score:** {final_score:.0f}/100\n"
        )
        return final_score, score_report

    # -----------------------------
    # German scoring path
    # -----------------------------
    def _score_text_german(
        self,
        *,
        optimized_text: str,
        original_topic: str,
        content_type: str,
        keywords: Optional[list] = None,
    ) -> Tuple[float, str]:
        """
        German-safe heuristic scoring path.
        Avoids English-specific readability/sentiment assumptions.
        """
        keywords = keywords or []
        text = self.preserve_paragraphs(optimized_text)

        words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        word_count = len(words)
        sentence_count = max(1, len(sentences))
        paragraph_count = len(paragraphs)
        avg_sentence_len = word_count / sentence_count

        # repetition penalty (trigrams)
        lowered_words = [w.lower() for w in words]
        trigrams = [" ".join(lowered_words[i:i + 3]) for i in range(max(0, len(lowered_words) - 2))]
        repeated_phrases = Counter(trigrams)

        topic_keywords = set(
            re.findall(r"\b\w+\b", (original_topic or "").lower(), flags=re.UNICODE)
        )

        repetition_penalty = 0.0
        for phrase, count in repeated_phrases.items():
            is_topic_phrase = any(k in phrase for k in topic_keywords if k)
            if (not is_topic_phrase) and count > 2:
                repetition_penalty += (count - 2) * 3

        # keyword coverage
        lowered_text = text.lower()
        keyword_hits = 0
        for kw in keywords:
            kw_clean = (kw or "").strip().lower()
            if kw_clean and kw_clean in lowered_text:
                keyword_hits += 1

        keyword_coverage = (
            (keyword_hits / max(1, len(keywords))) * 100.0
            if keywords
            else 80.0
        )

        # sentence length heuristic
        if avg_sentence_len <= 16:
            sentence_score = 100.0
        elif avg_sentence_len <= 22:
            sentence_score = 85.0
        elif avg_sentence_len <= 28:
            sentence_score = 65.0
        else:
            sentence_score = 40.0

        # paragraph structure
        if content_type in {"blog_article", "news_article", "script", "faq_section"}:
            paragraph_score = 100.0 if paragraph_count >= 3 else 60.0
        else:
            paragraph_score = 100.0 if paragraph_count >= 1 else 60.0

        # length / conciseness
        if content_type == "tweet":
            length_score = (
                100.0 if len(text) <= 280
                else max(0.0, 100.0 - (len(text) - 280) * 0.5)
            )
        elif content_type in {"social_post", "short_form_social"}:
            length_score = (
                100.0 if word_count <= 120
                else max(40.0, 100.0 - (word_count - 120) * 0.5)
            )
        elif content_type == "email_copy":
            length_score = 100.0 if 80 <= word_count <= 250 else 70.0
        elif content_type == "faq_section":
            length_score = 100.0 if word_count >= 120 else 70.0
        else:
            length_score = 100.0 if word_count >= 150 else 65.0

        final_score = (
            sentence_score * 0.30
            + paragraph_score * 0.20
            + length_score * 0.25
            + keyword_coverage * 0.25
            - repetition_penalty
        )
        final_score = max(0.0, min(100.0, final_score))

        score_report = (
            f"---\n"
            f"**German Optimization Analysis Report:**\n"
            f"- **Word Count:** {word_count}\n"
            f"- **Paragraph Count:** {paragraph_count}\n"
            f"- **Average Sentence Length:** {avg_sentence_len:.1f}\n"
            f"- **Sentence Length Score:** {sentence_score:.0f}/100\n"
            f"- **Paragraph Structure Score:** {paragraph_score:.0f}/100\n"
            f"- **Length Score:** {length_score:.0f}/100\n"
            f"- **Keyword Coverage:** {keyword_coverage:.0f}/100\n"
            f"- **Repetition Penalty:** {repetition_penalty:.0f}\n\n"
            f"**Final Optimization Score:** {final_score:.0f}/100\n"
        )
        return final_score, score_report

    # -----------------------------
    # Public optimize entrypoint
    # -----------------------------
    def optimize(
        self,
        *,
        text: str,
        original_topic: str,
        content_type: str = "blog_article",
        tone: str = "neutral",
        keywords: Optional[list] = None,
        language: str = "en",
        auto_revise: bool = True,
        revision_rounds: int = 0,
        max_revision_rounds: int = 1,
    ) -> Tuple[str, float, str, Optional[Dict[str, Any]]]:
        """
        Returns:
          optimized_text: str
          final_score: float
          notes: str
          revision_metrics: Optional[dict]
            - Only non-None when score < benchmark and auto revision was actually applied
        """
        keywords = keywords or []
        language = (language or "en").strip().lower()
        if language not in {"en", "de"}:
            language = "en"

        revision_metrics: Optional[Dict[str, Any]] = None

        # Pass 1: normalize + adjust tone
        optimized_text = self.preserve_paragraphs(text)
        optimized_text = self.llm_adjust_tone(optimized_text, tone)

        # Score pass 1
        if language == "de":
            final_score, notes = self._score_text_german(
                optimized_text=optimized_text,
                original_topic=original_topic,
                content_type=content_type,
                keywords=keywords,
            )
        else:
            final_score, notes = self._score_text(
                optimized_text,
                original_topic,
                content_type,
            )

        # Decide whether to auto-revise
        if (
            auto_revise
            and final_score < float(self.benchmark_score)
            and revision_rounds < max_revision_rounds
        ):
            logger.info(
                f"Score {final_score:.0f} < {self.benchmark_score:.0f}. "
                f"Auto-revision round {revision_rounds + 1}."
            )

            kw_text = ", ".join(keywords or []) if keywords else "None specified"

            if language == "de":
                revision_prompt = (
                    "Überarbeite den folgenden Text auf Deutsch so, dass er klarer, "
                    "einfacher und besser strukturiert ist.\n\n"
                    "Wichtige Regeln:\n"
                    "- Schreibe in einfachem, natürlichem Deutsch.\n"
                    "- Verwende eher kurze Sätze.\n"
                    "- Teile lange Absätze in kleinere Abschnitte.\n"
                    "- Vermeide unnötige Wiederholungen.\n"
                    "- Bewahre die Bedeutung des Inhalts.\n"
                    "- Behalte wichtige Schlüsselwörter bei.\n"
                    "- Erfinde keine neuen Fakten.\n\n"
                    f"Thema: {original_topic}\n"
                    f"Inhaltstyp: {content_type}\n"
                    f"Ton: {tone}\n"
                    f"Schlüsselwörter: {kw_text}\n\n"
                    "Hier ist der zu überarbeitende Text:\n"
                    "-----\n"
                    f"{optimized_text}\n"
                    "-----\n\n"
                    "Hier sind die Analysehinweise:\n"
                    "-----\n"
                    f"{notes}\n"
                    "-----\n\n"
                    "Gib nur die überarbeitete deutsche Fassung zurück, ohne Erklärungen."
                )
            else:
                revision_prompt = (
                    "Rewrite the following content in a MUCH simpler and more readable way.\n"
                    "Your goal is to dramatically increase readability.\n\n"
                    "IMPORTANT — You MUST follow these rules:\n"
                    "- Write at a 6th–8th grade reading level.\n"
                    "- Use ONLY short sentences (8–14 words each).\n"
                    "- Use simple, everyday vocabulary.\n"
                    "- Break long paragraphs into small sections.\n"
                    "- Remove all unnecessary details.\n"
                    "- Prioritize clarity over style.\n"
                    "- Do not use complex phrasing.\n"
                    "- Avoid long introductions; get to the point quickly.\n"
                    "- Aim for a Flesch Reading Ease score ABOVE 60.\n\n"
                    f"Topic: {original_topic}\n"
                    f"Content Type: {content_type}\n"
                    f"Tone: {tone}\n"
                    f"Keywords: {kw_text}\n\n"
                    "Here is the content that must be simplified:\n"
                    "-----\n"
                    f"{optimized_text}\n"
                    "-----\n\n"
                    "Here are analysis notes showing weaknesses:\n"
                    "-----\n"
                    f"{notes}\n"
                    "-----\n\n"
                    "Now rewrite the content so it is extremely easy to read.\n"
                    "Return ONLY the simplified content with no explanations."
                )

            try:
                revised_text = self.generator.generate(revision_prompt)
                revised_text = self.preserve_paragraphs(revised_text)

                if language == "de":
                    revised_score, revised_notes = self._score_text_german(
                        optimized_text=revised_text,
                        original_topic=original_topic,
                        content_type=content_type,
                        keywords=keywords,
                    )
                else:
                    revised_score, revised_notes = self._score_text(
                        revised_text,
                        original_topic,
                        content_type,
                    )

                revision_metrics = {
                    "initial_score": float(final_score),
                    "revised_score": float(revised_score),
                    "score_improvement": float(revised_score - final_score),
                    "revision_rounds": int(revision_rounds + 1),
                }

                # Keep revised text only if it improved or matched
                if revised_score >= final_score:
                    optimized_text = revised_text
                    final_score = revised_score
                    notes = revised_notes

            except Exception as e:
                logger.warning(f"Auto-revision failed: {e}")

        return optimized_text, float(final_score), notes, revision_metrics