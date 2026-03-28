from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "to", "of", "in", "on", "for", "with", "as", "at", "by", "from", "into",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that",
    "these", "those", "we", "you", "they", "i", "our", "your", "their",
    "can", "could", "should", "would", "may", "might", "must", "will", "do",
    "does", "did", "not", "no", "yes", "than", "too", "very", "more", "most",
}

_GERMAN_STOPWORDS = {
    "der", "die", "das", "und", "oder", "aber", "wenn", "dann", "sonst",
    "zu", "von", "im", "in", "am", "auf", "für", "mit", "als", "bei", "aus",
    "ist", "sind", "war", "waren", "sein", "dies", "diese", "dieser",
    "wir", "ihr", "sie", "ich", "unser", "euer", "ihren",
    "kann", "könnte", "sollte", "würde", "muss", "werden", "tun",
    "nicht", "nein", "ja", "sehr", "mehr", "meisten", "ein", "eine", "einer",
}

# Map UI labels (as seen in app.py) to internal types used by optimizer/templates
_CONTENT_TYPE_MAP = {
    "blog article": "blog_article",
    "news article": "news_article",
    "email copy": "email_copy",
    "linkedin & facebook post": "social_post",
    "tiktok caption": "short_form_social",
    "youtube video description": "video_description",
    "twitter tweet": "tweet",
    "webinar script": "script",
    "podcast transcript": "script",
    "faq section": "faq_section",

    # also accept internal keys directly
    "blog_article": "blog_article",
    "news_article": "news_article",
    "email_copy": "email_copy",
    "social_post": "social_post",
    "short_form_social": "short_form_social",
    "video_description": "video_description",
    "tweet": "tweet",
    "script": "script",
    "faq_section": "faq_section",
}


@dataclass
class AnalysisPayload:
    topic: str
    content_type: str
    tone: str
    language: str
    keywords: List[str]
    needs_retrieval: bool
    retrieval_query: str
    enhanced_prompt: str
    constraints: Dict[str, Any]


class Analyzer:
    """
    Analyzer module.

    Backward-compatible:
      analyze_input(input_text) -> STRING enhanced prompt

    Optional structured usage:
      analyze(topic=..., content_type=..., tone=..., language=...) -> dict
    """

    def analyze_input(self, input_text: str) -> str:
        payload = self.analyze(topic=input_text)
        return payload["enhanced_prompt"]

    def analyze(
        self,
        *,
        topic: str,
        content_type: str = "blog_article",
        tone: str = "neutral",
        keywords: Optional[List[str]] = None,
        max_words: Optional[int] = None,
        platform: Optional[str] = None,
        needs_retrieval: Optional[bool] = None,
        language: str = "en",
    ) -> Dict[str, Any]:
        cleaned_topic = self._clean(topic)
        norm_type = self._normalize_content_type(content_type)
        norm_tone = (tone or "neutral").strip().lower() or "neutral"
        norm_language = (language or "en").strip().lower()
        if norm_language not in {"en", "de"}:
            norm_language = "en"

        if not cleaned_topic:
            payload = AnalysisPayload(
                topic="",
                content_type=norm_type,
                tone=norm_tone,
                language=norm_language,
                keywords=[],
                needs_retrieval=False,
                retrieval_query="",
                enhanced_prompt="ERROR: Empty topic provided. Please enter a valid topic.",
                constraints={},
            )
            return asdict(payload)

        kw = (
            self._normalize_keywords(keywords)
            if keywords
            else self._extract_keywords(cleaned_topic, language=norm_language)
        )

        # Safe default: retrieval helps most for blog/news
        inferred_retrieval = norm_type in {"blog_article", "news_article"}
        do_retrieval = inferred_retrieval if needs_retrieval is None else bool(needs_retrieval)

        retrieval_query = (
            self._build_retrieval_query(
                cleaned_topic,
                norm_type,
                kw,
                language=norm_language,
            )
            if do_retrieval
            else ""
        )

        constraints: Dict[str, Any] = {}
        if max_words is not None:
            constraints["max_words"] = int(max_words)
        if platform:
            constraints["platform"] = platform

        enhanced_prompt = self._build_prompt(
            topic=cleaned_topic,
            content_type=norm_type,
            tone=norm_tone,
            keywords=kw,
            max_words=max_words,
            platform=platform,
            language=norm_language,
        )

        payload = AnalysisPayload(
            topic=cleaned_topic,
            content_type=norm_type,
            tone=norm_tone,
            language=norm_language,
            keywords=kw,
            needs_retrieval=do_retrieval,
            retrieval_query=retrieval_query,
            enhanced_prompt=enhanced_prompt,
            constraints=constraints,
        )
        return asdict(payload)

    # ----------------- helpers -----------------

    def _clean(self, text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _normalize_content_type(self, ct: str) -> str:
        raw = (ct or "blog_article").strip().lower()
        return _CONTENT_TYPE_MAP.get(raw, raw if raw else "blog_article")

    def _normalize_keywords(self, keywords: Optional[List[str]]) -> List[str]:
        if not keywords:
            return []
        out: List[str] = []
        for k in keywords:
            k = re.sub(r"\s+", " ", (k or "").strip().lower())
            if k and k not in out:
                out.append(k)
        return out

    def _extract_keywords(self, topic: str, language: str = "en", k: int = 8) -> List[str]:
        tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9][a-zA-ZÀ-ÿ0-9\-']+", topic.lower())

        stopwords = _GERMAN_STOPWORDS if language == "de" else _STOPWORDS
        tokens = [t for t in tokens if t not in stopwords and len(t) >= 3]

        seen = set()
        uniq: List[str] = []
        for t in tokens:
            if t not in seen:
                uniq.append(t)
                seen.add(t)

        # add bigrams for better SEO intent
        bigrams: List[str] = []
        for i in range(len(tokens) - 1):
            bigrams.append(f"{tokens[i]} {tokens[i + 1]}")

        merged: List[str] = []
        for item in bigrams + uniq:
            if item not in merged:
                merged.append(item)

        return merged[:k]

    def _build_retrieval_query(
        self,
        topic: str,
        content_type: str,
        keywords: List[str],
        language: str = "en",
    ) -> str:
        kw_part = ", ".join(keywords[:6])

        if language == "de":
            if content_type == "news_article":
                return (
                    f"Wichtige Fakten, aktueller Kontext und zentrale Punkte zu: "
                    f"{topic}. Schlüsselwörter: {kw_part}"
                )
            if content_type == "blog_article":
                return (
                    f"Fundierte Erklärungen, Beispiele und Kernpunkte zu: "
                    f"{topic}. Schlüsselwörter: {kw_part}"
                )
            return f"Hintergrundinformationen zu: {topic}. Schlüsselwörter: {kw_part}"

        if content_type == "news_article":
            return f"key facts, recent context, important points about: {topic}. keywords: {kw_part}"
        if content_type == "blog_article":
            return f"authoritative explanations, examples, and key points about: {topic}. keywords: {kw_part}"
        return f"background information about: {topic}. keywords: {kw_part}"

    def _build_prompt(
        self,
        *,
        topic: str,
        content_type: str,
        tone: str,
        keywords: List[str],
        max_words: Optional[int],
        platform: Optional[str],
        language: str = "en",
    ) -> str:
        if language == "de":
            return self._build_prompt_german(
                topic=topic,
                content_type=content_type,
                tone=tone,
                keywords=keywords,
                max_words=max_words,
                platform=platform,
            )

        return self._build_prompt_english(
            topic=topic,
            content_type=content_type,
            tone=tone,
            keywords=keywords,
            max_words=max_words,
            platform=platform,
        )

    def _build_prompt_german(
        self,
        *,
        topic: str,
        content_type: str,
        tone: str,
        keywords: List[str],
        max_words: Optional[int],
        platform: Optional[str],
    ) -> str:
        kw_line = ", ".join(keywords) if keywords else "Nicht angegeben"
        length_line = (
            f"Maximale Länge: {max_words} Wörter."
            if max_words
            else "Länge: passend für den Inhaltstyp."
        )
        platform_line = f" Zielplattform: {platform}." if platform else ""

        if content_type == "tweet":
            return (
                f"Du bist ein erfahrener Content-Texter.\n"
                f"Schreibe EINEN Tweet über: '{topic}'.\n"
                f"Ton: {tone}.{platform_line}\n"
                f"Harte Begrenzung: maximal 280 Zeichen.\n"
                f"Optional: 1–2 relevante Hashtags.\n"
                f"Integriere diese Schlüsselwörter natürlich: {kw_line}\n"
                f"Schreibe vollständig auf Deutsch.\n"
            )

        if content_type in {"social_post", "short_form_social"}:
            return (
                f"Du bist ein erfahrener Content-Texter.\n"
                f"Schreibe einen Social-Media-Beitrag über: '{topic}'.\n"
                f"Ton: {tone}.{platform_line}\n"
                f"{length_line}\n"
                f"Struktur:\n"
                f"- Starker Einstieg in der ersten Zeile\n"
                f"- 2–5 kurze Zeilen oder Stichpunkte\n"
                f"- Call-to-Action\n"
                f"- Hashtags am Ende\n"
                f"Schlüsselwörter: {kw_line}\n"
                f"Schreibe vollständig auf Deutsch.\n"
            )

        if content_type == "email_copy":
            return (
                f"Du bist ein erfahrener Content-Texter.\n"
                f"Schreibe eine E-Mail über: '{topic}'.\n"
                f"Ton: {tone}.{platform_line}\n"
                f"{length_line}\n"
                f"Enthalten sein sollen: Betreff, Begrüßung, Hauptteil, Call-to-Action, Grußformel.\n"
                f"Schlüsselwörter: {kw_line}\n"
                f"Schreibe vollständig auf Deutsch.\n"
            )

        if content_type == "faq_section":
            return (
                f"Du bist ein erfahrener Content-Texter.\n"
                f"Erstelle einen FAQ-Bereich über: '{topic}'.\n"
                f"Ton: {tone}.{platform_line}\n"
                f"Erstelle 6–10 Fragen mit Antworten.\n"
                f"Schlüsselwörter: {kw_line}\n"
                f"Schreibe vollständig auf Deutsch.\n"
            )

        if content_type == "news_article":
            return (
                f"Du bist ein erfahrener Content-Texter.\n"
                f"Schreibe einen nachrichtenartigen Artikel über: '{topic}'.\n"
                f"Ton: {tone}.{platform_line}\n"
                f"{length_line}\n"
                f"Enthalten sein sollen: Überschrift, Einleitung, wichtige Fakten, Kontext, Schlussabschnitt.\n"
                f"Erfinde keine Fakten; wenn etwas unklar ist, kennzeichne es deutlich.\n"
                f"Schlüsselwörter: {kw_line}\n"
                f"Schreibe vollständig auf Deutsch.\n"
            )

        if content_type in {"script", "video_description"}:
            return (
                f"Du bist ein erfahrener Content-Texter.\n"
                f"Schreibe einen klaren und gut strukturierten Text über: '{topic}'.\n"
                f"Ton: {tone}.{platform_line}\n"
                f"{length_line}\n"
                f"Enthalten sein sollen: starker Einstieg, Hauptpunkte, klarer Abschluss, Call-to-Action falls passend.\n"
                f"Schlüsselwörter: {kw_line}\n"
                f"Schreibe vollständig auf Deutsch.\n"
            )

        # default: blog_article and fallback
        return (
            f"Du bist ein erfahrener Content-Texter.\n"
            f"Schreibe einen ausführlichen Blogartikel über: '{topic}'.\n"
            f"Ton: {tone}.{platform_line}\n"
            f"{length_line}\n"
            f"Enthalten sein sollen: Einleitung, Überschriften/Unterüberschriften, Beispiele, Schluss + Call-to-Action.\n"
            f"Erfinde keine Statistiken.\n"
            f"Schlüsselwörter: {kw_line}\n"
            f"Schreibe vollständig auf Deutsch.\n"
        )

    def _build_prompt_english(
        self,
        *,
        topic: str,
        content_type: str,
        tone: str,
        keywords: List[str],
        max_words: Optional[int],
        platform: Optional[str],
    ) -> str:
        kw_line = ", ".join(keywords) if keywords else "N/A"
        length_line = (
            f"Maximum length: {max_words} words."
            if max_words
            else "Length: appropriate for the content type."
        )
        platform_line = f" Target platform: {platform}." if platform else ""

        if content_type == "tweet":
            return (
                f"You are an expert content writer.\n"
                f"Write ONE tweet about: '{topic}'.\n"
                f"Tone: {tone}.{platform_line}\n"
                f"Hard limit: maximum 280 characters.\n"
                f"Optional: include 1-2 relevant hashtags.\n"
                f"Naturally include these keywords: {kw_line}\n"
                f"Write fully in English.\n"
            )

        if content_type in {"social_post", "short_form_social"}:
            return (
                f"You are an expert content writer.\n"
                f"Write a social media post about: '{topic}'.\n"
                f"Tone: {tone}.{platform_line}\n"
                f"{length_line}\n"
                f"Structure:\n"
                f"- Strong opening line\n"
                f"- 2-5 short lines or bullet-style segments\n"
                f"- Call-to-action\n"
                f"- Hashtags at the end\n"
                f"Keywords: {kw_line}\n"
                f"Write fully in English.\n"
            )

        if content_type == "email_copy":
            return (
                f"You are an expert content writer.\n"
                f"Write an email about: '{topic}'.\n"
                f"Tone: {tone}.{platform_line}\n"
                f"{length_line}\n"
                f"Include: subject line, greeting, body, call-to-action, and sign-off.\n"
                f"Keywords: {kw_line}\n"
                f"Write fully in English.\n"
            )

        if content_type == "faq_section":
            return (
                f"You are an expert content writer.\n"
                f"Create an FAQ section about: '{topic}'.\n"
                f"Tone: {tone}.{platform_line}\n"
                f"Create 6-10 question-answer pairs.\n"
                f"Keywords: {kw_line}\n"
                f"Write fully in English.\n"
            )

        if content_type == "news_article":
            return (
                f"You are an expert content writer.\n"
                f"Write a news-style article about: '{topic}'.\n"
                f"Tone: {tone}.{platform_line}\n"
                f"{length_line}\n"
                f"Include: headline, lead paragraph, key facts, context, and closing section.\n"
                f"Do not invent facts; clearly signal uncertainty where needed.\n"
                f"Keywords: {kw_line}\n"
                f"Write fully in English.\n"
            )

        if content_type in {"script", "video_description"}:
            return (
                f"You are an expert content writer.\n"
                f"Write a clear and well-structured piece about: '{topic}'.\n"
                f"Tone: {tone}.{platform_line}\n"
                f"{length_line}\n"
                f"Include: strong opening, core points, clear ending, and a call-to-action where appropriate.\n"
                f"Keywords: {kw_line}\n"
                f"Write fully in English.\n"
            )

        # default: blog_article and fallback
        return (
            f"You are an expert content writer.\n"
            f"Write a detailed blog article about: '{topic}'.\n"
            f"Tone: {tone}.{platform_line}\n"
            f"{length_line}\n"
            f"Include: introduction, headings/subheadings, examples, conclusion, and a call-to-action.\n"
            f"Do not invent statistics.\n"
            f"Keywords: {kw_line}\n"
            f"Write fully in English.\n"
        )