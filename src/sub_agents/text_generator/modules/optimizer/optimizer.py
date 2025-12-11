import re
import logging
from collections import Counter
from spellchecker import SpellChecker
from textstat import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ..generator.content_generator import Generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Optimizer:
    def __init__(self):
        self.spell = SpellChecker()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.benchmark_score = 60
        self.generator = Generator()

    def llm_adjust_tone(self, text, tone):
        if tone.lower() == "neutral":
            return text
        prompt = (
            f"Please edit the following text to have a more professional and '{tone}' tone. "
            "It is critical that you keep all specific keywords, topics, and the core message exactly the same. "
            "Only modify the phrasing and sentence structure to change the tone. "
            "Preserve the original paragraph formatting and hashtags.\n\n"
            f"Original Text:\n---\n{text}\n---\n\nEdited Text:"
        )
        rewritten_text = self.generator.generate(prompt)
        return rewritten_text if rewritten_text and rewritten_text.strip() else text

    def preserve_paragraphs(self, text):
        if not text or not isinstance(text, str):
            return ""
        paragraphs = text.split('\n\n')
        optimized_paragraphs = [p for p in paragraphs if p.strip()]
        return "\n\n".join(optimized_paragraphs)

    def optimize(self, text, original_topic, content_type="blog_article", tone="neutral", keywords=None):
        logger.info(f"Starting optimization for content type: {content_type}...")

        optimized_text = self.preserve_paragraphs(text)
        optimized_text = self.llm_adjust_tone(optimized_text, tone)

        # --- Content type categories ---
        long_form_types = ["blog_article", "script", "news_article"]
        social_media_types = ["social_post", "tweet", "short_form_social"]
        direct_response_types = ["email_copy", "video_description"]
        utility_types = ["faq_section"]

        # --- Initialize metrics ---
        readability_notes = "**Not Applicable** (Readability assessment is not conducted for this content type)"
        conciseness_notes = "**Not Applicable** (Conciseness evaluation is not prioritized for this content type)"

        # --- Universal Calculations ---
        sentiment = self.sentiment_analyzer.polarity_scores(optimized_text)['compound']
        sentiment_score = ((sentiment + 1) / 2) * 100

        topic_keywords = set(original_topic.lower().split())
        words = optimized_text.lower().split()
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        repetition_penalty = 0
        repeated_phrases = Counter(trigrams)
        for phrase, count in repeated_phrases.items():
            is_topic_phrase = any(keyword in phrase for keyword in topic_keywords)
            if not is_topic_phrase and count > 2:
                repetition_penalty += (count - 2) * 3

        final_score = 0

        # --- Conditional Calculations ---
        if content_type in long_form_types:
            readability = textstat.flesch_reading_ease(optimized_text)
            readability_score = max(0, min(100, readability))
            final_score = (readability_score * 0.6) + (sentiment_score * 0.4) - repetition_penalty
            readability_notes = (
                f"**{readability:.2f}** \n"
                "(Flesch Reading Ease Score, range: 0–100, where higher values indicate greater readability. "
                "Based on the Flesch formula, standardized by the U.S. Department of Education: "
                "90–100 = Very Easy (5th grade level), 60–70 = Standard (suitable for general audiences), "
                "0–30 = Very Difficult (academic or technical content))"
            )
            conciseness_notes = (
                "**Not Applicable** \n"
                "(For long-form content such as blog articles, scripts, and news articles, emphasis is placed on comprehensive coverage and detail, "
                "aligning with best practices for in-depth communication as outlined by content marketing standards.)"
            )

        elif content_type in social_media_types:
            target_length = 280 if content_type == "tweet" else 600
            conciseness_score = max(0, 100 - (len(optimized_text) / (target_length / 10)))
            final_score = (sentiment_score * 0.7) + (conciseness_score * 0.3) - repetition_penalty
            conciseness_notes = (
                f"**{conciseness_score:.0f}/100** \n"
                "(Conciseness metric evaluates adherence to platform-specific character limits: 280 characters for tweets, "
                "600 characters for other social posts, per Twitter and social media marketing guidelines. Higher scores reflect concise, impactful content.)"
            )

        elif content_type in direct_response_types:
            readability = textstat.flesch_reading_ease(optimized_text)
            readability_score = max(0, min(100, readability))
            final_score = (readability_score * 0.5) + (sentiment_score * 0.5) - repetition_penalty
            readability_notes = (
                f"**{readability:.2f}** \n"
                "(Flesch Reading Ease Score, range: 0–100, where higher values indicate greater readability. "
                "Based on the Flesch formula, standardized by the U.S. Department of Education: "
                "90–100 = Very Easy (5th grade level), 60–70 = Standard (suitable for general audiences), "
                "0–30 = Very Difficult (academic or technical content). Optimal for direct response content is 60–70.)"
            )
            conciseness_notes = (
                "**Not Applicable** \n"
                "(Direct response content, such as email copy and video descriptions, prioritizes persuasive clarity and engagement, "
                "consistent with marketing principles from the Direct Marketing Association.)"
            )

        elif content_type in utility_types:
            readability = textstat.flesch_reading_ease(optimized_text)
            readability_score = max(0, min(100, readability))
            final_score = (readability_score * 0.8) + (sentiment_score * 0.2) - (repetition_penalty / 2)
            readability_notes = (
                f"**{readability:.2f}** \n"
                "(Flesch Reading Ease Score, range: 0–100, with a target of 70–90 for FAQs to ensure accessibility. "
                "Based on the Flesch formula, standardized by the U.S. Department of Education: "
                "90–100 = Very Easy (5th grade level), 60–70 = Standard, 0–30 = Very Difficult.)"
            )
            conciseness_notes = (
                "**Not Applicable** \n"
                "(FAQ sections prioritize clear, concise answers tailored to user queries, with length varying by complexity, "
                "aligning with usability standards from the Nielsen Norman Group.)"
            )

        final_score = max(0, min(100, final_score))

        # --- Construct Analysis Notes ---
        score_report = (
            f"---\n"
            f"**Optimization Analysis Report:**\n"
            f"- **Readability:** {readability_notes}\n"
            f"- **Conciseness:** {conciseness_notes}\n"
            f"- **Sentiment Score:** {sentiment:.2f} \n"
            "(Compound sentiment score from VADER, ranging from -1 (highly negative) to 1 (highly positive), "
            "a widely accepted metric in natural language processing for assessing emotional tone.)\n"
            f"- **Repetition Penalty:** {repetition_penalty} \n"
            "(Penalty applied for non-topic-related phrase repetition, calculated as (count - 2) * 3 per occurrence, "
            "promoting originality and clarity in line with content quality standards.)\n\n"
            f"**Final Optimization Score:** {final_score:.0f}/100\n"
        )
        if final_score < self.benchmark_score:
            score_report += "(Score falls below the established benchmark of 60, suggesting a need for manual review to ensure quality and effectiveness.)\n"

        notes = score_report
        logger.info(f"Optimization complete. Final Score: {final_score:.0f}")

        return optimized_text, final_score, notes