from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END

from src.sub_agents.text_generator.modules.analyzer.analyzer import Analyzer
from src.sub_agents.text_generator.modules.content_retrieval.llamaindex_retriever import Retriever
from src.sub_agents.text_generator.modules.generator.content_generator import Generator
from src.sub_agents.text_generator.modules.optimizer.optimizer import Optimizer
from src.sub_agents.text_generator.modules.human_review.reviewer import Reviewer
from src.sub_agents.text_generator.modules.exporter.exporter import Exporter
from src.shared.text_run_logger import TextRunLogger


@dataclass
class TextAgentState:
    # Inputs
    model_name: str = ""
    original_topic: str = ""
    content_type: str = "blog_article"
    tone: str = "neutral"
    keywords: List[str] = field(default_factory=list)
    language: str = "en"

    # Timing
    started_at: float = 0.0
    total_time_sec: float = 0.0

    # Working artifacts
    analysis: Dict[str, Any] = field(default_factory=dict)
    retrieved_context: str = ""
    raw_prompt: str = ""
    enhanced_prompt: str = ""
    final_prompt: str = ""

    # Outputs
    generated_text: str = ""
    optimized_text: str = ""
    final_text: str = ""
    score: float = 0.0
    notes: str = ""
    review: Dict[str, Any] = field(default_factory=dict)
    export_paths: Dict[str, Any] = field(default_factory=dict)

    # Revision / optimization metrics for logging
    revision_rounds: int = 0
    initial_score: Optional[float] = None
    revised_score: Optional[float] = None
    score_improvement: Optional[float] = None

    # Optional debug / full metrics payload
    revision_metrics: Optional[Dict[str, Any]] = None


class TextAgent:
    def __init__(self, review_threshold: float = 60.0):
        self.analyzer = Analyzer()
        self.retriever = Retriever()
        self.generator = Generator()
        self.optimizer = Optimizer(benchmark_score=review_threshold)
        self.reviewer = Reviewer(threshold=review_threshold)
        self.exporter = Exporter()
        self.run_logger = TextRunLogger()

        workflow = StateGraph(TextAgentState)
        workflow.add_node("analyze_step", self._analyze)
        workflow.add_node("retrieve_step", self._retrieve)
        workflow.add_node("generate_step", self._generate)
        workflow.add_node("optimize_step", self._optimize)
        workflow.add_node("review_step", self._review)
        workflow.add_node("export_step", self._export)

        workflow.set_entry_point("analyze_step")
        workflow.add_edge("analyze_step", "retrieve_step")
        workflow.add_edge("retrieve_step", "generate_step")
        workflow.add_edge("generate_step", "optimize_step")
        workflow.add_edge("optimize_step", "review_step")
        workflow.add_edge("review_step", "export_step")
        workflow.add_edge("export_step", END)

        self.app = workflow.compile()

    def _analyze(self, state: TextAgentState) -> TextAgentState:
        state.analysis = self.analyzer.analyze(
            topic=state.original_topic,
            content_type=state.content_type,
            tone=state.tone,
            keywords=state.keywords,
            language=state.language,
        )

        state.raw_prompt = state.original_topic
        state.enhanced_prompt = (
            state.analysis.get("enhanced_prompt", "")
            if isinstance(state.analysis, dict)
            else ""
        )
        return state

    def _retrieve(self, state: TextAgentState) -> TextAgentState:
        analysis = state.analysis if isinstance(state.analysis, dict) else {}

        needs_retrieval = bool(analysis.get("needs_retrieval", True))
        retrieval_query = analysis.get("retrieval_query", "") or state.original_topic

        if not needs_retrieval:
            state.retrieved_context = ""
            return state

        state.retrieved_context = self.retriever.retrieve(
            state.original_topic,
            query=retrieval_query,
        )
        return state

    def _generate(self, state: TextAgentState) -> TextAgentState:
        context_text = (state.retrieved_context or "").strip()
        enhanced_prompt = (state.enhanced_prompt or "").strip()

        if not enhanced_prompt:
            enhanced_prompt = (
                f"Write content about: {state.original_topic}\n"
                f"Content type: {state.content_type}\n"
                f"Tone: {state.tone}\n"
                f"Language: {state.language}\n"
            )

        final_generation_prompt = enhanced_prompt

        if context_text:
            if (state.language or "en").strip().lower() == "de":
                context_instruction = (
                    "Nutze den folgenden abgerufenen Kontext, wenn er relevant ist. "
                    "Verwende ihn nur, wenn er hilfreich ist, und erfinde keine Fakten."
                )
            else:
                context_instruction = (
                    "Use the following retrieved context where relevant. "
                    "Use it only if it is helpful and do not invent facts."
                )

            final_generation_prompt += (
                f"\n\n{context_instruction}\n\n{context_text}\n"
            )

        state.final_prompt = final_generation_prompt

        state.generated_text = self.generator.generate(
            final_generation_prompt,
            model_name=state.model_name,
        )
        return state

    def _optimize(self, state: TextAgentState) -> TextAgentState:
        optimized_text, final_score, notes, revision_metrics = self.optimizer.optimize(
            text=state.generated_text,
            original_topic=state.original_topic,
            content_type=state.content_type,
            tone=state.tone,
            keywords=state.keywords,
            language=state.language,
            auto_revise=True,
            revision_rounds=0,
            max_revision_rounds=1,
        )

        state.optimized_text = optimized_text
        state.final_text = optimized_text
        state.score = final_score
        state.notes = notes
        state.revision_metrics = revision_metrics

        rm = revision_metrics if isinstance(revision_metrics, dict) else {}
        state.revision_rounds = int(rm.get("revision_rounds", 0) or 0)
        state.initial_score = rm.get("initial_score")
        state.revised_score = rm.get("revised_score")
        state.score_improvement = rm.get("score_improvement")

        return state

    def _review(self, state: TextAgentState) -> TextAgentState:
        state.review = self.reviewer.review(
            content=state.optimized_text,
            score=state.score,
            notes=state.notes,
        )
        state.final_text = state.optimized_text
        return state

    def _export(self, state: TextAgentState) -> TextAgentState:
        analysis = state.analysis if isinstance(state.analysis, dict) else {}

        # Calculate actual elapsed runtime here, just before export + logging
        if state.started_at > 0:
            state.total_time_sec = round(time.perf_counter() - state.started_at, 4)
        else:
            state.total_time_sec = 0.0

        state.export_paths = self.exporter.export_content(
            content=state.optimized_text,
            content_type=state.content_type,
            topic=state.original_topic,
            tone=state.tone,
            keywords=state.keywords,
            score=state.score,
            notes=state.notes,
            review_result=state.review,
            formats=("txt", "md", "html", "json"),
            require_approved=True,
            model_name=state.model_name,
            prompt_id="text_generator_v2",
            language=state.language,
            enhanced_prompt=analysis.get("enhanced_prompt", ""),
            retrieval_query=analysis.get("retrieval_query", ""),
            raw_prompt=state.raw_prompt,
            final_prompt=state.final_prompt,
        )

        # Shared-service logging only
        self.run_logger.log_run(
            language=state.language,
            raw_prompt=state.raw_prompt,
            enhanced_prompt=state.enhanced_prompt,
            final_prompt=state.final_prompt,
            model_name=state.model_name,
            content_type=state.content_type,
            tone=state.tone,
            total_time_sec=state.total_time_sec,
            optimized_score=state.score,
            revision_rounds=state.revision_rounds,
            initial_score=state.initial_score,
            revised_score=state.revised_score,
            score_improvement=state.score_improvement,
            final_text=state.final_text,
        )

        return state

    def run(
        self,
        *,
        topic: str,
        content_type: str,
        tone: str = "neutral",
        keywords: Optional[List[str]] = None,
        model_name: str = "",
        language: str = "en",
    ) -> Dict[str, Any]:
        provider = (os.getenv("TEXT_PROVIDER") or "openai").strip().lower()

        mn = (model_name or "").strip()
        if not mn:
            if provider == "groq":
                mn = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
            else:
                mn = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o")

        state = TextAgentState(
            original_topic=topic,
            content_type=content_type,
            tone=tone,
            keywords=keywords or [],
            model_name=mn,
            language=(language or "en").strip().lower(),
            started_at=time.perf_counter(),
        )

        final_state = self.app.invoke(state)

        def pick(obj: Any, key: str, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        return {
            "topic": pick(final_state, "original_topic", topic),
            "content_type": pick(final_state, "content_type", content_type),
            "tone": pick(final_state, "tone", tone),
            "keywords": pick(final_state, "keywords", keywords or []),
            "language": pick(final_state, "language", language),
            "model_name": pick(final_state, "model_name", mn),
            "analysis": pick(final_state, "analysis", {}),
            "retrieved_context": pick(final_state, "retrieved_context", ""),
            "raw_prompt": pick(final_state, "raw_prompt", topic),
            "enhanced_prompt": pick(final_state, "enhanced_prompt", ""),
            "final_prompt": pick(final_state, "final_prompt", ""),
            "generated_text": pick(final_state, "generated_text", ""),
            "optimized_text": pick(final_state, "optimized_text", ""),
            "final_text": pick(final_state, "final_text", ""),
            "score": pick(final_state, "score", 0.0),
            "notes": pick(final_state, "notes", ""),
            "review": pick(final_state, "review", {}),
            "export_paths": pick(final_state, "export_paths", {}),
            "revision_metrics": pick(final_state, "revision_metrics", None),
            "total_time_sec": pick(final_state, "total_time_sec", 0.0),
            "revision_rounds": pick(final_state, "revision_rounds", 0),
            "initial_score": pick(final_state, "initial_score", None),
            "revised_score": pick(final_state, "revised_score", None),
            "score_improvement": pick(final_state, "score_improvement", None),
        }