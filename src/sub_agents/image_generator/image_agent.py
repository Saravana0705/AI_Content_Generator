from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

from langgraph.graph import StateGraph, END

from src.sub_agents.image_generator.modules.analyzer.analyzer import ImageAnalyzer
from src.sub_agents.image_generator.modules.content_retrieval.style_retriever import StyleRetriever
from src.sub_agents.image_generator.modules.generator.image_generator import ImageGenerator
from src.sub_agents.image_generator.modules.optimizer.optimizer import ImageOptimizer
from src.sub_agents.image_generator.modules.human_review.reviewer import ImageReviewer
from src.sub_agents.image_generator.modules.exporter.exporter import ImageExporter
from src.shared.image_run_logger import ImageRunLogger


@dataclass
class ImageAgentState:
    # Inputs
    original_prompt: str = ""
    style: str = ""
    size: str = "1024x1024"
    language: str = "en"

    # Analyzer output
    intent: Dict[str, Any] = field(default_factory=dict)

    # Retrieval output
    style_payload: Dict[str, Any] = field(default_factory=dict)
    enhanced_prompt: str = ""
    final_prompt: str = ""

    # Generation output
    image_bytes: Optional[bytes] = None
    image_meta: Dict[str, Any] = field(default_factory=dict)

    # Optimization / validation
    technical_ok: bool = False
    clip_score: float = 0.0
    aesthetic_score: float = 0.0
    optimizer_notes: str = ""
    retries: int = 0

    # Dynamic threshold metadata
    clip_threshold: Optional[float] = None
    clip_q: Optional[float] = None
    aesthetic_threshold: Optional[float] = None
    aesthetic_r: Optional[float] = None
    calibration_window_size: Optional[int] = None
    calibration_path: Optional[str] = None

    # Review + export
    review: Dict[str, Any] = field(default_factory=dict)
    export_paths: Dict[str, Any] = field(default_factory=dict)

    calibrated_at: Optional[str] = None
    style_default: Optional[str] = None
    size_default: Optional[str] = None


class ImageAgent:
    def __init__(
        self,
        *,
        calibration_path: str = "calibration/image_thresholds.json",
        max_retries: int = 1,
    ):
        self.analyzer = ImageAnalyzer()
        self.retriever = StyleRetriever()
        self.generator = ImageGenerator()

        self.optimizer = ImageOptimizer(
            calibration_path=calibration_path,
            max_retries=max_retries,
        )

        self.run_logger = ImageRunLogger()

        # Warm up scoring models in main thread (Windows DLL issue prevention)
        self.optimizer.warmup()

        self.reviewer = ImageReviewer()
        self.exporter = ImageExporter()

        graph = StateGraph(ImageAgentState)
        graph.add_node("analyze", self._analyze)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.add_node("optimize", self._optimize)
        graph.add_node("review_step", self._review)
        graph.add_node("export", self._export)

        graph.set_entry_point("analyze")
        graph.add_edge("analyze", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "optimize")
        graph.add_edge("optimize", "review_step")
        graph.add_edge("review_step", "export")
        graph.add_edge("export", END)

        self.graph = graph.compile()

    def _analyze(self, state: ImageAgentState) -> ImageAgentState:
        state.intent = self.analyzer.analyze(
            prompt=state.original_prompt,
            style=state.style,
            size=state.size,
            language=state.language,
        )
        return state

    def _retrieve(self, state: ImageAgentState) -> ImageAgentState:
        state.style_payload = self.retriever.retrieve(
            style=state.intent.get("style", state.style)
        )

        state.enhanced_prompt = self.retriever.build_final_prompt(
            intent=state.intent,
            style_payload=state.style_payload,
            language=state.language,
        )

        state.intent["enhanced_prompt"] = state.enhanced_prompt
        state.final_prompt = state.enhanced_prompt
        return state

    def _generate(self, state: ImageAgentState) -> ImageAgentState:
        img_bytes, meta = self.generator.generate(
            prompt=state.final_prompt,
            size=state.intent.get("size", state.size),
        )
        state.image_bytes = img_bytes
        state.image_meta = meta
        return state

    def _optimize(self, state: ImageAgentState) -> ImageAgentState:
        out = self.optimizer.validate_and_maybe_retry(
            intent=state.intent,
            prompt=state.final_prompt,
            image_bytes=state.image_bytes,
            image_meta=state.image_meta,
        )

        state.technical_ok = out["technical_ok"]
        state.clip_score = out["clip_score"]
        state.aesthetic_score = out["aesthetic_score"]
        state.optimizer_notes = out["notes"]
        state.retries = out["retries"]

        state.image_bytes = out["image_bytes"]
        state.image_meta = out["image_meta"]
        state.final_prompt = out["final_prompt"]
                
        state.clip_threshold = out.get("clip_threshold")
        state.clip_q = out.get("clip_q")
        state.aesthetic_threshold = out.get("aesthetic_threshold")
        state.aesthetic_r = out.get("aesthetic_r")
        state.calibration_window_size = out.get("calibration_window_size")
        state.calibrated_at = out.get("calibrated_at")
        state.style_default = out.get("style_default")
        state.size_default = out.get("size_default")
        state.calibration_path = out.get("calibration_path")

        return state

    def _review(self, state: ImageAgentState) -> ImageAgentState:
        state.review = self.reviewer.review(
            clip_score=state.clip_score,
            aesthetic_score=state.aesthetic_score,
            technical_ok=state.technical_ok,
            notes=state.optimizer_notes,
        )
        return state

    def _export(self, state: ImageAgentState) -> ImageAgentState:
        export_meta = {
            **(state.image_meta or {}),
            "clip_threshold": state.clip_threshold,
            "clip_q": state.clip_q,
            "aesthetic_threshold": state.aesthetic_threshold,
            "aesthetic_r": state.aesthetic_r,
            "calibration_window_size": state.calibration_window_size,
            "calibration_path": state.calibration_path,
            "calibrated_at": state.calibrated_at,
            "style_default": state.style_default,
            "size_default": state.size_default,
        }

        state.export_paths = self.exporter.export_image(
            image_bytes=state.image_bytes,
            raw_prompt=state.original_prompt,
            enhanced_prompt=state.enhanced_prompt,
            final_prompt=state.final_prompt,
            intent=state.intent,
            meta=export_meta,
            clip_score=state.clip_score,
            aesthetic_score=state.aesthetic_score,
            retries=state.retries,
            review_result=state.review,
            clip_threshold=state.clip_threshold,
            clip_q=state.clip_q,
            aesthetic_threshold=state.aesthetic_threshold,
            aesthetic_r=state.aesthetic_r,
            calibration_window_size=state.calibration_window_size,
            calibration_path=state.calibration_path,
            require_approved=False,
        )
        
        files = state.export_paths.get("files", {}) if isinstance(state.export_paths, dict) else {}

        self.run_logger.log_run(
            raw_prompt=state.original_prompt,
            enhanced_prompt=state.enhanced_prompt,
            final_prompt=state.final_prompt,
            intent=state.intent,
            clip_score=state.clip_score,
            aesthetic_score=state.aesthetic_score,
            retries=state.retries,
            approved=bool(state.review.get("approved", False)),
            technical_ok=state.review.get("technical_ok"),
            clip_threshold=state.clip_threshold,
            clip_q=state.clip_q,
            aesthetic_threshold=state.aesthetic_threshold,
            aesthetic_r=state.aesthetic_r,
            calibration_window_size=state.calibration_window_size,
            calibrated_at=state.calibrated_at,
            output_dir=state.export_paths.get("output_dir", ""),
            image_path=files.get("image", ""),
            metadata_path=files.get("metadata", ""),
        )
                
        return state

    def run(self, *, prompt: str, style: str = "", size: str = "1024x1024", language: str = "en") -> Dict[str, Any]:
        state = ImageAgentState(
            original_prompt=prompt,
            style=style,
            size=size,
            language=language,
        )
        out_state = self.graph.invoke(state)
        return out_state if isinstance(out_state, dict) else out_state.__dict__.copy()