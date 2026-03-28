from __future__ import annotations

from typing import Any, Dict, Optional, List
from src.main_agent.router import Router


class Supervisor:
    def __init__(self, review_threshold: float = 60.0):
        self.status = "idle"
        self.router = Router(review_threshold=review_threshold)

    def coordinate_workflow(
        self,
        input_data: str,
        *,
        subagent_type: str = "text_generator",
        content_type: str = "blog_article",
        tone: str = "neutral",
        keywords: Optional[List[str]] = None,
        model_name: str = "",
        image_size: str = "1024x1024",
        language: str = "en",
    ) -> Dict[str, Any]:
        self.status = "running"
        result = self.router.route_to_subagent(
            input_data,
            subagent_type,
            content_type=content_type,
            tone=tone,
            keywords=keywords or [],
            model_name=model_name,
            image_size=image_size,
            language=language,
        )
        self.status = "idle"
        return result
