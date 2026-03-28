from __future__ import annotations

from typing import Any, Dict, Optional, List
from src.sub_agents.text_generator.text_agent import TextAgent
from src.sub_agents.image_generator.image_agent import ImageAgent


class Router:
    def __init__(self, review_threshold: float = 60.0):
        self.text_agent = TextAgent(review_threshold=review_threshold)
        self.image_agent = ImageAgent()

    def route_to_subagent(
        self,
        input_data: str,
        subagent_type: str,
        *,
        content_type: str = "blog_article",
        tone: str = "neutral",
        keywords: Optional[List[str]] = None,
        model_name: str = "",
        image_size: str = "1024x1024",
        language: str = "en",
    ) -> Dict[str, Any]:
        if subagent_type == "text_generator":
            return self.text_agent.run(
                topic=input_data,
                content_type=content_type,
                tone=tone,
                keywords=keywords or [],
                model_name=model_name,
                language=language,
            )
        if subagent_type == "image_generator":
            return self.image_agent.run(prompt=input_data, style=tone, size=image_size, language=language)

        return {
            "error": f"No route found for subagent_type='{subagent_type}'.",
            "supported": ["text_generator"],
        }
