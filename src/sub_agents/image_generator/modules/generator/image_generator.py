from __future__ import annotations
import os
from typing import Any, Dict, Tuple, Optional

try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None

class ImageGenerator:
    def __init__(self):
        if OpenAIClient is None:
            raise ImportError("openai package not available. Run: pip install openai")

        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY missing")

        self.client = OpenAIClient(api_key=api_key)
        self.model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1-mini")

    def generate(self, *, prompt: str, size: str = "1024x1024") -> Tuple[bytes, Dict[str, Any]]:
        """
        Return: (image_bytes, metadata)
        """
        # NOTE: Exact method names depend on your installed OpenAI SDK version.
        # Keep this isolated so future swaps are easy (per your design). :contentReference[oaicite:12]{index=12}
        resp = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size=size,
        )

        # Typical patterns: resp.data[0].b64_json OR resp.data[0].url
        data0 = resp.data[0]
        meta: Dict[str, Any] = {"model": self.model, "size": size}

        if getattr(data0, "b64_json", None):
            import base64
            img_bytes = base64.b64decode(data0.b64_json)
            meta["encoding"] = "b64_json"
            return img_bytes, meta

        if getattr(data0, "url", None):
            # If URL-based, you must download bytes (requests)
            import requests
            r = requests.get(data0.url, timeout=30)
            r.raise_for_status()
            meta["encoding"] = "url"
            meta["url"] = data0.url
            return r.content, meta

        raise RuntimeError("Image generation response missing expected fields.")