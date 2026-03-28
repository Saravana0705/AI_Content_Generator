from __future__ import annotations

import os
import json
import re
import requests
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class ExportResult:
    ok: bool
    message: str
    output_dir: str
    files: Dict[str, str]  # format -> filepath
    metadata: Dict[str, Any]


class Exporter:
    """
    Exporter module (demo-ready):
    - saves generated/approved content to local files
    - supports txt, md, html, json
    - returns file paths + metadata so UI can display / download later

    IMPORTANT:
    - By default, exports are blocked when require_approved=True and Reviewer did not approve.
    - For benchmarking / analysis, you can override via env:
        EXPORT_ALLOW_UNAPPROVED=1
      which will export even if not approved (and mark the output folder as UNAPPROVED).
    """

    def __init__(self, output_root: str = "exports/text"):
        self.output_root = output_root
        
    def export_content(
        self,
        *,
        content: str,
        content_type: str = "blog_article",
        topic: str = "",
        tone: str = "neutral",
        keywords: Optional[list] = None,
        score: Optional[float] = None,
        notes: str = "",
        review_result: Optional[dict] = None,
        formats: tuple[str, ...] = ("txt", "md", "html", "json"),
        file_prefix: Optional[str] = None,
        require_approved: bool = True,
        model_name: str = "",
        prompt_id: str = "",
        language: str = "en",
        enhanced_prompt: str = "",
        retrieval_query: str = "",
        raw_prompt: str = "",
        final_prompt: str = "",
    ) -> Dict[str, Any]:

        """
        Exports the content to selected formats and returns a structured result.

        - If require_approved=True, it will only export if review_result indicates approved.
        - Override for benchmarking via env:
            EXPORT_ALLOW_UNAPPROVED=1
        """
        keywords = keywords or []
        review_result = review_result or {}

        if not content or not content.strip():
            result = ExportResult(
                ok=False,
                message="No content provided to export.",
                output_dir="",
                files={},
                metadata={},
            )
            return asdict(result)

        approved = bool(review_result.get("approved", False))
        allow_unapproved = (os.getenv("EXPORT_ALLOW_UNAPPROVED", "0").strip().lower() in ("1", "true", "yes"))

        if require_approved and not approved and not allow_unapproved:
            result = ExportResult(
                ok=False,
                message="Export blocked: content is not approved by Reviewer.",
                output_dir="",
                files={},
                metadata={"review_result": review_result},
            )
            return asdict(result)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = self._slugify(topic) if topic else "content"
        safe_prefix = self._slugify(file_prefix) if file_prefix else f"{safe_topic}_{content_type}"

        # If exporting unapproved, mark the directory clearly
        unapproved_tag = "_UNAPPROVED" if (require_approved and not approved and allow_unapproved) else ""
        run_dir = os.path.join(self.output_root, f"{safe_prefix}_{timestamp}{unapproved_tag}")
        os.makedirs(run_dir, exist_ok=True)

        meta: Dict[str, Any] = {
            "timestamp": timestamp,
            "topic": topic,
            "content_type": content_type,
            "tone": tone,
            "keywords": keywords,
            "score": score,
            "notes": notes,
            "review_result": review_result,
            "approved": approved,
            "exported_unapproved": bool(require_approved and not approved and allow_unapproved),
            "model_name": model_name,
            "prompt_id": prompt_id,
            "language": language,
            "raw_prompt": raw_prompt,
            "enhanced_prompt": enhanced_prompt,
            "retrieval_query": retrieval_query,
            "final_prompt": final_prompt,
        }

        files: Dict[str, str] = {}

        for fmt in formats:
            fmt = fmt.lower().strip()
            if fmt == "txt":
                path = os.path.join(run_dir, f"{safe_prefix}.txt")
                self._write_text(path, content)
                files["txt"] = path

            elif fmt == "md":
                path = os.path.join(run_dir, f"{safe_prefix}.md")
                md = self._to_markdown(content_type, topic, content)
                self._write_text(path, md)
                files["md"] = path

            elif fmt == "html":
                path = os.path.join(run_dir, f"{safe_prefix}.html")
                html = self._to_html(content_type, topic, content)
                self._write_text(path, html)
                files["html"] = path

            elif fmt == "json":
                path = os.path.join(run_dir, f"{safe_prefix}.json")
                payload = {"metadata": meta, "content": content}
                self._write_json(path, payload)
                files["json"] = path

            else:
                continue

        if files:
            if require_approved and not approved and allow_unapproved:
                msg = "Export completed (UNAPPROVED content exported for benchmarking)."
            else:
                msg = "Export completed."
        else:
            msg = "No files exported (no valid formats selected)."

        result = ExportResult(ok=bool(files), message=msg, output_dir=run_dir, files=files, metadata=meta)
        return asdict(result)

    # ---------------------------
    # Publishing (unchanged)
    # ---------------------------

    def publish_to_devto(
        self,
        *,
        title: str,
        body_markdown: str,
        published: bool = False,
        tags: Optional[list[str]] = None,
        description: str = "",
        canonical_url: str = "",
        cover_image: str = "",
    ) -> Dict[str, Any]:
        api_key = os.getenv("DEVTO_API_KEY", "").strip()
        if not api_key:
            return {"ok": False, "message": "DEVTO_API_KEY missing in .env"}

        payload: Dict[str, Any] = {
            "article": {
                "title": title,
                "published": bool(published),
                "body_markdown": body_markdown,
            }
        }

        if tags:
            payload["article"]["tags"] = tags[:5]
        if description:
            payload["article"]["description"] = description
        if canonical_url:
            payload["article"]["canonical_url"] = canonical_url
        if cover_image:
            payload["article"]["cover_image"] = cover_image

        try:
            resp = requests.post(
                "https://dev.to/api/articles",
                headers={"api-key": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )
            if resp.status_code not in (200, 201):
                return {"ok": False, "message": f"DEV.to publish failed: {resp.status_code}", "details": resp.text[:1200]}

            data = resp.json()
            return {
                "ok": True,
                "message": "Published to DEV.to successfully",
                "id": data.get("id"),
                "url": data.get("url"),
                "published": data.get("published"),
            }
        except Exception as e:
            return {"ok": False, "message": f"Exception publishing to DEV.to: {e}"}

    def publish_wordpress_stub(self, *, html: str, title: str) -> Dict[str, Any]:
        return {
            "platform": "wordpress",
            "endpoint": "/wp-json/wp/v2/posts",
            "payload": {"title": title, "content": html, "status": "draft"},
            "note": "Stub only. Add authentication + requests.post to publish.",
        }

    def publish_medium_stub(self, *, markdown: str, title: str) -> Dict[str, Any]:
        return {
            "platform": "medium",
            "payload": {"title": title, "contentFormat": "markdown", "content": markdown, "publishStatus": "draft"},
            "note": "Stub only. Add token + HTTP call to publish.",
        }

    def publish_linkedin_stub(self, *, text: str) -> Dict[str, Any]:
        return {"platform": "linkedin", "payload": {"text": text}, "note": "Stub only. LinkedIn requires OAuth."}

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def _write_text(self, path: str, text: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def _write_json(self, path: str, obj: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _to_markdown(self, content_type: str, topic: str, content: str) -> str:
        header = f"# {topic}\n\n" if topic and content_type in {"blog_article", "news_article"} else ""
        return header + content.strip() + "\n"

    def _to_html(self, content_type: str, topic: str, content: str) -> str:
        title = topic if topic else "Exported Content"
        body = self._escape_html(content).replace("\n", "<br/>\n")
        return (
            "<!doctype html>\n"
            "<html>\n<head>\n"
            f"<meta charset='utf-8'/>\n<title>{self._escape_html(title)}</title>\n"
            "</head>\n<body style='font-family: Arial, sans-serif; line-height: 1.5; padding: 24px;'>\n"
            f"<h1>{self._escape_html(title)}</h1>\n"
            f"<div>{body}</div>\n"
            "</body>\n</html>\n"
        )

    def _escape_html(self, s: str) -> str:
        return (
            (s or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def _slugify(self, text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text[:60] if text else "content"
