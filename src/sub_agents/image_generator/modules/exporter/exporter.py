from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ImageExportResult:
    ok: bool
    message: str
    output_dir: str
    files: Dict[str, str]
    metadata: Dict[str, Any]


class ImageExporter:
    def __init__(self, output_root: str = "exports/images"):
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

    def export_image(
        self,
        *,
        image_bytes: Optional[bytes],
        raw_prompt: str,
        enhanced_prompt: str,
        final_prompt: str,
        intent: Dict[str, Any],
        meta: Dict[str, Any],
        clip_score: float,
        aesthetic_score: float,
        retries: int,
        review_result: Optional[dict] = None,
        require_approved: bool = True,
        image_ext: str = "png",
        clip_threshold: Optional[float] = None,
        clip_q: Optional[float] = None,
        aesthetic_threshold: Optional[float] = None,
        aesthetic_r: Optional[float] = None,
        calibration_window_size: Optional[int] = None,
        calibration_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        review_result = review_result or {}
        approved = bool(review_result.get("approved", False))

        allow_unapproved = os.getenv("EXPORT_ALLOW_UNAPPROVED", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if require_approved and not approved and not allow_unapproved:
            return asdict(
                ImageExportResult(
                    ok=False,
                    message="Export blocked: image not approved by Reviewer.",
                    output_dir="",
                    files={},
                    metadata={"review_result": review_result},
                )
            )

        if not image_bytes:
            return asdict(
                ImageExportResult(
                    ok=False,
                    message="No image bytes to export.",
                    output_dir="",
                    files={},
                    metadata={},
                )
            )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_root, f"run_{ts}")
        os.makedirs(run_dir, exist_ok=True)

        img_path = os.path.join(run_dir, f"image.{image_ext}")
        with open(img_path, "wb") as f:
            f.write(image_bytes)

        # Use explicit threshold args if provided; otherwise try meta dict
        clip_threshold = clip_threshold if clip_threshold is not None else meta.get("clip_threshold")
        clip_q = clip_q if clip_q is not None else meta.get("clip_q")
        aesthetic_threshold = (
            aesthetic_threshold if aesthetic_threshold is not None else meta.get("aesthetic_threshold")
        )
        aesthetic_r = aesthetic_r if aesthetic_r is not None else meta.get("aesthetic_r")
        calibration_window_size = (
            calibration_window_size
            if calibration_window_size is not None
            else meta.get("calibration_window_size")
        )
        calibration_path = calibration_path if calibration_path is not None else meta.get("calibration_path")

        # Fallbacks so the semantics remain consistent for both English and German
        raw_prompt = (raw_prompt or "").strip()
        enhanced_prompt = (enhanced_prompt or "").strip() or final_prompt or raw_prompt
        final_prompt = (final_prompt or "").strip() or enhanced_prompt or raw_prompt

        payload = {
            "timestamp": ts,
            "raw_prompt": raw_prompt,
            "enhanced_prompt": enhanced_prompt,
            "final_prompt": final_prompt,
            "intent": intent,
            "image_meta": meta,
            "clip_score": clip_score,
            "aesthetic_score": aesthetic_score,
            "retries": retries,
            "review_result": review_result,
            "approved": approved,
            "source_language": intent.get("source_language") if isinstance(intent, dict) else None,
            "clip_threshold": clip_threshold,
            "clip_q": clip_q,
            "aesthetic_threshold": aesthetic_threshold,
            "aesthetic_r": aesthetic_r,
            "calibration_window_size": calibration_window_size,
            "calibration_path": calibration_path,
            "calibrated_at": meta.get("calibrated_at"),
            "style_default": meta.get("style_default"),
            "size_default": meta.get("size_default"),
        }

        json_path = os.path.join(run_dir, "metadata.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return asdict(
            ImageExportResult(
                ok=True,
                message="Image export completed.",
                output_dir=run_dir,
                files={"image": img_path, "metadata": json_path},
                metadata=payload,
            )
        )

    