from __future__ import annotations
from datetime import datetime
import csv
import os
from typing import Any, Dict, Optional, List 


class ImageRunLogger:
    def __init__(self, csv_path: str = "runs/image_runs.csv"):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

    def _safe(self, value: Optional[str]) -> str:
        if not value:
            return ""
        text = str(value).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        return " ".join(text.split())

    def _header(self):
        return [
            "timestamp",
            "run_type",
            "provider",
            "model",
            "raw_prompt",
            "enhanced_prompt",
            "final_prompt",
            "source_language",
            "style",
            "size",
            "technical_ok",
            "scoring_ok",
            "scoring_error",
            "clip_score",
            "clip_threshold",
            "clip_q",
            "aesthetic_score",
            "aesthetic_threshold",
            "aesthetic_r",
            "calibration_window_size",
            "calibrated_at",
            "retries",
            "approved",
            "output_dir",
            "image_path",
            "metadata_path",
        ]
    
    def _norm_path(self, path:str) -> str:
        if not path:
            return ""
        return path.replace("\\", "/")

    def log_run(
        self,
        *,
        raw_prompt: str,
        enhanced_prompt: str,
        final_prompt: str,
        intent: Dict[str, Any],
        clip_score: float,
        aesthetic_score: float,
        retries: int,
        approved: bool,
        technical_ok: Optional[bool],
        clip_threshold: Optional[float],
        clip_q: Optional[float],
        aesthetic_threshold: Optional[float],
        aesthetic_r: Optional[float],
        calibration_window_size: Optional[int],
        calibrated_at: Optional[str],
        output_dir: str,
        image_path: str,
        metadata_path: str,
    ) -> None:
        header = self._header()

        row = [
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            "image_generation",
            "OpenAI",  # or dynamic if you support multiple
            "GPT-Image-1-Mini",
            self._safe(raw_prompt),
            self._safe(enhanced_prompt),
            self._safe(final_prompt),
            intent.get("source_language"),
            intent.get("style"),
            intent.get("size"),
            technical_ok,
            True,  # scoring_ok
            "",    # scoring_error
            clip_score,
            clip_threshold,
            clip_q,
            aesthetic_score,
            aesthetic_threshold,
            aesthetic_r,
            calibration_window_size,
            calibrated_at,
            retries,
            approved,
            self._norm_path(output_dir),
        ]

        rewrite_header = False
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    existing_header = next(reader, None)
                if existing_header != header:
                    rewrite_header = True
            except Exception:
                rewrite_header = True

        mode = "w" if rewrite_header or not os.path.exists(self.csv_path) else "a"

        with open(self.csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if mode == "w":
                writer.writerow(header)
            writer.writerow(row)