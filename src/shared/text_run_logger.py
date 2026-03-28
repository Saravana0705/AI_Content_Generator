from __future__ import annotations
from typing import List
import csv
import os
from datetime import datetime
from typing import Any, Optional


class TextRunLogger:
    def __init__(self, csv_path: str = "runs/text_runs.csv"):
        self.csv_path = csv_path
        csv_dir = os.path.dirname(self.csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

    def _safe(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        return " ".join(text.split())

    def _header(self) -> list[str]:
        return [
            "timestamp",
            "language",
            "raw_prompt",
            "enhanced_prompt",
            "final_prompt",
            "model_name",
            "content_type",
            "tone",
            "total_time_sec",
            "optimized_score",
            "revision_rounds",
            "initial_score",
            "revised_score",
            "score_improvement",
            "final_text",
        ]

    def log_run(
        self,
        *,
        language: str,
        raw_prompt: str,
        enhanced_prompt: str,
        final_prompt: str,
        model_name: str,
        content_type: str,
        tone: str,
        total_time_sec: float = 0.0,
        optimized_score: float = 0.0,
        revision_rounds: int = 0,
        initial_score: Optional[float] = None,
        revised_score: Optional[float] = None,
        score_improvement: Optional[float] = None,
        final_text: str = "",
    ) -> None:
        header = self._header()

        row = [
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            self._safe(language),
            self._safe(raw_prompt),
            self._safe(enhanced_prompt),
            self._safe(final_prompt),
            self._safe(model_name),
            self._safe(content_type),
            self._safe(tone),
            0 if total_time_sec is None else total_time_sec,
            0 if optimized_score is None else optimized_score,
            0 if revision_rounds is None else revision_rounds,
            "" if initial_score is None else initial_score,
            "" if revised_score is None else revised_score,
            "" if score_improvement is None else score_improvement,
            self._safe(final_text),
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