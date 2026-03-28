import csv
import os
from datetime import datetime
from typing import Optional


RUNS_DIR = "runs"
RUNS_FILE = os.path.join(RUNS_DIR, "text_runs.csv")


def append_run(
    *,
    language: str = "",
    raw_prompt: str = "",
    enhanced_prompt: str = "",
    final_prompt: str = "",
    model_name: str = "",
    content_type: str = "",
    tone: str = "",
    total_time_sec: float = 0.0,
    final_text: str = "",
    optimized_score: float = 0.0,
    revision_rounds: int = 0,
    initial_score: Optional[float] = None,
    revised_score: Optional[float] = None,
    score_improvement: Optional[float] = None,
) -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)

    file_exists = os.path.isfile(RUNS_FILE)

    header = [
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

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        language,
        raw_prompt,
        enhanced_prompt,
        final_prompt,
        model_name,
        content_type,
        tone,
        round(float(total_time_sec or 0.0), 4),
        float(optimized_score or 0.0),
        int(revision_rounds or 0),
        "" if initial_score is None else float(initial_score),
        "" if revised_score is None else float(revised_score),
        "" if score_improvement is None else float(score_improvement),
        final_text,
    ]

    with open(RUNS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)