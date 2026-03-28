from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class ReviewResult:
    approved: bool
    decision: str
    quality_band: str
    score: float
    threshold: float
    comments: List[str]
    notes: str
    content: str


class Reviewer:
    def __init__(self, threshold: float = 60.0):
        self.threshold = threshold

    def _quality_band(self, score: float) -> str:
        if score >= 85:
            return "excellent"
        if score >= 70:
            return "good"
        if score >= self.threshold:
            return "acceptable"
        if score >= self.threshold * 0.7:
            return "needs_minor_revision"
        return "needs_major_revision"

    def review(self, content: str, score: float, notes: str = "") -> Dict[str, Any]:
        comments: List[str] = []

        if score >= self.threshold:
            approved = True
            decision = "approve"
            comments.append(
                "Content meets or exceeds the quality benchmark and is ready for export."
            )
        elif score >= self.threshold * 0.7:
            approved = False
            decision = "revise"
            comments.append(
                "Content is close to the benchmark but would benefit from light manual editing."
            )
        else:
            approved = False
            decision = "revise"
            comments.append(
                "Content falls significantly below the quality benchmark; substantial editing is recommended."
            )

        lower_notes = notes.lower() if notes else ""
        if "readability" in lower_notes:
            comments.append("Pay special attention to readability and structure.")
        if "conciseness" in lower_notes:
            comments.append("Tighten the content to improve conciseness.")
        if "sentiment" in lower_notes:
            comments.append("Check tone and sentiment to match the desired style.")
        if "repetition" in lower_notes:
            comments.append("Reduce unnecessary repetition and redundant phrases.")

        result = ReviewResult(
            approved=approved,
            decision=decision,
            quality_band=self._quality_band(score),
            score=score,
            threshold=self.threshold,
            comments=comments,
            notes=notes or "",
            content=content,
        )

        return asdict(result)
