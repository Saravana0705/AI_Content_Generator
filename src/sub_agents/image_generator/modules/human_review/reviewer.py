from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

@dataclass
class ImageReviewResult:
    approved: bool
    decision: str
    comments: List[str]
    technical_ok: bool
    clip_score: float
    aesthetic_score: float
    notes: str

class ImageReviewer:
    def review(self, *, technical_ok: bool, clip_score: float, aesthetic_score: float, notes: str = "") -> Dict[str, Any]:
        comments: List[str] = []

        if not technical_ok:
            return asdict(ImageReviewResult(
                approved=False, decision="revise",
                comments=["Image failed technical validation. Regenerate or check API response."],
                technical_ok=technical_ok, clip_score=clip_score, aesthetic_score=aesthetic_score, notes=notes
            ))

        # For images, “approve” can mean “good enough to export”
        approved = clip_score >= 0.25
        if approved:
            comments.append("Image meets minimum alignment threshold and can be exported.")
        else:
            comments.append("Image alignment is low; consider regenerating with clearer constraints.")

        # Aesthetic is advisory unless critically low (your spec)
        if aesthetic_score < 3.0:
            comments.append("Aesthetic score is critically low; retry recommended.")
        elif aesthetic_score < 5.0:
            comments.append("Aesthetic score is below preferred range; optional improvement.")

        return asdict(ImageReviewResult(
            approved=approved,
            decision="approve" if approved else "revise",
            comments=comments,
            technical_ok=technical_ok,
            clip_score=clip_score,
            aesthetic_score=aesthetic_score,
            notes=notes or "",
        ))