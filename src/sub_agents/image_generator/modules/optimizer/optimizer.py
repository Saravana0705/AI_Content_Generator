from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional

from src.sub_agents.image_generator.modules.generator.image_generator import ImageGenerator


class ImageOptimizer:
    """
    Streamlit-safe optimizer:
    - DOES NOT import torch or open_clip.
    - Calls a separate scorer process to compute CLIP + aesthetic.
    - Uses distribution-calibrated thresholds to decide whether to retry generation once.
    - Supports language-aware prompt strengthening for retries.
    """

    def __init__(
        self,
        *,
        calibration_path: Optional[str] = "calibration/image_thresholds.json",
        max_retries: int = 1,
        fallback_clip_threshold: float = 0.25,
        fallback_aesthetic_threshold: float = 3.0,
    ):
        self.max_retries = max_retries
        self.generator = ImageGenerator()

        # Thresholds
        self.clip_threshold: float = fallback_clip_threshold
        self.aesthetic_threshold: float = fallback_aesthetic_threshold
        self.clip_q: Optional[float] = None
        self.aesthetic_r: Optional[float] = None
        self.window_size: Optional[int] = None
        self.calibration_path: Optional[str] = calibration_path

        # Extra calibration metadata
        self.calibrated_at: Optional[str] = None
        self.style_default: Optional[str] = None
        self.size_default: Optional[str] = None

        if calibration_path:
            self._load_calibration(calibration_path)

        # Scoring subprocess behavior
        self.scoring_timeout_sec = int(os.getenv("SCORING_TIMEOUT_SEC", "120"))
        self.scoring_retries = int(os.getenv("SCORING_SUBPROCESS_RETRIES", "2"))

    def warmup(self) -> None:
        """No-op now. Torch is not loaded in this process."""
        return

    def _load_calibration(self, path: str) -> None:
        if not os.path.exists(path):
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            self.clip_threshold = float(cfg.get("clip_threshold", self.clip_threshold))
            self.aesthetic_threshold = float(
                cfg.get("aesthetic_threshold", self.aesthetic_threshold)
            )
            self.clip_q = cfg.get("clip_q", None)
            self.aesthetic_r = cfg.get("aesthetic_r", None)
            self.window_size = cfg.get("window_size", None)

            # Optional metadata
            self.calibrated_at = cfg.get("calibrated_at", None)
            self.style_default = cfg.get("style_default", None)
            self.size_default = cfg.get("size_default", None)

        except Exception:
            # Keep fallback values if calibration cannot be read
            return

    def validate_and_maybe_retry(
        self,
        *,
        intent: Dict[str, Any],
        prompt: str,
        image_bytes: Optional[bytes],
        image_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        retries = 0
        final_prompt = prompt
        final_bytes = image_bytes
        final_meta = image_meta or {}

        def run_eval(img_bytes: bytes, prompt_for_scoring: str) -> Dict[str, Any]:
            technical_ok, tech_notes = self._technical_validate(img_bytes, intent=intent)
            if not technical_ok:
                return {
                    "technical_ok": False,
                    "clip_score": 0.0,
                    "aesthetic_score": 0.0,
                    "notes": tech_notes,
                    "scoring_ok": False,
                    "scoring_error": "technical_failed",
                }

            scoring = self._score_via_subprocess(
                prompt_text=prompt_for_scoring,
                image_bytes=img_bytes,
            )
            if not scoring.get("ok"):
                return {
                    "technical_ok": True,
                    "clip_score": 0.0,
                    "aesthetic_score": 0.0,
                    "notes": f"{tech_notes}\nScoring failed: {scoring.get('error')}",
                    "scoring_ok": False,
                    "scoring_error": scoring.get("error"),
                }

            clip_score = float(scoring["clip_score"])
            aesthetic_score = float(scoring["aesthetic_score"])

            return {
                "technical_ok": True,
                "clip_score": clip_score,
                "aesthetic_score": aesthetic_score,
                "notes": (
                    f"{tech_notes}\nCLIP={clip_score:.3f} | "
                    f"Aesthetic={aesthetic_score:.2f}"
                ).strip(),
                "scoring_ok": True,
                "scoring_error": None,
            }

        # Initial evaluation
        if not final_bytes:
            eval0 = {
                "technical_ok": False,
                "clip_score": 0.0,
                "aesthetic_score": 0.0,
                "notes": "No image bytes returned.",
                "scoring_ok": False,
                "scoring_error": "no_image_bytes",
            }
        else:
            eval0 = run_eval(final_bytes, final_prompt)

        # Retry decision
        needs_retry = False
        if not eval0["technical_ok"]:
            needs_retry = True
        elif eval0["scoring_ok"]:
            if eval0["clip_score"] < self.clip_threshold:
                needs_retry = True
            elif (
                eval0["clip_score"] >= self.clip_threshold
                and eval0["aesthetic_score"] < self.aesthetic_threshold
            ):
                needs_retry = True
        else:
            # If scoring itself failed, do not regenerate just for that.
            needs_retry = False

        # Single retry if needed
        if needs_retry and retries < self.max_retries:
            retries += 1

            final_prompt = self._strengthen_prompt(
                intent=intent,
                prompt=final_prompt,
                reason=eval0["notes"],
            )

            final_bytes, gen_meta = self.generator.generate(
                prompt=final_prompt,
                size=intent.get("size", "1024x1024"),
            )

            final_meta = {
                **final_meta,
                **(gen_meta or {}),
                "retry": True,
                "retry_reason": eval0["notes"],
                "retry_prompt": final_prompt,
            }

            if not final_bytes:
                eval_final = {
                    "technical_ok": False,
                    "clip_score": 0.0,
                    "aesthetic_score": 0.0,
                    "notes": "Retry generation returned no image bytes.",
                    "scoring_ok": False,
                    "scoring_error": "retry_no_image_bytes",
                }
            else:
                eval_final = run_eval(final_bytes, final_prompt)
        else:
            eval_final = eval0

        return {
            "image_bytes": final_bytes,
            "image_meta": final_meta,
            "final_prompt": final_prompt,
            "retries": retries,
            "clip_threshold": self.clip_threshold,
            "clip_q": self.clip_q,
            "aesthetic_threshold": self.aesthetic_threshold,
            "aesthetic_r": self.aesthetic_r,
            "calibration_window_size": self.window_size,
            "calibration_path": self.calibration_path,
            "calibrated_at": self.calibrated_at,
            "style_default": self.style_default,
            "size_default": self.size_default,
            "scoring_ok": eval_final.get("scoring_ok", False),
            "scoring_error": eval_final.get("scoring_error"),
            **eval_final,
        }

    def _technical_validate(self, image_bytes: bytes, *, intent: Dict[str, Any]) -> tuple[bool, str]:
        if not image_bytes or len(image_bytes) < 100:
            return False, "Technical validation failed: empty/too-small image payload."

        magic = image_bytes[:12]
        is_png = magic.startswith(b"\x89PNG")
        is_jpg = magic.startswith(b"\xff\xd8\xff")
        if not (is_png or is_jpg):
            return False, "Technical validation failed: unknown image format."

        try:
            from PIL import Image

            im = Image.open(io.BytesIO(image_bytes))
            w, h = im.size
            expected = intent.get("size", "")
            if isinstance(expected, str) and "x" in expected:
                ew, eh = expected.split("x", 1)
                if int(ew) != w or int(eh) != h:
                    return False, f"Resolution mismatch: got {w}x{h}, expected {expected}."
        except Exception:
            return False, "Technical validation failed: could not decode image."

        return True, "Technical validation OK."

    def _score_via_subprocess(self, *, prompt_text: str, image_bytes: bytes) -> Dict[str, Any]:
        """
        Calls: python -m src.sub_agents.image_generator.utils.score_image
        Writes temp image + temp result json, reads result.
        Retries subprocess a couple times for reliability.
        """
        with tempfile.TemporaryDirectory() as td:
            img_path = os.path.join(td, "img.png")
            out_json = os.path.join(td, "scores.json")

            with open(img_path, "wb") as f:
                f.write(image_bytes)

            cmd = [
                os.environ.get("PYTHON_EXECUTABLE", "python"),
                "-m",
                "src.sub_agents.image_generator.utils.score_image",
                "--image_path",
                img_path,
                "--prompt",
                prompt_text,
                "--out_json",
                out_json,
            ]

            last_err = None
            for _ in range(max(1, self.scoring_retries)):
                try:
                    cp = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.scoring_timeout_sec,
                        check=False,
                    )

                    if not os.path.exists(out_json):
                        last_err = (
                            f"scorer_no_output_json | rc={cp.returncode} | "
                            f"stderr={cp.stderr[-400:]}"
                        )
                        continue

                    with open(out_json, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    if data.get("ok"):
                        return data

                    last_err = data.get("error") or f"scorer_failed_rc={cp.returncode}"

                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"

            return {"ok": False, "error": last_err or "unknown_scoring_error"}

    def _strengthen_prompt(self, *, intent: Dict[str, Any], prompt: str, reason: str = "") -> str:
        lang = str(intent.get("source_language", "en") or "en").strip().lower()
        if lang == "de":
            return self._strengthen_prompt_de(intent=intent, prompt=prompt, reason=reason)
        return self._strengthen_prompt_en(intent=intent, prompt=prompt, reason=reason)

    def _strengthen_prompt_en(
        self,
        *,
        intent: Dict[str, Any],
        prompt: str,
        reason: str = "",
    ) -> str:
        must_have = intent.get("must_have", []) or []
        counts = intent.get("counts", {}) or {}
        spatial = intent.get("spatial_relations", []) or []
        exclude = intent.get("exclude", []) or []

        parts = [prompt.strip()]

        if must_have:
            parts.append("Must include: " + ", ".join(map(str, must_have)))
        if counts:
            parts.append(
                "Counts: " + ", ".join(f"{k}={v}" for k, v in counts.items())
            )
        if spatial:
            parts.append("Spatial constraints: " + ", ".join(map(str, spatial)))
        if exclude:
            parts.append("Must avoid: " + ", ".join(map(str, exclude)))
        if reason:
            parts.append("Fix issues noted: " + str(reason))

        parts.append(
            "Clear composition, high detail, strong semantic alignment, high visual quality."
        )
        return ". ".join(parts).strip()

    def _strengthen_prompt_de(
        self,
        *,
        intent: Dict[str, Any],
        prompt: str,
        reason: str = "",
    ) -> str:
        must_have = intent.get("must_have", []) or []
        counts = intent.get("counts", {}) or {}
        spatial = intent.get("spatial_relations", []) or []
        exclude = intent.get("exclude", []) or []

        parts = [prompt.strip()]

        if must_have:
            parts.append("Muss enthalten: " + ", ".join(map(str, must_have)))
        if counts:
            parts.append(
                "Anzahl genau beachten: " + ", ".join(f"{k}={v}" for k, v in counts.items())
            )
        if spatial:
            parts.append(
                "Räumliche Beziehungen beachten: " + ", ".join(map(str, spatial))
            )
        if exclude:
            parts.append("Nicht enthalten: " + ", ".join(map(str, exclude)))
        if reason:
            parts.append("Zu korrigierende Probleme: " + str(reason))

        parts.append(
            "Klare Komposition, hohe Details, starke semantische Übereinstimmung, hohe visuelle Qualität."
        )
        return ". ".join(parts).strip()