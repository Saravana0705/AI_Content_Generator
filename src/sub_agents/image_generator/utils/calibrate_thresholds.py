from __future__ import annotations

import argparse
import csv
import json
import os
from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Pipeline modules
from src.sub_agents.image_generator.modules.analyzer.analyzer import ImageAnalyzer
from src.sub_agents.image_generator.modules.content_retrieval.style_retriever import StyleRetriever
from src.sub_agents.image_generator.modules.generator.image_generator import ImageGenerator
from src.sub_agents.image_generator.modules.optimizer.optimizer import ImageOptimizer


def _quantile(values: List[float], q: float) -> float:
    """
    Compute quantile without numpy, using linear interpolation.
    q in [0,1]. For n values, uses position p=(n-1)*q
    """
    if not values:
        raise ValueError("Cannot compute quantile of empty list.")
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)

    xs = sorted(values)
    n = len(xs)
    pos = (n - 1) * q
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def _load_prompts(prompts_file: str) -> List[str]:
    """
    Supports:
      - .txt (one prompt per line)
      - .csv with a column named 'prompt' (or first column if not found)
      - .json as a list of strings or list of dicts with key 'prompt'
    """
    path = Path(prompts_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        return [ln for ln in lines if ln]

    if suffix == ".csv":
        prompts: List[str] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return []
        header = rows[0]
        prompt_idx = None
        for i, col in enumerate(header):
            if col.strip().lower() == "prompt":
                prompt_idx = i
                break
        if prompt_idx is None:
            prompt_idx = 0

        for row in rows[1:]:
            if len(row) > prompt_idx:
                p = (row[prompt_idx] or "").strip()
                if p:
                    prompts.append(p)
        return prompts

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        prompts: List[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    p = item.strip()
                    if p:
                        prompts.append(p)
                elif isinstance(item, dict):
                    p = str(item.get("prompt", "")).strip()
                    if p:
                        prompts.append(p)
        else:
            raise ValueError("JSON prompts file must be a list.")
        return prompts

    raise ValueError("Unsupported prompts_file type. Use .txt, .csv, or .json.")


@dataclass
class CalibrationConfig:
    clip_q: float
    aesthetic_r: float
    window_size: int
    calibrated_at: str
    style_default: str
    size_default: str
    # Optional identifiers for traceability (fill if you have them)
    clip_model: Optional[str] = None
    aesthetic_model: Optional[str] = None


@dataclass
class CalibrationResult:
    clip_threshold: float
    aesthetic_threshold: float
    clip_q: float
    aesthetic_r: float
    window_size: int
    calibrated_at: str
    style_default: str
    size_default: str
    clip_model: Optional[str] = None
    aesthetic_model: Optional[str] = None


def run_calibration(
    prompts: List[str],
    *,
    clip_q: float,
    aesthetic_r: float,
    style_default: str,
    size_default: str,
    out_json: str,
    out_scores_csv: Optional[str] = None,
    limit: Optional[int] = None,
) -> CalibrationResult:
    if not prompts:
        raise ValueError("No prompts provided for calibration.")

    if limit is not None and limit > 0:
        prompts = prompts[:limit]

    analyzer = ImageAnalyzer()
    retriever = StyleRetriever()
    generator = ImageGenerator()

    # IMPORTANT: no retries during calibration
    optimizer = ImageOptimizer(
        calibration_path=None,  # don't load thresholds
        max_retries=0,
    )

    clip_scores: List[float] = []
    aesthetic_scores: List[float] = []
    technical_ok_count = 0

    per_prompt_rows: List[Dict[str, Any]] = []
    
    STYLES = ["photorealistic", "anime", "3d", "illustration"]
    
    for idx, prompt in enumerate(prompts, start=1):
        # Rotate style across prompts
        style = STYLES[(idx - 1) % len(STYLES)]

        print(f"[CAL] prompt {idx}/{len(prompts)} | style={style}")
        intent = analyzer.analyze(prompt=prompt, style=style, size=size_default)
        style_payload = retriever.retrieve(style=intent.get("style", style))
        final_prompt = retriever.build_final_prompt(intent=intent, style_payload=style_payload)

        image_bytes, image_meta = generator.generate(prompt=final_prompt, size=intent.get("size", size_default))

        # Run evaluation ONCE (no retry)
        technical_ok, tech_notes = optimizer._technical_validate(image_bytes, intent=intent)
        if technical_ok:
            technical_ok_count += 1
            clip = float(optimizer._clip_similarity(prompt_text=final_prompt, image_bytes=image_bytes))
            aes = float(optimizer._aesthetic_score(image_bytes=image_bytes))
            clip_scores.append(clip)
            aesthetic_scores.append(aes)
        else:
            clip = 0.0
            aes = 0.0

        per_prompt_rows.append({
            "idx": idx,
            "raw_prompt": prompt,
            "final_prompt": final_prompt,
            "style": style,
            "size": intent.get("size"),
            "technical_ok": technical_ok,
            "clip_score": clip,
            "aesthetic_score": aes,
            "notes": tech_notes,
            "model": image_meta.get("model"),
        })

    if not clip_scores or not aesthetic_scores:
        raise RuntimeError(
            "Calibration produced no valid scores (technical_ok never true). "
            "Check generator output and technical validator."
        )

    clip_threshold = float(_quantile(clip_scores, clip_q))
    aesthetic_threshold = float(_quantile(aesthetic_scores, aesthetic_r))

    calibrated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    result = CalibrationResult(
        clip_threshold=clip_threshold,
        aesthetic_threshold=aesthetic_threshold,
        clip_q=float(clip_q),
        aesthetic_r=float(aesthetic_r),
        window_size=len(prompts),
        calibrated_at=calibrated_at,
        style_default=style_default,
        size_default=size_default,
        # fill if you have explicit identifiers
        clip_model=None,
        aesthetic_model=None,
    )

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")

    if out_scores_csv:
        csv_path = Path(out_scores_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_scores_csv(csv_path, per_prompt_rows)

    print("\nCalibration complete")
    print(f"- prompts: {len(prompts)}")
    print(f"- technical_ok: {technical_ok_count}/{len(prompts)}")
    print(f"- clip_threshold (q={clip_q}): {clip_threshold:.6f}")
    print(f"- aesthetic_threshold (r={aesthetic_r}): {aesthetic_threshold:.6f}")
    print(f"- saved: {out_path}")

    return result


def _write_scores_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    header = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate dynamic thresholds for Image Optimizer (quantile-based).")
    ap.add_argument("--prompts_file", required=True, help="Path to prompts file (.txt, .csv, .json).")
    ap.add_argument("--clip_q", type=float, default=0.10, help="Lower-tail percentile for CLIP threshold (e.g. 0.10).")
    ap.add_argument("--aesthetic_r", type=float, default=0.10, help="Lower-tail percentile for aesthetic threshold (e.g. 0.10).")
    ap.add_argument("--style", type=str, default="photorealistic", help="Default style to use for calibration.")
    ap.add_argument("--size", type=str, default="1024x1024", help="Default size to use for calibration.")
    ap.add_argument("--out_json", type=str, default="calibration/image_thresholds.json", help="Output JSON path.")
    ap.add_argument("--out_scores_csv", type=str, default="calibration/calibration_scores.csv", help="Optional CSV of per-prompt scores.")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of prompts.")
    args = ap.parse_args()

    prompts = _load_prompts(args.prompts_file)
    limit = args.limit if args.limit and args.limit > 0 else None

    run_calibration(
        prompts,
        clip_q=args.clip_q,
        aesthetic_r=args.aesthetic_r,
        style_default=args.style,
        size_default=args.size,
        out_json=args.out_json,
        out_scores_csv=args.out_scores_csv,
        limit=limit,
    )


if __name__ == "__main__":
    main()