from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from PIL import Image

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Reuse your project modules
from src.sub_agents.image_generator.modules.analyzer.analyzer import ImageAnalyzer
from src.sub_agents.image_generator.modules.content_retrieval.style_retriever import StyleRetriever


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multilingual image benchmark across 3 providers.")
    p.add_argument("--prompts_file", type=str, required=True, help="Path to prompts_20 file. Format: lang|prompt")
    p.add_argument("--out_csv", type=str, default="exports/image_benchmark_runs_20x3.csv")
    p.add_argument("--out_dir", type=str, default="exports/image_benchmark_runs")
    p.add_argument("--size", type=str, default="1024x1024")
    p.add_argument("--sleep_sec", type=float, default=0.5)
    p.add_argument("--poll_sec", type=float, default=3.0)
    p.add_argument("--poll_timeout_sec", type=int, default=180)
    p.add_argument("--calibration_path", type=str, default="calibration/image_thresholds.json")
    p.add_argument("--max_retries", type=int, default=1)
    return p.parse_args()


# ---------------------------
# Prompt loading
# ---------------------------

def load_prompts(path: str) -> List[Dict[str, str]]:
    prompts: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if "|" not in raw:
                raise ValueError(f"Invalid prompt line {line_no}: expected 'lang|prompt'")
            lang, prompt = raw.split("|", 1)
            lang = lang.strip().lower()
            prompt = prompt.strip()
            if lang not in {"en", "de"}:
                raise ValueError(f"Invalid language on line {line_no}: {lang}")
            prompts.append(
                {
                    "prompt_id": f"p{len(prompts)+1:02d}",
                    "language": lang,
                    "prompt": prompt,
                }
            )
    if len(prompts) != 20:
        print(f"[WARN] Loaded {len(prompts)} prompts, not 20.")
    return prompts


# ---------------------------
# Calibration
# ---------------------------

def load_calibration(path: str) -> Dict[str, Any]:
    fallback = {
        "clip_threshold": 0.25,
        "aesthetic_threshold": 3.0,
        "clip_q": None,
        "aesthetic_r": None,
        "window_size": None,
        "calibrated_at": None,
    }
    p = Path(path)
    if not p.exists():
        return fallback
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {
            "clip_threshold": float(data.get("clip_threshold", fallback["clip_threshold"])),
            "aesthetic_threshold": float(data.get("aesthetic_threshold", fallback["aesthetic_threshold"])),
            "clip_q": data.get("clip_q"),
            "aesthetic_r": data.get("aesthetic_r"),
            "window_size": data.get("window_size"),
            "calibrated_at": data.get("calibrated_at"),
        }
    except Exception:
        return fallback


# ---------------------------
# Utility
# ---------------------------

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def aspect_ratio_from_size(size: str) -> str:
    if size == "1024x1024":
        return "1:1"
    if size == "1024x1536":
        return "2:3"
    if size == "1536x1024":
        return "3:2"
    return "1:1"


def freepik_aspect_ratio_from_size(size: str) -> str:
    if size == "1024x1024":
        return "square_1_1"
    if size == "1024x1536":
        return "portrait_2_3"
    if size == "1536x1024":
        return "landscape_3_2"
    return "square_1_1"


def csv_safe_text(value: Optional[str]) -> str:
    if not value:
        return ""
    text = str(value).replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    return " ".join(text.split())


def openai_quality_for_model(model: str) -> Optional[str]:
    # Conservative mapping for GPT Image models.
    if model == "gpt-image-1-mini":
        return "medium"
    if model == "gpt-image-1":
        return "high"
    return None


def save_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


# ---------------------------
# Prompt processing
# ---------------------------

def build_enhanced_prompt(
    *,
    analyzer: ImageAnalyzer,
    retriever: StyleRetriever,
    raw_prompt: str,
    language: str,
    size: str,
) -> Tuple[Dict[str, Any], str]:
    intent = analyzer.analyze(
        prompt=raw_prompt,
        style="",
        size=size,
        language=language,
    )
    style_payload = retriever.retrieve(style=intent.get("style", "photorealistic"))
    enhanced_prompt = retriever.build_final_prompt(
        intent=intent,
        style_payload=style_payload,
        language=language,
    )
    intent["enhanced_prompt"] = enhanced_prompt
    return intent, enhanced_prompt


# ---------------------------
# Scoring / validation
# ---------------------------

def technical_validate(image_bytes: bytes, expected_size: str) -> Tuple[bool, str]:
    if not image_bytes or len(image_bytes) < 100:
        return False, "Technical validation failed: empty/too-small image payload."

    try:
        im = Image.open(io.BytesIO(image_bytes))
        w, h = im.size
        if expected_size and "x" in expected_size:
            ew, eh = expected_size.split("x", 1)
            if int(ew) != w or int(eh) != h:
                return False, f"Resolution mismatch: got {w}x{h}, expected {expected_size}."
    except Exception as e:
        return False, f"Technical validation failed: could not decode image. {type(e).__name__}: {e}"

    return True, "Technical validation OK."


def score_image_via_subprocess(prompt_text: str, image_path: str) -> Dict[str, Any]:
    out_json = str(Path(image_path).with_suffix(".scores.json"))
    cmd = [
        sys.executable,
        "-m",
        "src.sub_agents.image_generator.utils.score_image",
        "--image_path",
        image_path,
        "--prompt",
        prompt_text,
        "--out_json",
        out_json,
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)

    if not os.path.exists(out_json):
        return {
            "ok": False,
            "error": f"scorer_no_output_json rc={cp.returncode} stderr={cp.stderr[-400:]}",
        }

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def strengthen_prompt(intent: Dict[str, Any], prompt: str, reason: str = "") -> str:
    lang = str(intent.get("source_language", "en") or "en").strip().lower()
    must_have = intent.get("must_have", []) or []
    counts = intent.get("counts", {}) or {}
    spatial = intent.get("spatial_relations", []) or []
    exclude = intent.get("exclude", []) or []

    base = prompt.rstrip(" .")

    if lang == "de":
        parts = [base]
        if must_have:
            parts.append("Muss enthalten: " + ", ".join(map(str, must_have)))
        if counts:
            parts.append("Anzahl genau beachten: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        if spatial:
            parts.append("Räumliche Beziehungen beachten: " + ", ".join(map(str, spatial)))
        if exclude:
            parts.append("Nicht enthalten: " + ", ".join(map(str, exclude)))
        if reason:
            parts.append("Zu korrigierende Probleme: " + str(reason))
        parts.append("Klare Komposition, hohe Details, starke semantische Übereinstimmung, hohe visuelle Qualität.")
        return ". ".join(parts).strip()

    parts = [base]
    if must_have:
        parts.append("Must include: " + ", ".join(map(str, must_have)))
    if counts:
        parts.append("Counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    if spatial:
        parts.append("Spatial constraints: " + ", ".join(map(str, spatial)))
    if exclude:
        parts.append("Must avoid: " + ", ".join(map(str, exclude)))
    if reason:
        parts.append("Fix issues noted: " + str(reason))
    parts.append("Clear composition, high detail, strong semantic alignment, high visual quality.")
    return ". ".join(parts).strip()


# ---------------------------
# Provider generation
# ---------------------------

def generate_openai(prompt: str, size: str, model: str) -> Tuple[bytes, Dict[str, Any]]:
    if OpenAI is None:
        raise ImportError("openai package not installed")
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing")

    client = OpenAI(api_key=api_key)
    kwargs: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "size": size,
    }
    quality = openai_quality_for_model(model)
    if quality:
        kwargs["quality"] = quality

    resp = client.images.generate(**kwargs)
    data0 = resp.data[0]

    meta = {"provider": "openai", "model": model, "size": size}
    if getattr(data0, "b64_json", None):
        img_bytes = base64.b64decode(data0.b64_json)
        meta["encoding"] = "b64_json"
        return img_bytes, meta

    if getattr(data0, "url", None):
        r = requests.get(data0.url, timeout=60)
        r.raise_for_status()
        meta["encoding"] = "url"
        meta["url"] = data0.url
        return r.content, meta

    raise RuntimeError("OpenAI response missing image payload")


def generate_stability(prompt: str, size: str, model: str) -> Tuple[bytes, Dict[str, Any]]:
    api_key = (os.getenv("STABILITY_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("STABILITY_API_KEY missing")

    # Default to Stable Image Core-style endpoint; override in env if your account/script uses another Stability model path.
    endpoint = os.getenv(
        "STABILITY_ENDPOINT",
        "https://api.stability.ai/v2beta/stable-image/generate/core",
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/*",
    }

    data = {
        "prompt": prompt,
        "output_format": "png",
        "aspect_ratio": aspect_ratio_from_size(size),
    }

    r = requests.post(endpoint, headers=headers, files={"none": ("", b"")}, data=data, timeout=180)
    if r.status_code >= 400:
        raise RuntimeError(f"Stability error {r.status_code}: {r.text[:400]}")

    return r.content, {
        "provider": "stability",
        "model": model,
        "size": size,
        "endpoint": endpoint,
        "encoding": "binary",
    }


def generate_freepik_mystic(
    prompt: str,
    size: str,
    model: str,
    poll_sec: float,
    poll_timeout_sec: int,
) -> Tuple[bytes, Dict[str, Any]]:
    api_key = (os.getenv("FREEPIK_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("FREEPIK_API_KEY missing")

    base_url = os.getenv("FREEPIK_BASE_URL", "https://api.freepik.com")
    create_url = f"{base_url}/v1/ai/mystic"

    headers = {
        "x-freepik-api-key": api_key,
        "Content-Type": "application/json",
    }

    body: Dict[str, Any] = {
        "prompt": prompt,
        "resolution": "1k",  # closest benchmark match to 1024-ish runs
        "aspect_ratio": freepik_aspect_ratio_from_size(size),
    }

    # Optional extras
    webhook_url = (os.getenv("FREEPIK_WEBHOOK_URL") or "").strip()
    if webhook_url:
        body["webhook_url"] = webhook_url

    # Optional model field, if you want to pin a Mystic variant.
    if model and model.lower() not in {"mystic", "freepik-mystic"}:
        body["model"] = model

    r = requests.post(create_url, headers=headers, json=body, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"Freepik create error {r.status_code}: {r.text[:400]}")

    data = r.json().get("data") or {}
    task_id = data.get("task_id")
    if not task_id:
        raise RuntimeError(f"Freepik create response missing task_id: {r.text[:400]}")

    status_url = f"{base_url}/v1/ai/mystic/{task_id}"
    started = time.time()

    last_status = None
    generated_urls: List[str] = []

    while time.time() - started < poll_timeout_sec:
        time.sleep(poll_sec)
        gr = requests.get(status_url, headers={"x-freepik-api-key": api_key}, timeout=60)
        if gr.status_code >= 400:
            raise RuntimeError(f"Freepik status error {gr.status_code}: {gr.text[:400]}")

        gd = gr.json().get("data") or {}
        last_status = gd.get("status")
        generated_urls = gd.get("generated") or []

        if generated_urls:
            img_url = generated_urls[0]
            ir = requests.get(img_url, timeout=120)
            ir.raise_for_status()
            return ir.content, {
                "provider": "freepik",
                "model": model or "mystic",
                "size": size,
                "task_id": task_id,
                "status": last_status,
                "url": img_url,
                "encoding": "url",
            }

        if str(last_status).upper() in {"FAILED", "ERROR", "CANCELLED"}:
            raise RuntimeError(f"Freepik task failed with status={last_status}")

    raise TimeoutError(f"Freepik task polling timed out. last_status={last_status}, task_id={task_id}")

def generate_gemini_llmapi(prompt: str, size: str, model: str) -> Tuple[bytes, Dict[str, Any]]:
    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("LLM_API_KEY missing")

    url = "https://api.llmapi.ai/v1/images/generations"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "size": size
    }

    r = requests.post(url, headers=headers, json=payload, timeout=120)

    if r.status_code >= 400:
        raise RuntimeError(f"LLMAPI error {r.status_code}: {r.text[:400]}")

    data = r.json()

    img_b64 = data["data"][0]["b64_json"]
    img_bytes = base64.b64decode(img_b64)

    return img_bytes, {
        "provider": "google",
        "model": model,
        "size": size,
        "encoding": "b64_json",
        "gateway": "llmapi",
    }


def generate_by_provider(
    *,
    provider: str,
    model: str,
    prompt: str,
    size: str,
    poll_sec: float,
    poll_timeout_sec: int,
) -> Tuple[bytes, Dict[str, Any]]:
    provider = provider.lower()

    if provider == "openai":
        return generate_openai(prompt=prompt, size=size, model=model)
    if provider == "stability":
        return generate_stability(prompt=prompt, size=size, model=model)
    if provider == "freepik":
        return generate_freepik_mystic(
            prompt=prompt,
            size=size,
            model=model,
            poll_sec=poll_sec,
            poll_timeout_sec=poll_timeout_sec,
        )
    if provider == "google":
        return generate_gemini_llmapi(
            prompt=prompt,
            size=size,
            model=model,
        )

    raise ValueError(f"Unsupported provider: {provider}")


# ---------------------------
# Benchmark run
# ---------------------------

def init_csv(path: str) -> None:
    ensure_dir(str(Path(path).parent))
    header = [
        "benchmark_run_id",
        "timestamp",
        "prompt_id",
        "prompt_language",
        "provider",
        "model",
        "raw_prompt",
        "enhanced_prompt",
        "final_prompt",
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
        "elapsed_sec",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)


def append_csv(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)


def run_one(
    *,
    benchmark_run_id: str,
    prompt_id: str,
    language: str,
    raw_prompt: str,
    provider: str,
    model: str,
    size: str,
    calibration: Dict[str, Any],
    analyzer: ImageAnalyzer,
    retriever: StyleRetriever,
    out_dir: str,
    poll_sec: float,
    poll_timeout_sec: int,
    max_retries: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    run_slug = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{provider}_{prompt_id}"
    run_dir = str(Path(out_dir) / run_slug)
    ensure_dir(run_dir)

    try:
        intent, enhanced_prompt = build_enhanced_prompt(
            analyzer=analyzer,
            retriever=retriever,
            raw_prompt=raw_prompt,
            language=language,
            size=size,
        )

        final_prompt = enhanced_prompt
        retries = 0

        image_bytes, image_meta = generate_by_provider(
            provider=provider,
            model=model,
            prompt=final_prompt,
            size=size,
            poll_sec=poll_sec,
            poll_timeout_sec=poll_timeout_sec,
        )

        image_path = str(Path(run_dir) / "image.png")
        save_bytes(image_path, image_bytes)

        technical_ok, tech_notes = technical_validate(image_bytes, size)
        scoring = score_image_via_subprocess(final_prompt, image_path)

        scoring_ok = bool(scoring.get("ok"))
        scoring_error = scoring.get("error") if not scoring_ok else ""

        clip_score = float(scoring.get("clip_score", 0.0)) if scoring_ok else 0.0
        aesthetic_score = float(scoring.get("aesthetic_score", 0.0)) if scoring_ok else 0.0

        needs_retry = False
        if not technical_ok:
            needs_retry = True
        elif scoring_ok and clip_score < calibration["clip_threshold"]:
            needs_retry = True
        elif scoring_ok and clip_score >= calibration["clip_threshold"] and aesthetic_score < calibration["aesthetic_threshold"]:
            needs_retry = True

        if needs_retry and retries < max_retries:
            retries += 1
            reason = tech_notes if not scoring_ok else f"{tech_notes}\nCLIP={clip_score:.3f} | Aesthetic={aesthetic_score:.2f}"
            final_prompt = strengthen_prompt(intent, final_prompt, reason=reason)

            image_bytes, image_meta_retry = generate_by_provider(
                provider=provider,
                model=model,
                prompt=final_prompt,
                size=size,
                poll_sec=poll_sec,
                poll_timeout_sec=poll_timeout_sec,
            )
            image_meta = {**image_meta, **image_meta_retry, "retry": True, "retry_reason": reason, "retry_prompt": final_prompt}

            image_path = str(Path(run_dir) / "image.png")
            save_bytes(image_path, image_bytes)

            technical_ok, tech_notes = technical_validate(image_bytes, size)
            scoring = score_image_via_subprocess(final_prompt, image_path)
            scoring_ok = bool(scoring.get("ok"))
            scoring_error = scoring.get("error") if not scoring_ok else ""
            clip_score = float(scoring.get("clip_score", 0.0)) if scoring_ok else 0.0
            aesthetic_score = float(scoring.get("aesthetic_score", 0.0)) if scoring_ok else 0.0

        approved = bool(technical_ok and scoring_ok and (clip_score >= 0.25))

        metadata = {
            "benchmark_run_id": benchmark_run_id,
            "timestamp": ts,
            "prompt_id": prompt_id,
            "provider": provider,
            "model": model,
            "raw_prompt": raw_prompt,
            "enhanced_prompt": enhanced_prompt,
            "final_prompt": final_prompt,
            "prompt_language": language,
            "intent": intent,
            "image_meta": image_meta,
            "technical_ok": technical_ok,
            "scoring_ok": scoring_ok,
            "scoring_error": scoring_error,
            "clip_score": clip_score,
            "clip_threshold": calibration["clip_threshold"],
            "clip_q": calibration["clip_q"],
            "aesthetic_score": aesthetic_score,
            "aesthetic_threshold": calibration["aesthetic_threshold"],
            "aesthetic_r": calibration["aesthetic_r"],
            "calibration_window_size": calibration["window_size"],
            "calibrated_at": calibration["calibrated_at"],
            "retries": retries,
            "approved": approved,
        }
        metadata_path = str(Path(run_dir) / "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        elapsed_sec = round(time.perf_counter() - started, 3)

        return {
            "benchmark_run_id": benchmark_run_id,
            "timestamp": ts,
            "prompt_id": prompt_id,
            "prompt_language": language,
            "provider": provider,
            "model": model,
            "raw_prompt": csv_safe_text(raw_prompt),
            "enhanced_prompt": csv_safe_text(enhanced_prompt),
            "final_prompt": csv_safe_text(final_prompt),
            "style": intent.get("style"),
            "size": size,
            "technical_ok": technical_ok,
            "scoring_ok": scoring_ok,
            "scoring_error": scoring_error,
            "clip_score": clip_score,
            "clip_threshold": calibration["clip_threshold"],
            "clip_q": calibration["clip_q"],
            "aesthetic_score": aesthetic_score,
            "aesthetic_threshold": calibration["aesthetic_threshold"],
            "aesthetic_r": calibration["aesthetic_r"],
            "calibration_window_size": calibration["window_size"],
            "calibrated_at": calibration["calibrated_at"],
            "retries": retries,
            "approved": approved,
            "output_dir": run_dir,
            "image_path": image_path,
            "metadata_path": metadata_path,
            "elapsed_sec": elapsed_sec,
            "error": "",
        }

    except Exception as e:
        elapsed_sec = round(time.perf_counter() - started, 3)
        return {
            "benchmark_run_id": benchmark_run_id,
            "timestamp": ts,
            "prompt_id": prompt_id,
            "prompt_language": language,
            "provider": provider,
            "model": model,
            "raw_prompt": csv_safe_text(raw_prompt),
            "enhanced_prompt": "",
            "final_prompt": "",
            "style": "",
            "size": size,
            "technical_ok": "",
            "scoring_ok": "",
            "scoring_error": "",
            "clip_score": "",
            "clip_threshold": calibration["clip_threshold"],
            "clip_q": calibration["clip_q"],
            "aesthetic_score": "",
            "aesthetic_threshold": calibration["aesthetic_threshold"],
            "aesthetic_r": calibration["aesthetic_r"],
            "calibration_window_size": calibration["window_size"],
            "calibrated_at": calibration["calibrated_at"],
            "retries": "",
            "approved": "",
            "output_dir": "",
            "image_path": "",
            "metadata_path": "",
            "elapsed_sec": elapsed_sec,
            "error": f"{type(e).__name__}: {e}",
        }


def main() -> int:
    load_dotenv()
    args = parse_args()

    prompts = load_prompts(args.prompts_file)
    calibration = load_calibration(args.calibration_path)

    analyzer = ImageAnalyzer()
    retriever = StyleRetriever()

    benchmark_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    init_csv(args.out_csv)
    ensure_dir(args.out_dir)

    # Provider/model list for your comparison
    targets = [
        {"provider": "openai", "model": "gpt-image-1-mini"},
        {"provider": "freepik", "model": "mystic"},
        {"provider": "google", "model": "gemini-3.1-flash-image-preview"},
    ]

    total = len(prompts) * len(targets)
    done = 0

    print(f"[INFO] prompts: {len(prompts)}")
    print(f"[INFO] targets: {targets}")
    print(f"[INFO] total runs: {total}")
    print(f"[INFO] csv: {args.out_csv}")

    for target in targets:
        provider = target["provider"]
        model = target["model"]

        for item in prompts:
            done += 1
            print(f"[{done}/{total}] provider={provider} model={model} lang={item['language']} id={item['prompt_id']}")

            row = run_one(
                benchmark_run_id=benchmark_run_id,
                prompt_id=item["prompt_id"],
                language=item["language"],
                raw_prompt=item["prompt"],
                provider=provider,
                model=model,
                size=args.size,
                calibration=calibration,
                analyzer=analyzer,
                retriever=retriever,
                out_dir=args.out_dir,
                poll_sec=args.poll_sec,
                poll_timeout_sec=args.poll_timeout_sec,
                max_retries=args.max_retries,
            )
            append_csv(args.out_csv, row)

            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

    print("[INFO] benchmark complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())