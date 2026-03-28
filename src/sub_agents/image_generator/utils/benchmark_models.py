from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import requests
from dotenv import load_dotenv, find_dotenv

# Load .env reliably
load_dotenv(find_dotenv(), override=True)


# ---------------------------------------------------
# Utilities
# ---------------------------------------------------

def utc_now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_prompts(path: str) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [x.strip() for x in lines if x.strip()]


def parse_size(size: str) -> Tuple[int, int]:
    w, h = size.lower().split("x")
    return int(w), int(h)


def strengthen_prompt(prompt: str) -> str:
    return prompt + "\nphotorealistic, high detail, sharp focus\nNegative: blurry, low resolution, watermark, text, logo"


def is_png_or_jpg(image_bytes: bytes) -> bool:
    if not image_bytes or len(image_bytes) < 16:
        return False
    if image_bytes.startswith(b"\x89PNG"):
        return True
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return True
    return False


def short_snip(b: bytes, n: int = 240) -> str:
    try:
        return b[:n].decode("utf-8", errors="replace")
    except Exception:
        return repr(b[:n])


def log_err(msg: str) -> None:
    print(f"[ERR] {msg}")


def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


# ---------------------------------------------------
# SCORING (subprocess)
# ---------------------------------------------------

def score_image(prompt: str, image_bytes: bytes) -> Dict[str, Any]:
    """
    Calls scorer:
      python -m src.sub_agents.image_generator.utils.score_image

    Expected JSON:
      {"ok": true, "clip_score": ..., "aesthetic_score": ...}
      or {"ok": false, "error": "..."}
    """
    timeout = int(os.getenv("SCORING_TIMEOUT_SEC", "180"))
    py = os.getenv("PYTHON_EXECUTABLE", sys.executable)

    with tempfile.TemporaryDirectory() as td:
        img_path = Path(td) / "img.png"
        out_json = Path(td) / "scores.json"

        img_path.write_bytes(image_bytes)

        cmd = [
            py,
            "-m",
            "src.sub_agents.image_generator.utils.score_image",
            "--image_path",
            str(img_path),
            "--prompt",
            prompt,
            "--out_json",
            str(out_json),
        ]

        try:
            cp = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except Exception as e:
            return {"ok": False, "error": f"scorer_subprocess_exception {type(e).__name__}: {e}"}

        if not out_json.exists():
            return {
                "ok": False,
                "error": f"scorer_no_output_json rc={cp.returncode} stderr={cp.stderr[-400:]} stdout={cp.stdout[-200:]}",
            }

        try:
            data = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception as e:
            return {"ok": False, "error": f"scorer_bad_json {type(e).__name__}: {e}"}

        if not data.get("ok"):
            err = str(data.get("error", "scoring_failed"))
            if cp.stderr:
                err += f" | scorer_stderr_tail={cp.stderr[-250:]}"
            data["error"] = err
        return data


# ---------------------------------------------------
# Providers
# ---------------------------------------------------

def gen_freepik(prompt: str, size: str, style: str) -> Tuple[bytes, Dict[str, Any]]:
    """
    Freepik Classic fast text-to-image:
      POST https://api.freepik.com/v1/ai/text-to-image
      header: x-freepik-api-key
      body: prompt, negative_prompt, guidance_scale, seed, num_images, image.size, styling.style, filter_nsfw
    Response: data[].base64 (PNG) :contentReference[oaicite:4]{index=4}
    """
    api_key = (os.getenv("FREEPIK_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("FREEPIK_API_KEY missing")

    url = (os.getenv("FREEPIK_T2I_URL") or "https://api.freepik.com/v1/ai/text-to-image").strip()

    w, h = parse_size(size)

    # Freepik uses named sizes like square_1_1 / portrait_4_5 etc.
    # We'll map common ratios; fall back to square_1_1.
    if w == h:
        size_name = "square_1_1"
    elif abs((w / h) - (4 / 5)) < 0.05:
        size_name = "portrait_4_5"
    elif abs((w / h) - (16 / 9)) < 0.05:
        size_name = "landscape_16_9"
    else:
        size_name = "square_1_1"

    guidance = float(os.getenv("FREEPIK_GUIDANCE_SCALE", "1.2"))
    num_steps = int(os.getenv("FREEPIK_NUM_STEPS", "8"))
    filter_nsfw = (os.getenv("FREEPIK_FILTER_NSFW", "true").strip().lower() == "true")

    # Minimal negative prompt (optional)
    negative = "blurry, low resolution, watermark, text, logo"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-freepik-api-key": api_key,
    }

    allowed = {s.strip().lower() for s in (os.getenv("FREEPIK_ALLOWED_STYLES", "anime,3d").split(",")) if s.strip()}
    style_clean = (style or "").strip().lower()

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative,
        "guidance_scale": guidance,
        "num_inference_steps": num_steps,
        "num_images": 1,
        "image": {"size": size_name},
        "filter_nsfw": filter_nsfw,
    }

    allowed = {s.strip().lower() for s in (os.getenv("FREEPIK_ALLOWED_STYLES", "anime,3d").split(",")) if s.strip()}
    style_clean = (style or "").strip().lower()

    r = requests.post(url, headers=headers, json=payload, timeout=180)
    if r.status_code >= 400:
        log_err(f"Freepik error {r.status_code}: {short_snip(r.content)}")
    r.raise_for_status()

    data = r.json()
    b64 = data["data"][0]["base64"]
    img_bytes = base64.b64decode(b64)

    # Freepik returns PNG bytes (base64). :contentReference[oaicite:5]{index=5}
    return img_bytes, {"provider": "freepik", "model": url, "meta": data.get("meta", {})}


def gen_stability(prompt: str, size: str, engine: str) -> Tuple[bytes, Dict[str, Any]]:
    api_key = os.getenv("STABILITY_API_KEY")
    if not api_key:
        raise ValueError("STABILITY_API_KEY missing")

    if not engine:
        raise ValueError("Stability engine missing (check STABILITY_MODEL_A / STABILITY_MODEL_B)")

    w, h = parse_size(size)

    url = f"https://api.stability.ai/v1/generation/{engine}/text-to-image"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "text_prompts": [{"text": prompt}],
        "width": w,
        "height": h,
        "samples": 1,
        "cfg_scale": 7,
        "steps": 30,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=180)
    if r.status_code >= 400:
        log_err(f"Stability v1 error {r.status_code}: {short_snip(r.content)}")
    r.raise_for_status()

    data = r.json()
    img = base64.b64decode(data["artifacts"][0]["base64"])
    return img, {"provider": "stability", "model": engine, "api": "v1"}


def gen_openai(prompt: str, size: str) -> Tuple[bytes, Dict[str, Any]]:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY missing")

    model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1-mini")
    client = OpenAI(api_key=api_key)

    resp = client.images.generate(model=model, prompt=prompt, size=size)
    data0 = resp.data[0]

    if getattr(data0, "b64_json", None):
        img = base64.b64decode(data0.b64_json)
        return img, {"provider": "openai", "model": model}

    if getattr(data0, "url", None):
        r = requests.get(data0.url, timeout=60)
        r.raise_for_status()
        return r.content, {"provider": "openai", "model": model, "url": data0.url}

    raise RuntimeError("OpenAI response missing b64_json/url")


# ---------------------------------------------------
# Provider Router
# ---------------------------------------------------

def generate(provider: str, prompt: str, size: str, style: str) -> Tuple[bytes, Dict[str, Any]]:
    provider = provider.lower().strip()

    if provider == "freepik":
        return gen_freepik(prompt, size, style=style)

    if provider == "openai":
        return gen_openai(prompt, size)

    if provider == "stability_a":
        return gen_stability(prompt, size, engine=os.getenv("STABILITY_MODEL_A", "").strip())

    if provider == "stability_b":
        return gen_stability(prompt, size, engine=os.getenv("STABILITY_MODEL_B", "").strip())

    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------
# CSV schema (FULL, fixed order)
# ---------------------------------------------------

CSV_HEADER = [
    "run_id",
    "timestamp_utc",
    "provider",
    "model",
    "prompt_id",
    "raw_prompt",
    "style",
    "size",
    "final_prompt",
    "latency_sec",
    "retries",
    "technical_ok",
    "scoring_ok",
    "scoring_error",
    "clip_score",
    "clip_threshold",
    "clip_q",
    "aesthetic_score",
    "aesthetic_threshold",
    "aesthetic_r",
    "passed",
    "image_path",
    "metadata_path",
]


def csv_append_row(csv_path: Path, header: List[str], row: Dict[str, Any]) -> None:
    ensure_dir(csv_path.parent)
    exists = csv_path.exists()

    if exists:
        with csv_path.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
        if first != ",".join(header):
            raise RuntimeError(
                f"CSV header mismatch in {csv_path}.\n"
                f"Fix: Use a new --out_csv or delete the existing file.\n"
                f"Expected header len={len(header)} got: {first}"
            )

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


# ---------------------------------------------------
# Benchmark runner
# ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--out_csv", default="exports/image_benchmark_runs.csv")
    parser.add_argument("--out_dir", default="exports/benchmark_images")
    parser.add_argument("--providers", default="freepik,stability_a,openai")
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--styles", default="photorealistic,anime,3d,illustration")
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--calibration", default="calibration/image_thresholds.json")

    args = parser.parse_args()

    prompts = read_prompts(args.prompts_file)
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]

    thresholds = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    clip_t = float(thresholds["clip_threshold"])
    aes_t = float(thresholds["aesthetic_threshold"])
    clip_q = thresholds.get("clip_q", "")
    aes_r = thresholds.get("aesthetic_r", "")

    out_csv = Path(args.out_csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    log_info(f"scoring python: {os.getenv('PYTHON_EXECUTABLE', sys.executable)}")
    log_info(f"freepik url: {os.getenv('FREEPIK_T2I_URL', 'https://api.freepik.com/v1/ai/text-to-image')}")

    for i, raw_prompt in enumerate(prompts, start=1):
        style = styles[(i - 1) % len(styles)] if styles else (os.getenv("FREEPIK_STYLE_DEFAULT", "photorealistic"))

        base_prompt = f"{raw_prompt}\n{style}"
        final_prompt_used = base_prompt

        for provider in providers:
            run_id = uuid.uuid4().hex[:12]

            retries = 0
            scoring_ok = False
            scoring_error = ""
            clip = 0.0
            aes = 0.0
            tech_ok = False
            passed = False

            chosen_img: Optional[bytes] = None
            chosen_meta: Dict[str, Any] = {}
            used_prompt_for_image = base_prompt

            t0 = time.perf_counter()

            for attempt in range(args.max_retries + 1):
                used_prompt_for_image = base_prompt if attempt == 0 else strengthen_prompt(base_prompt)

                # --- generation ---
                try:
                    img, meta = generate(provider, used_prompt_for_image, args.size, style=style)
                    chosen_img = img
                    chosen_meta = meta
                except Exception as e:
                    scoring_ok = False
                    scoring_error = f"generation_failed: {type(e).__name__}: {e}"
                    log_err(f"{provider} prompt {i}: {scoring_error}")
                    break

                # --- validate bytes are actual image ---
                tech_ok = is_png_or_jpg(chosen_img)
                if not tech_ok:
                    scoring_ok = False
                    clip = 0.0
                    aes = 0.0
                    passed = False
                    scoring_error = f"non_image_response first_bytes={chosen_img[:12]!r} snip={short_snip(chosen_img)}"
                    log_err(f"{provider} prompt {i}: {scoring_error}")
                else:
                    # --- scoring ---
                    scores = score_image(used_prompt_for_image, chosen_img)
                    scoring_ok = bool(scores.get("ok"))
                    if scoring_ok:
                        clip = float(scores.get("clip_score", 0.0))
                        aes = float(scores.get("aesthetic_score", 0.0))
                        scoring_error = ""
                    else:
                        clip = 0.0
                        aes = 0.0
                        scoring_error = str(scores.get("error", "scoring_failed"))
                        log_err(f"{provider} prompt {i}: scoring_failed: {scoring_error}")

                    passed = scoring_ok and (clip >= clip_t) and (aes >= aes_t)

                if passed:
                    break

                if attempt < args.max_retries:
                    retries += 1
                    continue

            latency = time.perf_counter() - t0
            final_prompt_used = used_prompt_for_image

            # Export image + metadata
            image_path = ""
            metadata_path = ""
            if chosen_img is not None:
                fname_base = f"{run_id}_{provider}"
                img_file = out_dir / f"{fname_base}.png"
                meta_file = out_dir / f"{fname_base}.json"

                if is_png_or_jpg(chosen_img):
                    img_file.write_bytes(chosen_img)
                    image_path = str(img_file)

                meta_payload = {
                    "run_id": run_id,
                    "timestamp_utc": utc_now_utc_str(),
                    "provider": provider,
                    "model": chosen_meta.get("model", ""),
                    "prompt_id": i,
                    "raw_prompt": raw_prompt,
                    "style": style,
                    "size": args.size,
                    "final_prompt": final_prompt_used,
                    "latency_sec": latency,
                    "retries": retries,
                    "technical_ok": tech_ok,
                    "scoring_ok": scoring_ok,
                    "scoring_error": scoring_error,
                    "clip_score": clip,
                    "aesthetic_score": aes,
                    "thresholds": {
                        "clip_threshold": clip_t,
                        "clip_q": clip_q,
                        "aesthetic_threshold": aes_t,
                        "aesthetic_r": aes_r,
                        "calibration_path": args.calibration,
                    },
                    "passed": passed,
                    "generation_meta": chosen_meta,
                    "debug": {
                        "python_executable": os.getenv("PYTHON_EXECUTABLE", sys.executable),
                        "image_magic_ok": is_png_or_jpg(chosen_img),
                        "first_bytes": repr(chosen_img[:12]),
                    },
                }
                meta_file.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                metadata_path = str(meta_file)

            row = {
                "run_id": run_id,
                "timestamp_utc": utc_now_utc_str(),
                "provider": provider,
                "model": chosen_meta.get("model", ""),
                "prompt_id": i,
                "raw_prompt": raw_prompt,
                "style": style,
                "size": args.size,
                "final_prompt": final_prompt_used,
                "latency_sec": f"{latency:.3f}",
                "retries": retries,
                "technical_ok": str(bool(tech_ok)).upper(),
                "scoring_ok": str(bool(scoring_ok)).upper(),
                "scoring_error": scoring_error,
                "clip_score": f"{clip:.6f}",
                "clip_threshold": f"{clip_t:.6f}",
                "clip_q": "" if clip_q == "" else str(clip_q),
                "aesthetic_score": f"{aes:.6f}",
                "aesthetic_threshold": f"{aes_t:.6f}",
                "aesthetic_r": "" if aes_r == "" else str(aes_r),
                "passed": str(bool(passed)).upper(),
                "image_path": image_path,
                "metadata_path": metadata_path,
            }
            csv_append_row(out_csv, CSV_HEADER, row)

            print(
                f"{provider} | prompt {i}/{len(prompts)} | tech_ok={tech_ok} scoring_ok={scoring_ok} "
                f"| clip={clip:.3f} aes={aes:.2f} | passed={passed}"
            )


if __name__ == "__main__":
    main()