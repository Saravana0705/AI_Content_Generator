from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test one Gemini image generation via LLM API.")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--model", type=str, default="gemini-3.1-flash-image-preview")
    p.add_argument("--size", type=str, default="1024x1024")
    p.add_argument("--out_dir", type=str, default="exports/llmapi_gemini_test")
    p.add_argument("--base_url", type=str, default="https://api.llmapi.ai/v1")
    return p.parse_args()


def size_to_aspect_ratio(size: str) -> str:
    if size == "1024x1024":
        return "1:1"
    if size == "1024x1536":
        return "2:3"
    if size == "1536x1024":
        return "3:2"
    return "1:1"


def size_to_image_size(size: str) -> str:
    # LLM API docs for Google models show image_size values like 1K / 2K / 4K.
    return "1K"


def extract_image_bytes(resp_json: dict) -> bytes:
    choices = resp_json.get("choices") or []
    if not choices:
        raise ValueError("No choices in response")

    message = (choices[0] or {}).get("message") or {}
    images = message.get("images") or []
    if not images:
        raise ValueError("No images found in choices[0].message.images")

    img0 = images[0] or {}
    image_url = img0.get("image_url") or {}
    url = image_url.get("url")
    if not url:
        raise ValueError("No image_url.url found in first image")

    prefix = "data:image/png;base64,"
    if url.startswith(prefix):
        return base64.b64decode(url[len(prefix):])

    # fallback if another data-url mime type appears
    if url.startswith("data:image/") and ";base64," in url:
        b64 = url.split(";base64,", 1)[1]
        return base64.b64decode(b64)

    raise ValueError("Image URL is not a base64 data URL")


def main() -> int:
    args = parse_args()

    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("LLM_API_KEY missing in environment")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": args.prompt,
            }
        ],
        "image_config": {
            "aspect_ratio": size_to_aspect_ratio(args.size),
            "image_size": size_to_image_size(args.size),
        },
    }

    print(f"[INFO] POST {url}")
    print(f"[INFO] model={args.model}")
    print(f"[INFO] size={args.size}")

    r = requests.post(url, headers=headers, json=payload, timeout=180)

    raw_response_path = out_dir / "raw_response.json"
    try:
        resp_json = r.json()
        raw_response_path.write_text(
            json.dumps(resp_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        raw_response_path.write_text(r.text, encoding="utf-8")
        resp_json = None

    print(f"[INFO] status_code={r.status_code}")
    print(f"[INFO] raw response saved to {raw_response_path}")

    if r.status_code >= 400:
        print("[ERROR] request failed")
        print(r.text[:1500])
        return 1

    if not isinstance(resp_json, dict):
        print("[ERROR] response was not JSON")
        return 1

    try:
        image_bytes = extract_image_bytes(resp_json)
    except Exception as e:
        print(f"[ERROR] could not extract image: {type(e).__name__}: {e}")
        print(json.dumps(resp_json, ensure_ascii=False, indent=2)[:2000])
        return 1

    image_path = out_dir / "image.png"
    image_path.write_bytes(image_bytes)
    print(f"[INFO] image saved to {image_path}")

    scores_path = out_dir / "scores.json"
    cmd = [
        sys.executable,
        "-m",
        "src.sub_agents.image_generator.utils.score_image",
        "--image_path",
        str(image_path),
        "--prompt",
        args.prompt,
        "--out_json",
        str(scores_path),
    ]
    print("[INFO] running scorer...")
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=240, check=False)

    print(f"[INFO] scorer rc={cp.returncode}")
    if cp.stdout.strip():
        print("[STDOUT]")
        print(cp.stdout[-1000:])
    if cp.stderr.strip():
        print("[STDERR]")
        print(cp.stderr[-1000:])

    if scores_path.exists():
        print(f"[INFO] scores saved to {scores_path}")
        print(scores_path.read_text(encoding="utf-8"))
    else:
        print("[ERROR] scorer did not produce scores.json")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())