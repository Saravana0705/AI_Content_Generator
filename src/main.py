import os
from typing import Any, Dict

from main_agent.interface import Interface
from main_agent.supervisor import Supervisor


def _print_text_result(result: Dict[str, Any]) -> None:
    final_text = (
        result.get("optimized_text")
        or result.get("final_text")
        or result.get("text")
        or ""
    )
    score = result.get("score")
    review = result.get("review_result") or result.get("review") or {}

    print("\n--- TEXT RESULT ---")
    if score is not None:
        print(f"Score: {score}")
    if isinstance(review, dict) and review:
        print(f"Approved: {review.get('approved', None)}")
        if review.get("comments"):
            print(f"Review comments: {review.get('comments')}")
    print("\nGenerated content:\n")
    print(final_text.strip() or "[No text returned]")


def _print_image_result(result: Dict[str, Any]) -> None:
    export_paths = result.get("export_paths") or result.get("export_result") or {}
    files = export_paths.get("files") or {}

    print("\n--- IMAGE RESULT ---")
    if files.get("image"):
        print(f"Image saved at: {files['image']}")
    if files.get("metadata"):
        print(f"Metadata saved at: {files['metadata']}")

    # Fallback if your image agent returns bytes/base64 directly
    if not files.get("image"):
        img_bytes = result.get("image_bytes")
        img_b64 = result.get("image_b64") or result.get("image_base64") or result.get("b64")

        if isinstance(img_bytes, (bytes, bytearray)):
            print(f"Image bytes returned (len={len(img_bytes)}) but no exported file path.")
        elif isinstance(img_b64, str) and img_b64.strip():
            print(f"Image base64 returned (len={len(img_b64)}) but no exported file path.")
        else:
            print("No image payload found in result.")


def main():
    """
    CLI runner for both text and image workflows.
    Uses Supervisor as the single entry-point (Router + sub-agents behind it).
    """
    interface = Interface()
    supervisor = Supervisor()

    # Simple CLI choice
    mode = interface.get_user_input("Choose mode: 'text' or 'image'").strip().lower()
    user_prompt = interface.get_user_input("Enter your prompt")

    if mode in ("image", "img"):
        # Minimal image params (maps style through 'tone' to avoid changing Supervisor signature)
        style = interface.get_user_input("Enter image style (photorealistic/anime/3d/illustration) [photorealistic]") \
                         .strip().lower() or "photorealistic"
        size = interface.get_user_input("Enter image size (1024x1024/1024x1536/1536x1024) [1024x1024]") \
                        .strip().lower() or "1024x1024"

        # If your image generator reads size from env or intent, set an env var (optional)
        os.environ["IMAGE_SIZE"] = size

        result = supervisor.coordinate_workflow(
            user_prompt,
            subagent_type="image_generator",
            content_type="image",
            tone=style,       # style passed through tone
            keywords=[],
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
        )

        if isinstance(result, dict):
            _print_image_result(result)
        else:
            print("Unexpected result type from supervisor (expected dict).")

    else:
        # Defaults for text
        content_type = interface.get_user_input("Enter content type key [blog_article]") \
                                .strip().lower() or "blog_article"
        tone = interface.get_user_input("Enter tone [neutral]").strip().lower() or "neutral"
        kw = interface.get_user_input("Enter keywords comma-separated (optional)").strip()
        keywords = [k.strip() for k in kw.split(",") if k.strip()] if kw else []

        result = supervisor.coordinate_workflow(
            user_prompt,
            subagent_type="text_generator",
            content_type=content_type,
            tone=tone,
            keywords=keywords,
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
        )

        if isinstance(result, dict):
            _print_text_result(result)
        else:
            print("Unexpected result type from supervisor (expected dict).")


if __name__ == "__main__":
    main()