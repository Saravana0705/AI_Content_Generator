import os
import json
import time

from src.main_agent.supervisor import Supervisor
from metrics.logger import append_run


PROMPTS_FILE = "runs/prompts_20.json"

# Cloud models (OpenAI + Groq)
CLOUD_MODELS = [
    "gpt-4o",                      # OpenAI
    "llama-3.3-70b-versatile",     # Groq
    "moonshotai/kimi-k2-instruct", # Groq
    "qwen/qwen3-32b",              # Groq
]


def load_prompts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def provider_for_model(model_name: str) -> str:
    # simple rule: all gpt-* -> openai, else -> groq
    return "openai" if model_name.startswith("gpt-") else "groq"


def get_review_threshold() -> float:
    # keep constant across models for fairness
    return 60.0


def main():
    prompts = load_prompts(PROMPTS_FILE)

    # Keep constants for fair comparison
    content_type = "blog_article"
    tone = "neutral"
    keywords = []
    repeats = int(os.getenv("REPEATS", "1"))
    review_threshold = get_review_threshold()

    # Optional: allow selecting only first N prompts (defaults to all prompts in file)
    max_prompts = os.getenv("MAX_PROMPTS")
    if max_prompts:
        prompts = prompts[: int(max_prompts)]

    print("=== Cloud Benchmark Run ===")
    print(f"Prompts file : {PROMPTS_FILE}")
    print(f"Prompts count: {len(prompts)}")
    print(f"Models       : {CLOUD_MODELS}")
    print(f"Repeats      : {repeats}")
    print(f"Threshold    : {review_threshold}")
    print("===========================")

    # We intentionally create Supervisor inside the loop AFTER setting LLM_PROVIDER,
    # because your pipeline initializes provider-specific clients during construction.
    for r in range(repeats):
        for item in prompts:
            pid = str(item.get("id", "")).strip()
            prompt = str(item.get("prompt", "")).strip()
            if not prompt:
                continue

            for model_name in CLOUD_MODELS:
                provider = provider_for_model(model_name)
                os.environ["LLM_PROVIDER"] = provider

                supervisor = Supervisor(review_threshold=review_threshold)

                print(f"Running id={pid} provider={provider} model={model_name} repeat={r+1}")

                t0 = time.perf_counter()
                try:
                    result = supervisor.coordinate_workflow(
                        prompt,
                        subagent_type="text_generator",
                        content_type=content_type,
                        tone=tone,
                        keywords=keywords,
                        model_name=model_name,
                    )
                except Exception as e:
                    print(f"Failed id={pid} provider={provider} model={model_name}: {e}")
                    continue

                total_time_sec = time.perf_counter() - t0

                final_text = result.get("optimized_text") or ""
                optimized_score = float(result.get("score", 0.0) or 0.0)

                revision_rounds = int(result.get("revision_rounds", 0) or 0)
                initial_score = result.get("initial_score", None)
                revised_score = result.get("revised_score", None)
                score_improvement = result.get("score_improvement", None)

                # Only keep improvement metrics when revision happened
                if revision_rounds <= 0:
                    initial_score = None
                    revised_score = None
                    score_improvement = None

                append_run(
                    prompt=prompt,
                    model_name=model_name,
                    content_type=content_type,
                    tone=tone,
                    total_time_sec=total_time_sec,
                    final_text=final_text,
                    optimized_score=optimized_score,
                    revision_rounds=revision_rounds,
                    initial_score=initial_score,
                    revised_score=revised_score,
                    score_improvement=score_improvement,
                )

    print("Benchmark complete. Results appended to runs/runs_log.csv")


if __name__ == "__main__":
    main()
