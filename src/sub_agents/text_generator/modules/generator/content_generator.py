import os
from dotenv import load_dotenv

# Load .env for local development (safe in Docker; no-op if .env not present)
load_dotenv()

# Import OpenAI client once (top-level)
try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None


class Generator:
    """
    Multi-provider TEXT generator (Cloud APIs only).

    Supported providers:
      - openai
      - groq (OpenAI-compatible endpoint)

    Env vars:
      - TEXT_PROVIDER: "openai" (default) or "groq"
      - OPENAI_API_KEY: required if TEXT_PROVIDER=openai
      - GROQ_API_KEY: required if TEXT_PROVIDER=groq
      - OPENAI_TEXT_MODEL: default OpenAI text model if model_name not passed
      - GROQ_TEXT_MODEL: default Groq text model if model_name not passed
    """

    def __init__(self):
        self.provider = (os.getenv("TEXT_PROVIDER") or "openai").strip().lower()

        # Defaults for TEXT generator only
        self.default_openai_model = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o")
        self.default_groq_model = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")

        if self.provider not in ("openai", "groq"):
            raise ValueError(
                f"Unsupported TEXT_PROVIDER '{self.provider}'. Use 'openai' or 'groq'."
            )

        if OpenAIClient is None:
            raise ImportError("openai package not available. Run: pip install openai")

        self._client = None

        if self.provider == "openai":
            api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found (required for TEXT_PROVIDER=openai)")
            self._client = OpenAIClient(api_key=api_key)

        elif self.provider == "groq":
            api_key = (os.getenv("GROQ_API_KEY") or "").strip()
            if not api_key:
                raise ValueError("GROQ_API_KEY not found (required for TEXT_PROVIDER=groq)")
            self._client = OpenAIClient(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1",
            )

    def _normalize_model(self, model_name: str | None) -> str:
        """
        Normalize model_name to avoid legacy values like 'gpt'
        and to provide provider-specific defaults.
        """
        mn = (model_name or "").strip()
        if not mn or mn.lower() == "gpt":
            if self.provider == "openai":
                return self.default_openai_model
            if self.provider == "groq":
                return self.default_groq_model
        return mn

    def generate(self, input_text: str, model_name: str | None = None) -> str:
        model = self._normalize_model(model_name)

        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": input_text}],
            )
            return resp.choices[0].message.content

        except Exception as e:
            raise RuntimeError(
                f"LLM generation failed (provider={self.provider}, model={model}): {e}"
            ) from e