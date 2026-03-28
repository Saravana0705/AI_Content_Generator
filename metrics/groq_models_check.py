import os
import requests

BASE_URL = "https://api.groq.com/openai/v1"

def main():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise SystemExit("Missing GROQ_API_KEY")

    r = requests.get(
        f"{BASE_URL}/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    ids = sorted([m["id"] for m in data.get("data", []) if "id" in m])

    print("Active Groq model IDs:")
    for mid in ids:
        print(" -", mid)

if __name__ == "__main__":
    main()
