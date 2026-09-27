"""Quick test of the hosted Laya API (api.impossibl.com).

Usage:
    Put IMPOSSIBL_API_KEY=your-key in a .env file next to this script,
    or set it in the shell:  $env:IMPOSSIBL_API_KEY = "your-key"  (PowerShell)

    python test_laya.py
    python test_laya.py "Some other customer message to evaluate"

No third-party packages needed; uses only the standard library.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

API_URL = "https://api.impossibl.com/v1/systemone"
MODEL = "convaiinnovations/laya"  # or "convaiinnovations/laya-multilingual"

DEFAULT_TEXT = (
    "Hi, we were billed twice for March. "
    "Refund the duplicate today or we cancel."
)

# One of each question type so you can see all three response shapes.
QUESTIONS = {
    "churn_risk": {
        "type": "noul",
        "instructions": "Is this customer likely to churn?",
    },
    "sentiment": {
        "type": "choice",
        "instructions": "What is the overall tone of this message?",
        "criteria": {
            "angry": "The customer is upset, frustrated, or threatening",
            "neutral": "Plain, factual, no strong emotion",
            "happy": "The customer is pleased or thankful",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": [
            "low: no time pressure",
            "medium: should be handled within a few days",
            "high: needs attention today",
            "critical: immediate action or the relationship is at risk",
        ],
    },
}


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader: KEY=value lines, no dependency on python-dotenv."""
    here = os.path.dirname(os.path.abspath(__file__))
    full = os.path.join(here, path)
    if not os.path.exists(full):
        return
    with open(full, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def call_laya(text: str, api_key: str) -> dict:
    payload = {"model": MODEL, "state": text, "questions": QUESTIONS}
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    load_dotenv()
    api_key = os.environ.get("IMPOSSIBL_API_KEY")
    if not api_key:
        print("ERROR: IMPOSSIBL_API_KEY is not set.")
        print('PowerShell:  $env:IMPOSSIBL_API_KEY = "your-key"')
        return 1

    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TEXT

    print("=== Request ===")
    print(f"Model : {MODEL}")
    print(f"State : {text}")
    print(f"Questions: {', '.join(QUESTIONS)}")
    print()

    start = time.perf_counter()
    try:
        result = call_laya(text, api_key)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code} {e.reason}")
        print(body)
        return 1
    except urllib.error.URLError as e:
        print(f"Network error: {e.reason}")
        return 1
    elapsed_ms = (time.perf_counter() - start) * 1000

    print("=== Raw response ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()

    # Best-effort summary; the exact response shape may differ slightly.
    answers = result.get("answers") or result.get("results") or result
    print("=== Summary ===")
    for name in QUESTIONS:
        ans = answers.get(name) if isinstance(answers, dict) else None
        print(f"{name:12s} -> {json.dumps(ans, ensure_ascii=False)}")
    usage = result.get("usage")
    if usage:
        print(f"usage        -> {usage}")
    print(f"latency      -> {elapsed_ms:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
