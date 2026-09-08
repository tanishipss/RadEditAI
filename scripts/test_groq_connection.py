"""
Smoke test for groq_client.call_groq — confirms GROQ_API_KEY and the model string both work.

Run from the radiology_pipeline/ directory:
    python scripts/test_groq_connection.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pydantic import BaseModel

from groq_client import call_groq


class SmokeTestResponse(BaseModel):
    answer: str


def main() -> None:
    result = call_groq(
        system="You are a helpful assistant. Respond only with JSON matching the given schema.",
        user="Reply with the word ok in the 'answer' field.",
        response_model=SmokeTestResponse,
    )
    print("Groq connection OK.")
    print("Response:", result)


if __name__ == "__main__":
    main()
