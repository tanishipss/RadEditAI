"""
List models available to this Groq account/key — run this whenever a pinned model string starts
404ing, instead of guessing a replacement.

Run from the radiology_pipeline/ directory:
    python scripts/list_groq_models.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groq_client import _get_client


def main() -> None:
    client = _get_client()
    models = client.models.list().data
    print(f"{len(models)} models available:\n")
    for m in sorted(models, key=lambda m: m.id):
        print(f"  {m.id}")


if __name__ == "__main__":
    main()
