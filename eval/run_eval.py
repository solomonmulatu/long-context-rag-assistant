import json
import os
from pathlib import Path

import requests

API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
QUESTIONS_FILE = Path(__file__).with_name("questions.json")


def main() -> None:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    for item in questions:
        response = requests.get(
            f"{API_URL}/query",
            params={"query": item["question"]},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        # This expects your API to return a structured object.
        answer = result.get("answer", "")
        citations = result.get("citations", [])
        abstained = result.get("abstained", False)

        answer_lower = answer.lower()
        points_found = [
            point
            for point in item["expected_points"]
            if all(word in answer_lower for word in point.lower().split()[:4])
        ]

        print(f"\n{item['id']}: {item['question']}")
        print(f"Answer: {answer}")
        print(f"Citations returned: {len(citations)}")
        print(
            f"Expected abstention: {item['expected_abstention']} | "
            f"Actual abstention: {abstained}"
        )
        print(
            f"Expected points found: "
            f"{len(points_found)}/{len(item['expected_points'])}"
        )


if __name__ == "__main__":
    main()