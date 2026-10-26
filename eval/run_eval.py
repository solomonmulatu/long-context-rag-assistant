import json
import os
from pathlib import Path

import requests

API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
EVAL_DIR = Path(__file__).parent
QUESTIONS_FILE = EVAL_DIR / "questions.json"


def upload_document(pdf_path: Path) -> str:
    with pdf_path.open("rb") as pdf_file:
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": (pdf_path.name, pdf_file, "application/pdf")},
            timeout=300,
        )

    response.raise_for_status()
    return response.json()["doc_id"]


def main() -> None:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    uploaded_documents: dict[Path, str] = {}

    for item in questions:
        doc_id = None

        # Upload each distinct test PDF only once during this run.
        if item.get("document"):
            pdf_path = (EVAL_DIR / item["document"]).resolve()

            if not pdf_path.is_file():
                raise FileNotFoundError(f"Evaluation PDF not found: {pdf_path}")

            if pdf_path not in uploaded_documents:
                print(f"Uploading evaluation document: {pdf_path.name}")
                uploaded_documents[pdf_path] = upload_document(pdf_path)

            doc_id = uploaded_documents[pdf_path]

        params = {"query": item["question"]}
        if doc_id:
            params["doc_id"] = doc_id

        response = requests.get(
            f"{API_URL}/query",
            params=params,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

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