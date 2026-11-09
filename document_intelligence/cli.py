from __future__ import annotations

import argparse
from pathlib import Path

from .model import DocumentIntelligenceModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a cited question over sample documents")
    parser.add_argument("question")
    parser.add_argument("--documents", default=str(Path(__file__).parents[1] / "data" / "documents"))
    args = parser.parse_args()

    result = DocumentIntelligenceModel().fit_directory(args.documents).ask(args.question)
    print(f"Answer: {result.answer}")
    for citation in result.citations:
        print(
            f"Citation: {citation.source} > {citation.section} "
            f"[chunk {citation.chunk_id}, score={citation.score}]"
        )


if __name__ == "__main__":
    main()

