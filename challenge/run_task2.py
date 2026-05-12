"""Script to run Task 2 of the RAG EvalLLM Challenge (Source Attribution)."""

import json
import logging
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from challenge.schemas import AttributedSource, Attribution, ChallengeSubmission
from src.dependencies import get_rag_service
from src.schemas import QueryRequest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_task2(input_file: Path, output_file: Path) -> None:
    logger.info(f"Lecture des données depuis {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    submission = ChallengeSubmission.model_validate(data)
    rag_service = get_rag_service()

    logger.info(f"Traitement de {len(submission.results)} questions pour la Tâche 2...")

    for result in submission.results:
        if not result.question:
            continue

        logger.info(f"Requête QID {result.qid} : {result.question[:60]}...")

        req = QueryRequest(query=result.question, top_k=15)
        response = rag_service.query(req)

        text_response = response.chat_response
        source_docs = response.source_documents

        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", text_response) if s.strip()
        ]

        attributions = []
        for s_idx, sentence in enumerate(sentences):
            citation_indices = [int(idx) for idx in re.findall(r"\[(\d+)\]", sentence)]

            attributed_sources = []
            for idx in citation_indices:
                doc_idx = idx - 1
                if 0 <= doc_idx < len(source_docs):
                    doc = source_docs[doc_idx]
                    attributed_sources.append(
                        AttributedSource(
                            doc_name=doc.metadata.file_name, page=doc.metadata.page
                        )
                    )

            unique_sources = []
            seen = set()
            for source in attributed_sources:
                identifier = f"{source.doc_name}_{source.page}"
                if identifier not in seen:
                    seen.add(identifier)
                    unique_sources.append(source)

            attributions.append(
                Attribution(
                    sid=f"{result.qid}_s{s_idx}",
                    text=sentence,
                    attributed_to=unique_sources,
                )
            )

        result.attributions = attributions

    submission.run_id = "Baseline RAG EvalLLM - Task 2"

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(submission.model_dump_json(indent=2, exclude_none=True))

    logger.info(f"✅ Soumission de la Tâche 2 générée avec succès : {output_file}")


if __name__ == "__main__":
    input_path = Path("data/challenge_input.json")
    output_path = Path("submissions/task2_submission.json")

    if not input_path.exists():
        logger.error(f"Fichier introuvable : {input_path}")
        sys.exit(1)

    process_task2(input_path, output_path)
