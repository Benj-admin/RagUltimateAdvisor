"""Script to run Task 1 of the RAG EvalLLM Challenge."""

import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from challenge.schemas import ChallengeSubmission, RetrievedChunk
from src.dependencies import get_rag_service
from src.schemas import QueryRequest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_task1(input_file: Path, output_file: Path) -> None:
    logger.info(f"Lecture des données du challenge depuis {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    submission = ChallengeSubmission.model_validate(data)
    rag_service = get_rag_service()

    logger.info(f"Traitement de {len(submission.results)} questions pour la Tâche 1...")

    for result in submission.results:
        if not result.question:
            logger.warning(f"QID {result.qid} ignoré : Aucune question trouvée.")
            continue

        logger.info(f"Requête QID {result.qid} : {result.question[:60]}...")

        req = QueryRequest(query=result.question, top_k=15)
        response = rag_service.query(req)

        result.answer = response.chat_response

        result.retrieved = []
        for i, doc in enumerate(response.source_documents):
            chunk = RetrievedChunk(
                rank=i + 1,
                doc_name=doc.metadata.file_name,
                page=doc.metadata.page,
                metadata={"score": round(doc.score, 4)},
            )
            result.retrieved.append(chunk)

    submission.run_id = "Baseline RAG EvalLLM - Task 1"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(submission.model_dump_json(indent=2, exclude_none=True))

    logger.info(f"✅ Soumission de la Tâche 1 générée avec succès : {output_file}")


if __name__ == "__main__":
    input_path = Path("data/challenge_input.json")
    output_path = Path("submissions/task1_submission.json")

    if not input_path.exists():
        logger.error(f"Fichier introuvable : {input_path}")
        logger.info(
            "Veuillez placer le fichier de test du challenge à cet emplacement."
        )
        sys.exit(1)

    process_task1(input_path, output_path)
