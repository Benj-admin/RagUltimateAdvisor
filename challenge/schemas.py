"""Pydantic schemas for the RAG EvalLLM Challenge submissions."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# --- Tâche 1 : RAG et Retrieval ---


class RetrievedChunk(BaseModel):
    """Represents a source document retrieved by the system."""

    rank: int
    doc_name: str
    page: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# --- Tâche 2 : Attribution des sources ---


class AttributedSource(BaseModel):
    """Represents a specific document and page attributed to a sentence."""

    doc_name: str
    page: Optional[int] = None


class Attribution(BaseModel):
    """Represents a single sentence of the answer and its sources."""

    sid: str
    text: str
    attributed_to: List[AttributedSource] = Field(default_factory=list)


# --- Structure Globale du Challenge ---


class ChallengeResult(BaseModel):
    """Represents the results for a single question."""

    qid: str

    # Champs Tâche 1
    question: Optional[str] = None
    retrieved: Optional[List[RetrievedChunk]] = None
    answer: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    # Champs Tâche 2
    attributions: Optional[List[Attribution]] = None


class ChallengeSubmission(BaseModel):
    """Represents the complete JSON submission file."""

    run_id: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    results: List[ChallengeResult]
