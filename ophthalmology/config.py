from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class OphthalmologyConfig:
    text_dir: Path = Path("text")
    artifacts_dir: Path = Path("artifacts/ophthalmology")
    chunk_size: int = 1200
    chunk_overlap: int = 150
    embedding_model: str = "text-embedding-3-large"
    extraction_model: str = "gpt-4o-mini"
    diagnosis_model: str = "gpt-4o"

    @property
    def entities_path(self) -> Path:
        return self.artifacts_dir / "entities.jsonl"

    @property
    def triples_path(self) -> Path:
        return self.artifacts_dir / "triples.jsonl"

    @property
    def chunks_path(self) -> Path:
        return self.artifacts_dir / "chunks.jsonl"

    @property
    def embeddings_path(self) -> Path:
        return self.artifacts_dir / "chunk_embeddings.npy"

    @property
    def graph_path(self) -> Path:
        return self.artifacts_dir / "knowledge_graph.json"


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Please export your OpenAI key first.")
    return api_key
