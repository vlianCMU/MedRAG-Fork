import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from openai import OpenAI

from ophthalmology.config import OphthalmologyConfig, get_openai_api_key


class OphthalmologyRetriever:
    def __init__(self, config: OphthalmologyConfig):
        self.config = config
        self.client = OpenAI(api_key=get_openai_api_key())
        self.chunks = self._load_chunks(config.chunks_path)
        self.embeddings = np.load(config.embeddings_path).astype(np.float32)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    @staticmethod
    def _load_chunks(path: Path) -> List[Dict]:
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
        return items

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        q = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=query,
        )
        qv = np.array([q.data[0].embedding], dtype=np.float32)
        _, ids = self.index.search(qv, top_k)
        return [self.chunks[i] for i in ids[0] if i < len(self.chunks)]
