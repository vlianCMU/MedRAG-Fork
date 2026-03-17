import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from openai import OpenAI

from ophthalmology.config import OphthalmologyConfig, get_openai_api_key


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    text: str


class OphthalmologyKGBuilder:
    def __init__(self, config: OphthalmologyConfig):
        self.config = config
        self.client = OpenAI(api_key=get_openai_api_key())
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def load_and_chunk_txt_files(self) -> List[Chunk]:
        txt_files = sorted(self.config.text_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No txt files found in {self.config.text_dir}.")

        chunks: List[Chunk] = []
        for txt_file in txt_files:
            text = txt_file.read_text(encoding="utf-8")
            start = 0
            idx = 0
            while start < len(text):
                end = min(start + self.config.chunk_size, len(text))
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(Chunk(
                        chunk_id=f"{txt_file.stem}_{idx}",
                        source_file=txt_file.name,
                        text=chunk_text,
                    ))
                if end == len(text):
                    break
                start = end - self.config.chunk_overlap
                idx += 1

        with self.config.chunks_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")
        return chunks

    def extract_entities_and_triples(self, chunks: List[Chunk]) -> List[Dict]:
        schema_hint = (
            "Return strict JSON with keys: entities, triples. "
            "entities is a list of {name, type, description}. "
            "triples is a list of {subject, relation, object, evidence}."
        )

        extracted = []
        with self.config.entities_path.open("w", encoding="utf-8") as ef, self.config.triples_path.open("w", encoding="utf-8") as tf:
            for chunk in chunks:
                prompt = (
                    "你是眼科医学知识图谱构建助手。\n"
                    f"请从以下教材片段中抽取实体和关系：\n{chunk.text}\n\n{schema_hint}"
                )
                resp = self.client.chat.completions.create(
                    model=self.config.extraction_model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": "你负责构建高质量眼科知识图谱。"},
                        {"role": "user", "content": prompt},
                    ],
                )
                payload = json.loads(resp.choices[0].message.content)
                payload["chunk_id"] = chunk.chunk_id
                extracted.append(payload)

                for entity in payload.get("entities", []):
                    ef.write(json.dumps({"chunk_id": chunk.chunk_id, **entity}, ensure_ascii=False) + "\n")
                for triple in payload.get("triples", []):
                    tf.write(json.dumps({"chunk_id": chunk.chunk_id, **triple}, ensure_ascii=False) + "\n")

        return extracted

    def build_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        vectors = []
        for chunk in chunks:
            emb = self.client.embeddings.create(
                model=self.config.embedding_model,
                input=chunk.text,
            )
            vectors.append(emb.data[0].embedding)
        arr = np.array(vectors, dtype=np.float32)
        np.save(self.config.embeddings_path, arr)
        return arr

    def export_graph_json(self) -> Dict:
        nodes = {}
        edges = []
        with self.config.triples_path.open("r", encoding="utf-8") as f:
            for line in f:
                triple = json.loads(line)
                s = triple["subject"]
                o = triple["object"]
                nodes.setdefault(s, {"id": s, "type": "Entity"})
                nodes.setdefault(o, {"id": o, "type": "Entity"})
                edges.append({
                    "source": s,
                    "relation": triple["relation"],
                    "target": o,
                    "evidence": triple.get("evidence", ""),
                    "chunk_id": triple.get("chunk_id", ""),
                })

        graph = {"nodes": list(nodes.values()), "edges": edges}
        self.config.graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        return graph

    def run(self) -> Dict:
        chunks = self.load_and_chunk_txt_files()
        self.extract_entities_and_triples(chunks)
        self.build_embeddings(chunks)
        return self.export_graph_json()
