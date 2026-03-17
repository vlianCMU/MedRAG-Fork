import json
from typing import Dict, Any

from openai import OpenAI

from ophthalmology.config import OphthalmologyConfig, get_openai_api_key
from ophthalmology.retriever import OphthalmologyRetriever


SYSTEM_PROMPT = """
你是眼科问诊助手，需要执行以下流程：
1) 基于输入主诉总结关键信息。
2) 结合检索片段和知识图谱关系，给出推理链。
3) 主动生成追问问题（优先鉴别诊断价值高的问题）。
4) 给出当前最可能诊断、备选诊断与下一步检查建议。
输出JSON，字段如下：
- reasoning
- follow_up_questions
- likely_diagnosis
- differential_diagnosis
- recommended_tests
- confidence
""".strip()


class OphthalmologyConsultationEngine:
    def __init__(self, config: OphthalmologyConfig):
        self.config = config
        self.client = OpenAI(api_key=get_openai_api_key())
        self.retriever = OphthalmologyRetriever(config)
        self.graph = json.loads(config.graph_path.read_text(encoding="utf-8"))

    def _graph_context(self, limit: int = 20) -> str:
        edges = self.graph.get("edges", [])[:limit]
        return "\n".join([f"{e['source']} -[{e['relation']}]-> {e['target']}" for e in edges])

    def consult(self, patient_input: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        query = json.dumps(patient_input, ensure_ascii=False)
        retrieved = self.retriever.search(query=query, top_k=top_k)

        prompt = {
            "patient_input": patient_input,
            "retrieved_chunks": retrieved,
            "graph_relations": self._graph_context(),
        }

        response = self.client.chat.completions.create(
            model=self.config.diagnosis_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        )
        return json.loads(response.choices[0].message.content)
