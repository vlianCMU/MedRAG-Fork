# 眼科问诊系统搭建说明（基于 MedRAG）

本方案实现了一个可落地的端到端流程：
1. 从 `text/*.txt` 眼科教材抽取实体与关系。
2. 生成眼科知识图谱（`knowledge_graph.json`）。
3. 对教材片段建立向量索引（OpenAI Embedding）。
4. 在问诊时执行「检索 + 图谱推理 + 主动追问 + 诊断建议」。

## 目录约定
- 教材输入目录：`text/`
- 产物输出目录：`artifacts/ophthalmology/`

## 第一步：构建知识图谱与向量库
```bash
export OPENAI_API_KEY="你的key"
python scripts/build_ophthalmology_kg.py
```

运行后会生成：
- `artifacts/ophthalmology/chunks.jsonl`
- `artifacts/ophthalmology/entities.jsonl`
- `artifacts/ophthalmology/triples.jsonl`
- `artifacts/ophthalmology/chunk_embeddings.npy`
- `artifacts/ophthalmology/knowledge_graph.json`

## 第二步：运行问诊
```bash
export OPENAI_API_KEY="你的key"
python scripts/run_ophthalmology_consultation.py
```

返回 JSON 字段包括：
- `reasoning`：知识图谱 + 检索证据的推理过程
- `follow_up_questions`：主动追问
- `likely_diagnosis`：最可能诊断
- `differential_diagnosis`：鉴别诊断
- `recommended_tests`：建议检查
- `confidence`：置信度

## 你需要准备的数据
- 将眼科教材 txt 文件放入 `text/`。
- 建议教材按章节拆分，单文件不要过大，利于实体关系抽取质量。

## OpenAI API 在本项目中的作用
- 实体关系抽取：`gpt-4o-mini` + JSON 输出
- 向量化：`text-embedding-3-large`
- 问诊推理与生成：`gpt-4o`

## 后续可扩展方向
- 引入标准医学术语映射（ICD-10 / SNOMED）
- 加入图数据库（Neo4j）实现多跳推理
- 加入多轮对话记忆（追问-回答-再推理）
- 增加结构化病历输入（视力、眼压、裂隙灯结果）
