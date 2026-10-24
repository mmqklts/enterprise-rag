# Enterprise Knowledge Agent

基于 FastAPI 的企业知识库 Agent，支持 PDF、Markdown 和 TXT 文档入库。模型会根据问题动态调用混合检索、原文读取和文档列表工具，经过多步推理后返回带来源引用的答案。

## 功能

- PDF、Markdown、TXT 文档解析
- 文本切分与重叠窗口
- 本地 BGE 中文 Embedding
- Qdrant 向量存储与相似度检索
- BM25 与向量检索混合召回
- RRF 排名融合
- BGE CrossEncoder Rerank
- DeepSeek 答案生成
- 多步 Agent 工具调用
- 检索、原文读取和文档列表工具
- 工具执行轨迹与引用校验
- 最大步数和重复调用保护
- 文件名、页码和原文来源引用
- 检索策略评测：Recall@1、Recall@5、MRR
- FastAPI 接口和 Web 页面

## Agent 流程

```text
用户问题
   |
   v
Agent 分析问题
   |
   +--> search_knowledge_base
   |      Vector + BM25 + RRF + Rerank
   |
   +--> read_source_chunk
   |      读取完整来源 chunk
   |
   +--> list_documents
   |      查看知识库文档范围
   |
   v
根据工具结果决定下一步
   |
   v
最终答案 + 引用校验 + 执行轨迹
```

## Docker 部署

确保已安装 Docker Desktop 和 WSL2。

复制环境变量：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填写 `DEEPSEEK_API_KEY`，然后启动：

```powershell
docker compose up -d --build
```

打开 `http://127.0.0.1:8000/`。

## 本地开发

```powershell
uv sync
uv run uvicorn enterprise_rag.main:app --reload
```

## Agent 接口

`POST /agent/chat`

```json
{
  "question": "对比两份制度中关于报销流程的差异",
  "limit": 5
}
```

响应包含：

- `answer`：最终答案。
- `sources`：答案实际引用的来源。
- `trace`：每一步工具名称、参数、状态和耗时。
- `usage`：LLM 请求次数与 Token 用量。
- `citation_status`：`ok`、`missing` 或 `invalid`。
- `stop_reason`：结束原因，例如 `completed`、`max_steps` 或 `no_evidence`。

旧的固定 RAG 接口 `POST /chat` 继续保留，用于对比固定工作流与 Agent 的行为。

## Agent 限制

- 默认最多执行 6 个模型步骤，可通过 `AGENT_MAX_STEPS` 调整到 1-12。
- 相同工具和参数不会重复执行。
- 单次检索最多返回 8 条来源。
- 工具观测结果超过 16000 字符会被截断。
- 最终引用编号必须对应 Agent 实际检索到的来源。

## 测试

```powershell
uv run pytest
uv run ruff check .
```

## 检索评测

```powershell
uv run python -m enterprise_rag.evaluation
```
